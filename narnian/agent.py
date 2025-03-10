import os
import pickle
import inspect
from .model import Model
from .streams import Stream
from typing_extensions import Self
from .fsm import FiniteStateMachine
from collections.abc import Iterable


class Agent:

    def __init__(self, name: str, model: Model, authority: float):
        """Create a new agent."""

        self.name = name  # name of the agent (keep it unique)
        self.behav = FiniteStateMachine(self)  # FSM that describes the agent's behaviour
        self.model = model  # the PyTorch module that implements the agent
        self.authority = authority  # authority level (right now assuming 0 = student and 1 = teacher)
        self.known_streams = {}  # streams that are known to this agent
        self.known_agents = {}  # other agents that are known to this agent
        self.buffered_streams_offsets = {}  # from buffered-stream hash to offset to apply
        self.print_enabled = True  # if output should be printed to screen
        self.env = None  # environment where the agent is currently living (it will be set when joining the environment)
        self.output_messages = [""] * 20
        self.output_messages_ids = [-1] * 20
        self.output_messages_count = 0
        self.output_messages_last_pos = -1

        self.commands_to_send = [  # list of known commands (to be sent)
            "enable_print",
            "disable_print",
            "authority"
        ]
        self.commands_to_receive = [  # list of known commands (to be received)
            "enable_print",
            "disable_print",
            "set_authority"
        ]

        assert authority in [0.0, 1.0], f"We just support authority = 0.0 (student) and authority = 1.0 (teacher)"

    def __str__(self):
        """String representation of an agent."""

        s = f"[Agent] {self.name} (authority: {self.authority})"
        if len(self.known_agents) > 0:
            s += "\n\t- known_agents:"
            for _s in self.known_agents:
                s += "\n\t\t" + str(_s.name) + " (authority: " + str(_s.authority) + ")"
        if len(self.known_streams) > 0:
            s += "\n\t- known_streams:"
            for _s in self.known_streams:
                s += "\n\t\t" + str(_s)
        s += "\n\t- behaviour:"
        s += "\n\t\t" + str(self.behav).replace("\n", "\n\t\t")
        s += "\n\t- model:"
        s += "\n\t\t" + str(self.model).replace("\n", "\n\t\t")
        return s

    def out(self, msg: str, show_state: bool = True, show_act: bool = True):
        """Print a message to the console, if enabled."""

        s = f"agent: {self.name}"
        if show_state:
            s += f", state: {self.behav.limbo_state if self.behav.limbo_state is not None else self.behav.state}"
        if show_act:
            caller = str(inspect.stack()[1].function)
            i = 0
            while str(caller).startswith("__") or str(caller).startswith("err"):
                i += 1
                caller = str(inspect.stack()[1 + i].function)
            args, _, _, values = inspect.getargvalues(inspect.stack()[1 + i].frame)
            s_args = Agent.__string_args(args, values)
            s += f", act: {caller}({s_args})"
        s = f"[{s}] {msg}"

        last_id = self.output_messages_ids[self.output_messages_last_pos]
        self.output_messages_last_pos = (self.output_messages_last_pos + 1) % len(self.output_messages)
        self.output_messages_count = min(self.output_messages_count + 1, len(self.output_messages))
        self.output_messages_ids[self.output_messages_last_pos] = last_id + 1
        self.output_messages[self.output_messages_last_pos] = msg

        if self.print_enabled:
            print(s)

    def err(self, msg: str, show_state: bool = True, show_act: bool = True):
        """Print an error message to the console, if enabled."""
        self.out("<FAILED> " + msg, show_state, show_act)

    def send_command(self, command: str, dest_agent: Self, data: dict | None = None) -> bool:
        """Send a predefined command."""

        print(f"Sending command: {command}, to: {dest_agent}")
        if command not in self.commands_to_send:
            self.out(f"Unknown command: {command}")
            return False

        if dest_agent not in self.known_agents:
            self.out(f"Unknown destination agent {dest_agent} for command {command}")
            return False

        if command == "enable_print":
            return dest_agent.receive_command(command, src_agent=self)

        elif command == "disable_print":
            return dest_agent.receive_command(command, src_agent=self)

        elif command == "set_authority":
            return dest_agent.receive_command(command, src_agent=self, data=data)

        else:
            return False

    def receive_command(self, command: str, src_agent: Self | None, data: dict | None = None) -> bool | Iterable:
        """Receive a predefined command."""

        print(f"Receiving command: {command}, from: {src_agent if src_agent is not None else 'environment'}")
        if command not in self.commands_to_receive:
            self.out(f"Unknown command: {command}")
            return False

        if src_agent not in self.known_agents:
            self.out(f"Unknown source agent {src_agent} for command {command}")
            return False

        if command == "enable_print":
            self.print_enabled = True
            return True

        elif command == "disable_print":
            self.print_enabled = False
            return True

        elif command == "set_authority":
            if src_agent.authority < self.authority:
                self.out(f"Discarding request to change authority level (provided by {src_agent} with "
                         f"authority {src_agent.authority}, smaller than the one of this agent)")
                return False

            self.authority = data['authority']
            return True

        else:
            return False

    def add_transit(self, *args, wildcards: dict[str, str] | None = None, **kwargs):
        self.behav.set_wildcards(wildcards)
        self.behav.add_transit(*args, **kwargs)

    def add_state_action(self, *args, wildcards: dict[str, str] | None = None, **kwargs):
        self.behav.set_wildcards(wildcards)
        self.behav.add_state_action(*args, **kwargs)

    def get_action_step(self):
        return self.behav.action_step

    def nop(self):
        """Do nothing."""

        self.out(f"Dummy action")
        return True

    def send_agents(self, agent: Self):
        """Sending known agents to another agent."""

        self.out(f"Sending contacts of {len(self.known_agents)} known agents to {agent.name}")

        if agent.behav.set_next_action("get_agents"):
            agent.behav.set_buffer_param_value("agent", self)
            agent.behav.set_buffer_param_value("agents", self.known_agents)
            return True
        else:
            self.out(f"Unable to share known agents with {agent.name}")
            return False

    def get_agents(self, agent: Self, agents: dict):
        """Receiving agents from another agent."""

        self.out(f"Getting contacts of {len(agents)} agents from "
                 f"{agent.name if agent is not None else 'the environment'}")

        if agent is not None and agent not in self.known_agents:
            self.out(f"Unknown agent: {agent.name}")
            return False

        for _agent_name, _agent in agents.items():
            if _agent != self:
                self.known_agents[_agent_name] = _agent
        return True

    def send_streams(self, agent: Self):
        """Sending known streams to another agent."""

        self.out(f"Sending {len(self.known_streams)} streams to {agent.name}")

        if agent.behav.set_next_action("get_streams"):
            agent.behav.set_buffer_param_value("agent", self)
            agent.behav.set_buffer_param_value("streams", self.known_streams)
            return True
        else:
            self.out(f"Unable to send streams to agent {agent.name}")
            return False

    def get_streams(self, agent: Self, streams: dict):
        """Receiving streams."""

        self.out(f"Getting {len(streams)} streams from {agent.name if agent is not None else 'the environment'}")

        if agent is not None and agent.name not in self.known_agents:
            self.out(f"Unknown agent: {agent.name}")
            return False

        for stream_name, stream in streams.items():
            self.known_streams[stream_name] = stream
        return True

    def behave_as(self, agent: Self):
        self.behav.include(agent.behav, copy=True)

    @staticmethod
    def __string_args(args, values):
        s = ""
        for j, arg in enumerate(args):
            if arg == "self":
                continue
            val = values[arg]
            if val is None:
                s += str(arg) + "=None"
            elif isinstance(val, float | int):
                s += str(arg) + "=" + str(val)
            elif isinstance(val, str):
                s += str(arg) + "='" + str(val) + "'"
            elif isinstance(val, Agent):
                s += str(arg) + "=" + val.name
            elif isinstance(val, Stream):
                s += str(arg) + "=" + val.name
            elif isinstance(val, dict):
                s += str(arg) + "={...}"
            elif isinstance(val, tuple):
                s += str(arg) + "=(...)"
            elif isinstance(val, list):
                s += str(arg) + "=[...]"
            else:
                s += str(arg) + "=..."
            if j < len(args) - 1:
                s += ", "
        return s
