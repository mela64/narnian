import inspect
import threading
from .agent import Agent
from .streams import Stream
from .fsm import FiniteStateMachine


class Environment:

    def __init__(self, name):
        """Create a new environment."""

        self.name = name  # name of the environment (keep it unique)
        self.behav = FiniteStateMachine(self, policy="sampling")  # FSM that describes the environment's behaviour
        self.streams = {}  # streams that are available in this environment
        self.agents = {}  # agents living in this environment
        self.print_enabled = True  # if output should be printed to screen
        self.using_server = False  # it will be changed by the server, if any
        self.step_event = None  # event that triggers a new step (manipulated by the server)
        self.wait_event = None  # event that triggers a new "wait-for-step-event" case (manipulated by the server)
        self.skip_clear_for = 0
        self.step = 0
        self.steps = None
        self.output_messages = [""] * 20
        self.output_messages_ids = [-1] * 20
        self.output_messages_count = 0
        self.output_messages_last_pos = -1
        self.commands_to_send = [  # list of known commands (to be sent)
            "enable_print",
            "disable_print"
        ]
        self.commands_to_receive = [  # list of known commands (either to be received)
            "enable_print",
            "disable_print"
        ]

    def __str__(self):
        """String representation of the environment."""

        s = f"[Environment] {self.name}"
        if len(self.agents) > 0:
            s += "\n\t- agents:"
            for _s in self.agents.values():
                s += "\n\t\t" + str(_s.name) + " (authority: " + str(_s.authority) + ")"
        if len(self.streams) > 0:
            s += "\n\t- streams:"
            for _s in self.streams:
                s += "\n\t\t" + str(_s)
        s += "\n\t- behaviour:"
        s += "\n\t\t" + str(self.behav).replace("\n", "\n\t\t")
        return s

    def add_agent(self, agent: Agent):
        """Add a new agent to this environment."""

        self.agents[agent.name] = agent

    def add_stream(self, stream: Stream):
        """Add a new stream to this environment."""

        self.streams[stream.get_hash()] = stream

    def out(self, msg: str, show_state: bool = True, show_act: bool = True):
        """Print a message to the console, if enabled."""

        s = f"envir: {self.name}"
        if show_state:
            s += f", state: {self.behav.limbo_state}"
        if show_act:
            caller = str(inspect.stack()[1].function)
            i = 0
            while str(caller).startswith("__"):
                i += 1
                caller = str(inspect.stack()[1 + i].function)
            args, _, _, values = inspect.getargvalues(inspect.stack()[1 + i].frame)
            s_args = Environment.__string_args(args, values)
            s += f", act: {caller}({s_args})"
        s = f"[{s}] {msg}"

        last_id = self.output_messages_ids[self.output_messages_last_pos]
        self.output_messages_last_pos = (self.output_messages_last_pos + 1) % len(self.output_messages)
        self.output_messages_count = min(self.output_messages_count + 1, len(self.output_messages))
        self.output_messages_ids[self.output_messages_last_pos] = last_id + 1
        self.output_messages[self.output_messages_last_pos] = s

        if self.print_enabled:
            print(s)

    def err(self, msg: str, show_state: bool = True, show_act: bool = True):
        """Print an error message to the console, if enabled."""
        self.out("<FAILED> " + msg, show_state, show_act)

    def run(self, steps: int | None = None):
        """Run the environment."""

        assert steps is None or steps > 0, "Invalid number of steps"
        self.steps = steps

        # sort agents by authority (largest authorities go first)
        sorted_agents = sorted(self.agents.values(), key=lambda x: x.authority, reverse=True)

        # external events
        if self.using_server:
            self.step_event = threading.Event()
            self.wait_event = threading.Event()

        # main loop
        self.step = 0
        while True:

            # in server mode, we wait for an external event to go ahead (step_event.set())
            if self.using_server:
                self.wait_event.set()
                self.step_event.wait()
                self.wait_event.clear()

            # increase the step index (keep it here, after "wait", even if it sounds odd)
            if self.step > 0:
                for _, stream in self.streams.items():
                    stream.next_step()

            self.out(f">>> Running step {self.step} <<<", show_state=False, show_act=False)

            # self.out("Running inner state actions (if any)", show_act=False)
            self.behav.act_states()
            for agent in sorted_agents:
                # agent.out("Running inner state actions (if any)", show_act=False)
                agent.behav.act_states()

            # self.out("Running transition-related actions (if any)", show_act=False)
            self.behav.act_transitions()
            for agent in sorted_agents:
                # agent.out("Running transition-related actions (if any)", show_act=False)
                agent.behav.act_transitions()

            self.step += 1
            if self.steps is not None and self.step == self.steps:
                break

            # in step mode, we clear the external event to be able to wait for a new one
            if self.using_server:
                if self.skip_clear_for <= 0:
                    self.step_event.clear()
                else:
                    self.skip_clear_for -= 1

    def send_command(self, command: str, dest_agent: Agent, data: dict | None = None) -> bool:
        """Send a predefined command to an agent."""

        self.out(f"Sending command {command} to {dest_agent}")
        if command not in self.commands_to_send:
            self.err(f"Unknown command: {command}")
            return False

        if dest_agent not in self.agents:
            self.err(f"Unknown destination agent {dest_agent} for command {command}")
            return False

        if command == "enable_print":
            return dest_agent.receive_command(command, src_agent=None)

        elif command == "disable_print":
            return dest_agent.receive_command(command, src_agent=None)

        else:
            return False

    # noinspection PyUnusedLocal
    def receive_command(self, command: str, src_agent: Agent, data: dict | None = None) -> bool:
        """Receive a predefined command."""

        self.out(f"Receiving command {command} from {src_agent}")
        if command not in self.commands_to_receive:
            self.err(f"Unknown command: {command}")
            return False

        if src_agent not in self.agents:
            self.err(f"Unknown source agent {src_agent} for command {command}")
            return False

        elif command == "enable_print":
            self.print_enabled = True
            return True

        elif command == "disable_print":
            self.print_enabled = False
            return True

        else:
            return False

    def nop(self):
        """Do nothing."""

        self.out(f"Dummy action")
        return True

    def send_streams_to_all(self):
        """Sending streams to all agents."""

        self.out(f"Providing {len(self.streams)} streams to all agents")
        for agent in self.agents.values():
            if agent.behav.set_action_score("get_streams", new_score='top'):
                agent.behav.set_buffer_param_value("agent", None)
                agent.behav.set_buffer_param_value("streams", self.streams)
            else:
                self.err(f"Unable to provide streams to agent {agent.name} (stopping)")
                return False
        return True

    def send_agents_to_all(self):
        """Sharing each agent with all the other ones."""

        self.out(f"Sharing contacts of {len(self.agents)} agents with all the other agents")
        for agent in self.agents.values():
            if agent.behav.set_action_score("get_agents", new_score='top'):
                agent.behav.set_buffer_param_value("agent", None)
                agent.behav.set_buffer_param_value("agents", self.agents)
            else:
                self.err(f"Unable to share with agent {agent.name} (stopping)")
                return False
        return True

    def enable_all_streams(self):
        """Enables all streams."""

        self.out(f"Enabling all streams")
        for _, stream in self.streams.items():
            stream.enable()
        return True

    def disable_all_streams(self):
        """Disables all stream."""

        self.out(f"Disabling all streams")
        for _, stream in self.streams.items():
            stream.disable()
        return True

    def enable_stream(self, name: str, creator: str | None = None):
        """Enables a stream."""
        stream_hash = Stream.build_hash(name, creator)

        self.out(f"Enabling stream: {stream_hash}")
        if stream_hash not in self.streams:
            self.err(f"Stream {stream_hash} is unknown")
            return False

        self.streams[stream_hash].enable()
        return True

    def disable_stream(self, name: str, creator: str | None = None):
        """Disables a stream."""

        stream_hash = Stream.build_hash(name, creator)

        self.out(f"Disabling stream: {stream_hash}")
        if stream_hash not in self.streams:
            self.err(f"Stream {stream_hash} is unknown")
            return False

        self.streams[stream_hash].disable()
        return True

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
