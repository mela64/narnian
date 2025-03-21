from narnian.model import Model
from narnian.agent import Agent
from typing_extensions import Self
from collections.abc import Iterable
from narnian.streams import Stream, BufferedStream


class BasicAgent(Agent):

    def __init__(self, name: str, model: Model, authority: float):
        """Create a new basic agent."""

        super(BasicAgent, self).__init__(name, model, authority)
        self.available = True  # if available for engagement
        self.target_agents = []  # other agents which are available for possible engagement
        self.engaged_agents = []  # agents involved in current interactions
        self.received_hashes = []  # streams that were sent (e.g., generated) by other agents
        self.eval_results = []  # results of the last evaluation
        self.valid_cmp_agents = []  # agents for which the last evaluation was positive
        self.preferred_streams = []  # list of preferred streams
        self.repeat = 1  # number of repetitions of the playlist
        self.cur_preferred_stream = 0  # id of the current preferred stream from the list
        self.last_recorded_stream_num = 0  # numerical index of to the last recorded stream, if any (1: first)
        self.last_generated_stream_num = 0  # numerical index of to the last generated stream, if any (1: first)
        self.last_eval_stream_num = 0  # numerical index of to the last evaluated stream, if any (1: first)
        self.failed_communicating_completion = None  # it keeps track of failed __complete_do actions

        # commands that can be handled by the agent
        self.commands_to_send.append("stop_current_action")
        self.commands_to_send.append("kill")
        self.commands_to_receive.append("stop_current_action")
        self.commands_to_receive.append("kill")

    def send_command(self, command: str, dest_agent: Self, data: dict | None = None) -> bool:
        """Send a predefined command."""

        ret = super().send_command(command, dest_agent, data)

        if command == "stop_current_action":
            return dest_agent.receive_command(command, src_agent=self)

        elif command == "kill":
            return dest_agent.receive_command(command, src_agent=self)

        else:
            return ret

    def receive_command(self, command: str, src_agent: Self | None, data: dict | None = None) -> bool | Iterable:
        """Receive a predefined command."""

        ret = super().receive_command(command, src_agent, data)

        if command == "stop_current_action":
            self.err(f"In the current implementation, actions cannot be stopped.")
            return False

        elif command == "kill":
            self.err(f"In the current implementation, agents cannot be killed.")
            return False

        else:
            return ret

    def find_agent_to_engage(self, min_auth: float, max_auth: float):
        """Find an agent whose authority is in the specified range."""

        self.out(f"Finding an available agent whose authority is in [{min_auth}, {max_auth}]")
        self.target_agents = []
        for agent in self.known_agents.values():
            if agent.available and min_auth <= agent.authority <= max_auth:
                self.target_agents.append(agent)
        return True

    def disengage(self):
        """Clear all the currently established engagements."""

        self.out(f"Disengaging from all agents")
        self.engaged_agents = []
        self.available = True
        return True

    def send_disengagement(self):
        """Ask for disengagement."""

        at_least_one_sent = False

        if len(self.engaged_agents) > 0:
            self.out(f"Sending disengagement request to {', '.join([x.name for x in self.engaged_agents])}")
        for agent in self.engaged_agents:
            if self.set_next_action(agent, "get_disengagement", {"agent": self}):
                at_least_one_sent = True
            else:
                self.err(f"Unable to send disengagement to {agent.name}")

        return at_least_one_sent

    def get_disengagement(self, agent: Self):
        """Get a disengagement request from an agent."""

        self.out(f"Getting a disengagement request from {agent.name}")
        if agent.name not in self.known_agents:
            self.err(f"Unknown agent: {agent.name}")
            return False

        if agent not in self.engaged_agents:
            self.err(f"Not previously engaged to {agent.name}")
            return False

        self.engaged_agents = [x for x in self.engaged_agents if x != agent]
        self.available = len(self.engaged_agents) == 0
        return True

    def send_engagement(self):
        """Offer engagement to another agent."""

        at_least_one_sent = False

        if len(self.target_agents) > 0:
            self.out(f"Sending engagement request to {', '.join([x.name for x in self.target_agents])}")
        for target_agent in self.target_agents:
            if self.set_next_action(target_agent, "get_engagement", {"agent": self}):
                at_least_one_sent = True
            else:
                self.err(f"Unable to send engagement to {target_agent.name}")

        return at_least_one_sent

    def get_engagement(self, agent: Self, min_auth: float, max_auth: float):
        """Receive engagement from another agent whose authority is in the specified range."""

        self.out(f"Getting engagement from {agent.name}")
        if agent.name not in self.known_agents:
            self.err(f"Unknown agent: {agent.name}")
            return False

        # confirming
        if self.available and min_auth <= agent.authority <= max_auth:
            if self.set_next_action(agent, "got_engagement", {"agent": self}):
                self.engaged_agents.append(agent)
                self.available = False
                return True
            else:
                self.err(f"Unable to confirm engagement to {agent.name}")
                return False
        else:
            self.err(f"Cannot engage to {agent.name}")
            return False

    def got_engagement(self, agent: Self):
        """Confirm an engagement."""

        self.out(f"Confirming engagement with {agent.name}")
        if agent in self.target_agents:
            self.engaged_agents.append(agent)
            self.target_agents = [x for x in self.target_agents if x != agent]
            self.available = False
            return True
        else:
            self.err(f"Unable to confirm engagement with {agent.name}")
            return False

    def gen(self, dhat_hash: str, u_hash: str | None = None, du_hash: str | None = None, steps: int = 100) -> bool:
        """Generate a signal."""

        self.out(f"Generating signal")
        return self.__process_streams(u_hash=u_hash, du_hash=du_hash,
                                      yhat_hash=None, dhat_hash=dhat_hash,
                                      skip_gen=False, skip_pred=True,
                                      steps=steps)

    def ask_gen(self, dhat_hash: str, u_hash: str | None = None, du_hash: str | None = None, ask_steps: int = 100,
                agent: Self | None = None):
        """Asking for generation."""

        involved_agents = self.__involved_agents(agent)
        if len(involved_agents) == 0:
            return False

        self.received_hashes = []
        at_least_one_completed = False
        self.out(f"Asking {', '.join([x.name for x in involved_agents])} to generate signal")
        for agent in involved_agents:
            ret = self.__ask(for_what="gen", agent=agent,
                             u_hash=u_hash, du_hash=du_hash,
                             yhat_hash=None, dhat_hash=dhat_hash,
                             steps=ask_steps)
            at_least_one_completed = at_least_one_completed or ret
        return at_least_one_completed

    def do_gen(self, agent: Self,
               dhat_hash: str, u_hash: str | None = None, du_hash: str | None = None, steps: int = 100) -> bool:
        """Generate a signal."""

        if len(self.engaged_agents) == 0 or agent not in self.engaged_agents:
            self.err(f"Not engaged to agent {agent.name}")
            return False

        if self.failed_communicating_completion is None or self.failed_communicating_completion != "gen":
            ret = self.gen(dhat_hash, u_hash, du_hash, steps)
        else:
            ret = True
        return self.__complete_do("gen", agent, ret) \
            if self.get_action_step() == steps - 1 else ret

    def done_gen(self, agent: Self, streams: dict):
        """Confirm generation."""

        if len(self.engaged_agents) == 0 or agent not in self.engaged_agents:
            self.err(f"Not engaged to agent {agent.name}")
            return False

        self.out(f"Agent {agent.name} finished generation")
        return self.__done(agent, streams)

    def pred(self, yhat_hash: str, steps: int = 100) -> bool:
        """Predict a descriptor."""

        self.out(f"Predicting descriptor of {yhat_hash}")
        return self.__process_streams(u_hash=None, du_hash=None,
                                      yhat_hash=yhat_hash, dhat_hash=None,
                                      skip_gen=True, skip_pred=False,
                                      steps=steps)

    def ask_pred(self, yhat_hash: str, ask_steps: int = 100,
                 agent: Self | None = None):
        """Asking for prediction of a descriptor."""

        involved_agents = self.__involved_agents(agent)
        if len(involved_agents) == 0:
            return False

        self.received_hashes = []
        at_least_one_completed = False
        self.out(f"Asking {', '.join([x.name for x in involved_agents])} to predict descriptor of {yhat_hash}")
        for agent in involved_agents:
            ret = self.__ask(for_what="pred", agent=agent,
                             u_hash=None, du_hash=None,
                             yhat_hash=yhat_hash, dhat_hash=None,
                             steps=ask_steps)
            at_least_one_completed = at_least_one_completed or ret
        return at_least_one_completed

    def do_pred(self, agent: Self,
                yhat_hash: str, steps: int = 100) -> bool:
        """Predict a descriptor."""

        if len(self.engaged_agents) == 0 or agent not in self.engaged_agents:
            self.err(f"Not engaged to agent {agent.name}")
            return False

        if self.failed_communicating_completion is None or self.failed_communicating_completion != "pred":
            ret = self.pred(yhat_hash, steps)
        else:
            ret = True
        return self.__complete_do("pred", agent, ret) \
            if self.get_action_step() == steps - 1 else ret

    def done_pred(self, agent: Self, streams: dict):
        """Confirm learning to generate."""

        if len(self.engaged_agents) == 0 or agent not in self.engaged_agents:
            self.err(f"Not engaged to agent {agent.name}")
            return False

        self.out(f"Agent {agent.name} finished predicting descriptor")
        return self.__done(agent, streams)

    def gen_and_pred(self, u_hash: str | None = None, du_hash: str | None = None, steps: int = 100) -> bool:
        """Generate a signal and predict a descriptor."""

        self.out(f"Generating signal and predicting descriptor")
        return self.__process_streams(u_hash=u_hash, du_hash=du_hash,
                                      yhat_hash=None, dhat_hash=None,
                                      skip_gen=False, skip_pred=False,
                                      steps=steps)

    def ask_gen_and_pred(self, u_hash: str | None = None, du_hash: str | None = None, ask_steps: int = 100,
                         agent: Self | None = None):
        """Asking for generation and prediction of descriptor."""

        involved_agents = self.__involved_agents(agent)
        if len(involved_agents) == 0:
            return False

        self.received_hashes = []
        at_least_one_completed = False
        self.out(f"Asking {', '.join([x.name for x in involved_agents])} to generate signal and predict descriptor")
        for agent in involved_agents:
            ret = self.__ask(for_what="gen_and_pred", agent=agent,
                             u_hash=u_hash, du_hash=du_hash,
                             yhat_hash=None, dhat_hash=None,
                             steps=ask_steps)
            at_least_one_completed = at_least_one_completed or ret
        return at_least_one_completed

    def do_gen_and_pred(self, agent: Self,
                        u_hash: str | None = None, du_hash: str | None = None, steps: int = 100) -> bool:
        """Generate a signal and predict a descriptor."""

        if len(self.engaged_agents) == 0 or agent not in self.engaged_agents:
            self.err(f"Not engaged to agent {agent.name}")
            return False

        if self.failed_communicating_completion is None or self.failed_communicating_completion != "gen_and_pred":
            ret = self.gen_and_pred(u_hash, du_hash, steps)
        else:
            ret = True
        return self.__complete_do("gen_and_pred", agent, ret) \
            if self.get_action_step() == steps - 1 else ret

    def done_gen_and_pred(self, agent: Self, streams: dict):
        """Confirm generation and prediction."""

        if len(self.engaged_agents) == 0 or agent not in self.engaged_agents:
            self.err(f"Not engaged to agent {agent.name}")
            return False

        self.out(f"Agent {agent.name} finished generation and prediction of descriptor")
        return self.__done(agent, streams)

    def learn_gen(self, yhat_hash: str, dhat_hash: str, u_hash: str | None = None, du_hash: str | None = None,
                  steps: int = 100) -> bool:
        """Learn to generate a signal."""

        self.out(f"Learning to generate signal {yhat_hash}")
        return self.__process_streams(u_hash=u_hash, du_hash=du_hash,
                                      yhat_hash=yhat_hash, dhat_hash=dhat_hash,
                                      skip_gen=False, skip_pred=True,
                                      steps=steps)

    def ask_learn_gen(self, yhat_hash: str, dhat_hash: str, u_hash: str | None = None, du_hash: str | None = None,
                      ask_steps: int = 100,
                      agent: Self | None = None):
        """Asking for learning to generate."""

        involved_agents = self.__involved_agents(agent)
        if len(involved_agents) == 0:
            return False

        self.received_hashes = []
        at_least_one_completed = False
        self.out(f"Asking {', '.join([x.name for x in involved_agents])} to learn to generate signal "
                 f"{yhat_hash if yhat_hash != '<playlist>' else self.preferred_streams[self.cur_preferred_stream]}")
        for agent in involved_agents:
            ret = self.__ask(for_what="learn_gen", agent=agent,
                             u_hash=u_hash, du_hash=du_hash,
                             yhat_hash=yhat_hash, dhat_hash=dhat_hash,
                             steps=ask_steps)
            at_least_one_completed = at_least_one_completed or ret
        return at_least_one_completed

    def do_learn_gen(self, agent: Self,
                     yhat_hash: str, dhat_hash: str,
                     u_hash: str | None = None, du_hash: str | None = None, steps: int = 100) -> bool:
        """Learn to generate a signal."""

        if len(self.engaged_agents) == 0 or agent not in self.engaged_agents:
            self.err(f"Not engaged to agent {agent.name}")
            return False

        if self.failed_communicating_completion is None or self.failed_communicating_completion != "learn_gen":
            ret = self.learn_gen(yhat_hash, dhat_hash, u_hash, du_hash, steps)
        else:
            ret = True
        return self.__complete_do("learn_gen", agent, ret) \
            if self.get_action_step() == steps - 1 else ret

    def done_learn_gen(self, agent: Self, streams: dict):
        """Confirm learning to generate."""

        if len(self.engaged_agents) == 0 or agent not in self.engaged_agents:
            self.err(f"Not engaged to agent {agent.name}")
            return False

        self.out(f"Agent {agent.name} finished learning to generate")
        return self.__done(agent, streams)

    def learn_pred(self, yhat_hash: str, dhat_hash: str, steps: int = 100) -> bool:
        """Learn to predict a descriptor."""

        self.out(f"Learning to predict descriptor {dhat_hash}")
        return self.__process_streams(u_hash=None, du_hash=None,
                                      yhat_hash=yhat_hash, dhat_hash=dhat_hash,
                                      skip_gen=True, skip_pred=False,
                                      steps=steps)

    def ask_learn_pred(self, yhat_hash: str, dhat_hash: str, ask_steps: int = 100,
                       agent: Self | None = None):
        """Asking for learning to predict descriptor."""

        involved_agents = self.__involved_agents(agent)
        if len(involved_agents) == 0:
            return False

        self.received_hashes = []
        at_least_one_completed = False
        self.out(f"Asking {', '.join([x.name for x in involved_agents])} to learn to predict descriptor "
                 f"{dhat_hash if dhat_hash != '<playlist>' else self.preferred_streams[self.cur_preferred_stream]}")
        for agent in involved_agents:
            ret = self.__ask(for_what="learn_pred", agent=agent,
                             u_hash=None, du_hash=None,
                             yhat_hash=yhat_hash, dhat_hash=dhat_hash,
                             steps=ask_steps)
            at_least_one_completed = at_least_one_completed or ret
        return at_least_one_completed

    def do_learn_pred(self, agent: Self,
                      yhat_hash: str, dhat_hash: str, steps: int = 100) -> bool:
        """Learn to predict a descriptor."""

        if len(self.engaged_agents) == 0 or agent not in self.engaged_agents:
            self.err(f"Not engaged to agent {agent.name}")
            return False

        if self.failed_communicating_completion is None or self.failed_communicating_completion != "learn_pred":
            ret = self.learn_pred(yhat_hash, dhat_hash, steps)
        else:
            ret = True
        return self.__complete_do("learn_pred", agent, ret) \
            if (self.get_action_step() == steps - 1) else ret

    def done_learn_pred(self, agent: Self, streams: dict):
        """Confirm learning to generate."""

        if len(self.engaged_agents) == 0 or agent not in self.engaged_agents:
            self.err(f"Not engaged to agent {agent.name}")
            return False

        self.out(f"Agent {agent.name} finished learning to predict descriptor")
        return self.__done(agent, streams)

    def learn_gen_and_pred(self, yhat_hash: str, dhat_hash: str,
                           u_hash: str | None = None, du_hash: str | None = None, steps: int = 100) -> bool:
        """Learn to generate a signal and predict a descriptor."""

        self.out(f"Learning to generate signal {yhat_hash} and predict descriptor {dhat_hash}")
        return self.__process_streams(u_hash=u_hash, du_hash=du_hash,
                                      yhat_hash=yhat_hash, dhat_hash=dhat_hash,
                                      skip_gen=False, skip_pred=False,
                                      steps=steps)

    def ask_learn_gen_and_pred(self,
                               yhat_hash: str, dhat_hash: str,
                               u_hash: str | None = None, du_hash: str | None = None,
                               ask_steps: int = 100,
                               agent: Self | None = None):
        """Asking to learn to generate signal and predict descriptor."""

        involved_agents = self.__involved_agents(agent)
        if len(involved_agents) == 0:
            return False

        self.received_hashes = []
        at_least_one_completed = False
        self.out(f"Asking {', '.join([x.name for x in involved_agents])} to learn to generate signal "
                 f"{yhat_hash if yhat_hash != '<playlist>' else self.preferred_streams[self.cur_preferred_stream]} "
                 f"and to learn to predict descriptor "
                 f"{dhat_hash if dhat_hash != '<playlist>' else self.preferred_streams[self.cur_preferred_stream]}")
        for agent in involved_agents:
            ret = self.__ask(for_what="learn_gen_and_pred", agent=agent,
                             u_hash=u_hash, du_hash=du_hash,
                             yhat_hash=yhat_hash, dhat_hash=dhat_hash,
                             steps=ask_steps)
            at_least_one_completed = at_least_one_completed or ret
        return at_least_one_completed

    def do_learn_gen_and_pred(self, agent: Self,
                              yhat_hash: str, dhat_hash: str,
                              u_hash: str | None = None, du_hash: str | None = None, steps: int = 100) -> bool:
        """Learn to generate a signal and predict a descriptor."""

        if len(self.engaged_agents) == 0 or agent not in self.engaged_agents:
            self.err(f"Not engaged to agent {agent.name}")
            return False

        if self.failed_communicating_completion is None or self.failed_communicating_completion != "learn_gen_and_pred":
            ret = self.learn_gen_and_pred(yhat_hash, dhat_hash, u_hash, du_hash, steps)
        else:
            ret = True
        return self.__complete_do("learn_gen_and_pred", agent, ret) \
            if self.get_action_step() == steps - 1 else ret

    def done_learn_gen_and_pred(self, agent: Self, streams: dict):
        """Confirm learning to generate."""

        if len(self.engaged_agents) == 0 or agent not in self.engaged_agents:
            self.err(f"Not engaged to agent {agent.name}")
            return False

        self.out(f"Agent {agent.name} finished learning to generate and learning to predict descriptor")
        return self.__done(agent, streams)

    def eval(self, stream_hash: str, what: str, how: str, steps: int = 100) -> float:
        """Compare two signals."""

        assert what in ["y", "d"], "Only 'y' and 'd' are allowed for the 'what' argument"

        if stream_hash is not None and stream_hash == "<playlist>":
            stream_hash = self.preferred_streams[self.cur_preferred_stream]

        self.eval_results = []
        for agent, received_hash in self.received_hashes:
            self.out(f"Comparing {received_hash} from agent {agent.name} with {stream_hash} ({what}, {how})")
            eval_result, ret = self.__compare_streams(stream_a_hash=received_hash,
                                                      stream_b_hash=stream_hash, what=what, how=how, steps=steps)
            self.out(f"Result: {eval_result}")
            if not ret:
                return False
            else:
                self.eval_results.append((agent, eval_result))

        return True

    def compare_eval(self, cmp: str, thres: float, good_if_true: True) -> bool:
        """After having completed an evaluation."""

        assert cmp in ["<", ">", ">=", "<="], f"Invalid comparison operator: {cmp}"
        assert thres >= 0., f"Invalid evaluation threshold: {thres} (it must be in >= 0.)"

        self.valid_cmp_agents = []
        msgs = []

        for agent, eval_result in self.eval_results:
            self.out(f"Checking if result {eval_result} {cmp} {thres}, for agent {agent.name}")

            if eval_result < 0.:
                self.err(f"Invalid evaluation result: {eval_result}")
                return False

            outcome = False
            if cmp == "<" and eval_result < thres:
                outcome = True
            elif cmp == "<=" and eval_result <= thres:
                outcome = True
            elif cmp == ">" and eval_result > thres:
                outcome = True
            elif cmp == ">=" and eval_result >= thres:
                outcome = True

            if cmp[0] == "<" or cmp[0] == "<=":
                alias = 'error level' if good_if_true else 'mark'
            else:
                alias = 'mark' if good_if_true else 'error level'

            if good_if_true:
                if outcome:
                    msgs.append(f"Agent {agent.name} passed with {alias} {eval_result}/{thres}")
                    self.valid_cmp_agents.append(agent)
                else:
                    msgs.append(f"Agent {agent.name} did not pass")
            else:
                if outcome:
                    msgs.append(f"Agent {agent.name} did not pass")
                else:
                    msgs.append(f"Agent {agent.name} passed with {alias} {eval_result}/{thres}")
                    self.valid_cmp_agents.append(agent)

            if len(msgs) > 1:
                msgs[-1] = msgs[-1].lower()[0] + msgs[-1][1:]

        if len(self.valid_cmp_agents) == 0:
            self.err(f"The evaluation was not passed by any agents")
            return False
        else:
            self.out(", ".join(msgs))
            return True

    def set_authority(self, agent: Self | str, auth: float):
        assert 0 <= auth <= 1., f"Invalid authority {auth}"
        assert isinstance(agent, Agent) or (isinstance(agent, str) and agent == "<valid_cmp>"), \
            f"Invalid agent: {agent}"

        agents = [agent] if isinstance(agent, Agent) else self.valid_cmp_agents

        for _agent in agents:
            if self.authority <= _agent.authority:
                self.out(f"Not changing the authority of {_agent.name} "
                         f"since it has an authority higher-than/equal-to mine")
            else:
                self.out(f"Changing the authority of {_agent.name} to {auth}")
                _agent.authority = auth

        return True

    def __involved_agents(self, agent: Self | str) -> list[Self]:
        involved_agents = [agent] if agent is not None and isinstance(agent, Agent) else (
            self.valid_cmp_agents) if isinstance(agent, str) and agent == "<valid_cmp>" else self.engaged_agents
        if len(involved_agents) == 0:
            self.err("Not engaged to any agents or no agent specified")
        return involved_agents

    def wait_for_actions(self, agent: Self | str, from_state: str, to_state: str, wait: bool):
        """Lock or unlock every action between a pair of states."""

        involved_agents = self.__involved_agents(agent)
        if len(involved_agents) == 0:
            return False

        at_least_one_completed = False
        for _agent in involved_agents:
            ret = _agent.behav.wait_for_actions(from_state, to_state, wait)
            at_least_one_completed = at_least_one_completed or ret
        return at_least_one_completed

    def set_next_action(self, agent: Self | str, action: str, args: dict | None = None):
        """Try to tell another agent what is the next action it should run."""

        involved_agents = self.__involved_agents(agent)
        if len(involved_agents) == 0:
            return False

        at_least_one_completed = False
        for _agent in involved_agents:
            if _agent.behav.set_next_action(action):
                if args is None:
                    args = {}
                for k, v in args.items():
                    _agent.behav.set_buffer_param_value(k, v)
                ret = True
            else:
                ret = False
            at_least_one_completed = at_least_one_completed or ret
        return at_least_one_completed

    def next_pref_stream(self):
        """Moves to the next stream in the list of preferred ones."""

        if len(self.preferred_streams) == 0:
            self.err(f"Cannot move to the next stream because the list of preferred streams is empty")
            return False

        self.cur_preferred_stream = (self.cur_preferred_stream + 1) % len(self.preferred_streams)
        suffix = ", warning: restarted" if self.cur_preferred_stream == 0 else ""
        self.out(f"Moving to the next preferred stream ({self.preferred_streams[self.cur_preferred_stream]}){suffix}")
        return True

    def first_pref_stream(self):
        """Moves to the next stream in the list of preferred ones."""

        if len(self.preferred_streams) == 0:
            self.err(f"Cannot move to the first stream because the list of preferred streams is empty")
            return False

        self.cur_preferred_stream = 0
        self.out(f"Moving to the first preferred stream ({self.preferred_streams[self.cur_preferred_stream]})")
        return True

    def check_pref_stream(self, what: str = "last") -> bool:
        """Check the current preferred stream."""

        valid = ['first', 'last', 'not_first', 'not_last', 'last_round', 'not_last_round', 'last_song', 'not_last_song']
        assert what in valid, f"The what argument can only be one of {valid}"

        self.out(f"Checking if the current preferred playlist item is the '{what}' one")
        if what == "first":
            return self.cur_preferred_stream == 0
        elif what == "last":
            return self.cur_preferred_stream == len(self.preferred_streams) - 1
        elif what == "not_first":
            return self.cur_preferred_stream != 0
        elif what == "not_last":
            return self.cur_preferred_stream != len(self.preferred_streams) - 1
        elif what == "last_round":
            return self.cur_preferred_stream + len(self.preferred_streams) // self.repeat >= len(self.preferred_streams)
        elif what == "not_last_round":
            return self.cur_preferred_stream + len(self.preferred_streams) // self.repeat < len(self.preferred_streams)
        elif what == "last_song":
            num_streams_in_playlist = len(self.preferred_streams) // self.repeat
            return (self.cur_preferred_stream + 1) % num_streams_in_playlist == 0
        elif what == "not_last_song":
            num_streams_in_playlist = len(self.preferred_streams) // self.repeat
            return (self.cur_preferred_stream + 1) % num_streams_in_playlist != 0

    def set_pref_streams(self, stream_hashes: list[str], repeat: int = 1):
        """Fill a list with preferred streams."""

        self.out(f"Setting up a list of {len(stream_hashes)} preferred streams")
        self.cur_preferred_stream = 0
        self.preferred_streams = []
        self.repeat = repeat
        for i in range(0, self.repeat):
            for stream_hash in stream_hashes:
                self.preferred_streams.append(stream_hash)
        return True

    def share_streams(self, agent: Self | None = None):
        """Share streams with the currently engaged agent."""

        involved_agents = self.__involved_agents(agent)
        if len(involved_agents) == 0:
            return False

        at_least_one_completed = False
        self.out(f"Sharing streams with {', '.join([x.name for x in involved_agents])}")
        for agent in involved_agents:
            ret = self.send_streams(agent)
            at_least_one_completed = at_least_one_completed or ret
        return at_least_one_completed

    def record(self, stream_hash: str, steps: int = 100):
        """Record a stream."""

        self.out(f"Recording stream {stream_hash}")
        if self.get_action_step() == 0:

            if stream_hash not in self.known_streams:
                self.err(f"Unknown stream (stream_hash): {stream_hash}")
                return False

            if steps <= 0:
                "Invalid number of steps to record: " + str(steps)
                return False

            stream_src = self.known_streams[stream_hash]

            if isinstance(stream_src, BufferedStream):
                if steps > len(stream_src):
                    self.err(f"Cannot process stream {stream_src} for {steps} steps, since it is is shorter")
                offset_stream_src = stream_src.get_first_step_offset_given_current_step()
                self.buffered_streams_offsets[stream_src.get_hash()] = offset_stream_src
            else:
                offset_stream_src = 0

            # new recorded stream
            self.last_recorded_stream_num += 1

            # creating the new recorded stream
            stream_dest = BufferedStream(use_static_descriptor=False)
            stream_dest.set_name("recorded" + str(self.last_recorded_stream_num))
            stream_dest.set_creator(f"{self.name}")
            stream_dest.set_meta("Recorded stream: " + stream_hash)
            stream_dest.attributes[0] = stream_src.attributes[0]
            stream_dest.attributes[1] = stream_src.attributes[1]
            self.known_streams[stream_dest.get_hash()] = stream_dest
        else:

            # retrieving the stream(s)
            stream_dest = self.known_streams[Stream.build_hash("recorded" + str(self.last_recorded_stream_num),
                                                               self.name)]
            stream_src = self.known_streams[stream_hash]
            if isinstance(stream_src, BufferedStream):
                offset_stream_src = self.buffered_streams_offsets[stream_src.get_hash()]
            else:
                offset_stream_src = 0

        # recording
        y, d = stream_src[stream_src.k + offset_stream_src]
        stream_dest.append_data(y, d)

        return True

    def __done(self, agent: Self, streams: dict):
        """Confirming generation, prediction, learning."""

        assert len(streams) == 1, f"Only one stream is expected (got {len(streams)})"

        # checking confirmation and saving streams
        for stream_hash, stream in streams.items():
            self.known_streams[stream_hash] = stream
            self.received_hashes.append((agent, stream_hash))
            return True

    def __ask(self, for_what: str, agent: Self,
              u_hash: str | None, du_hash: str | None,
              yhat_hash: str | None, dhat_hash: str | None,
              steps: int = 100):

        if agent.name not in self.known_agents:
            self.err(f"Unknown agent: {agent.name}")
            return False

        if self.known_agents[agent.name] != agent:
            self.err(f"Not-matching agent objects for agent {agent.name}")
            return False

        assert for_what in ["gen", "pred", "gen_and_pred", "learn_gen", "learn_pred", "learn_gen_and_pred"]

        # overwriting stream hash in case of playlist
        if u_hash is not None and u_hash == "<playlist>":
            u_hash = self.preferred_streams[self.cur_preferred_stream]
        if du_hash is not None and du_hash == "<playlist>":
            du_hash = self.preferred_streams[self.cur_preferred_stream]
        if yhat_hash is not None and yhat_hash == "<playlist>":
            yhat_hash = self.preferred_streams[self.cur_preferred_stream]
        if dhat_hash is not None and dhat_hash == "<playlist>":
            dhat_hash = self.preferred_streams[self.cur_preferred_stream]

        if for_what == "learn_gen" or for_what == "learn_gen_and_pred":
            yhat_stream = self.known_streams[yhat_hash]
            if isinstance(yhat_stream, BufferedStream):
                y_text, _ = yhat_stream.to_text(length=200)
                if y_text is not None and len(y_text) > 0:
                    self.out("Asking to generate: \"" + y_text + "\"")

        # triggering
        if for_what == "gen":
            if self.set_next_action(agent, "do_gen",
                                    {"agent": self, "u_hash": u_hash, "du_hash": du_hash,
                                     "dhat_hash": dhat_hash, "steps": steps}):
                return True
            else:
                self.err(f"Unable to ask {agent.name} to generate")
                return False
        elif for_what == "pred":
            if self.set_next_action(agent, "do_pred",
                                    {"agent": self, "yhat_hash": yhat_hash, "steps": steps}):
                return True
            else:
                self.err(f"Unable to ask {agent.name} to predict")
                return False
        elif for_what == "gen_and_pred":
            if self.set_next_action(agent, "do_gen_and_pred",
                                    {"agent": self, "u_hash": u_hash, "du_hash": du_hash, "steps": steps}):
                return True
            else:
                self.err(f"Unable to ask {agent.name} to generated and predict")
                return False
        elif for_what == "learn_gen":
            if self.set_next_action(agent, "do_learn_gen",
                                    {"agent": self, "u_hash": u_hash, "du_hash": du_hash, "yhat_hash": yhat_hash,
                                     "dhat_hash": dhat_hash, "steps": steps}):
                return True
            else:
                self.err(f"Unable to ask {agent.name} to learn to generate")
                return False
        elif for_what == "learn_pred":
            if self.set_next_action(agent, "do_learn_pred",
                                    {"agent": self, "yhat_hash": yhat_hash,
                                     "dhat_hash": dhat_hash, "steps": steps}):
                return True
            else:
                self.err(f"Unable to ask {agent.name} to learn to predict")
                return False
        elif for_what == "learn_gen_and_pred":
            if self.set_next_action(agent, "do_learn_gen_and_pred",
                                    {"agent": self, "u_hash": u_hash, "du_hash": du_hash, "yhat_hash": yhat_hash,
                                     "dhat_hash": dhat_hash, "steps": steps}):
                return True
            else:
                self.err(f"Unable to ask {agent.name} to learn to generate and learn to predict")
                return False

    def __process_streams(self,
                          u_hash: str | None, du_hash: str | None,
                          yhat_hash: str | None, dhat_hash: str | None,
                          skip_gen: bool, skip_pred: bool,
                          steps: int = 100) -> bool:
        """Loop on data streams, for learning and/or generation purposes."""

        # overwriting stream hash in case of playlist
        if u_hash is not None and u_hash == "<playlist>":
            u_hash = self.preferred_streams[self.cur_preferred_stream]
        if du_hash is not None and du_hash == "<playlist>":
            du_hash = self.preferred_streams[self.cur_preferred_stream]
        if yhat_hash is not None and yhat_hash == "<playlist>":
            yhat_hash = self.preferred_streams[self.cur_preferred_stream]
        if dhat_hash is not None and dhat_hash == "<playlist>":
            dhat_hash = self.preferred_streams[self.cur_preferred_stream]

        # getting current step index
        k = self.get_action_step()

        # checking data and creating new buffered streams
        if k == 0:

            # checking data
            if u_hash is not None and u_hash not in self.known_streams:
                self.err(f"Unknown stream (u_hash): {u_hash}")
                return False
            if du_hash is not None and du_hash not in self.known_streams:
                self.err(f"Unknown stream (du_hash): {du_hash}")
                return False
            if yhat_hash is not None and yhat_hash not in self.known_streams:
                self.err(f"Unknown stream (yhat_hash): {yhat_hash}")
                return False
            if dhat_hash is not None and dhat_hash not in self.known_streams:
                self.err(f"Unknown stream (dhat_hash): {dhat_hash}")
                return False
            if u_hash is not None and not self.known_streams[u_hash].enabled:
                self.err(f"Disabled stream (u_hash): {u_hash}")
                return False
            if du_hash is not None and not self.known_streams[du_hash].enabled:
                self.err(f"Disabled stream (du_hash): {du_hash}")
                return False
            if yhat_hash is not None and not self.known_streams[yhat_hash].enabled:
                self.err(f"Disabled stream (yhat_hash): {yhat_hash}")
                return False
            if dhat_hash is not None and not self.known_streams[dhat_hash].enabled:
                self.err(f"Disabled stream (dhat_hash): {dhat_hash}")
                return False

            if skip_gen and (yhat_hash is None or u_hash is not None or du_hash is not None):
                self.err(f"Invalid request: you are asking to skip_gen by either not providing a yhat-stream or "
                         f"providing a u-stream/du_hash (that would not be used).")
                return False
            if skip_pred and dhat_hash is None:
                self.err(f"Invalid request: you are asking to skip_pred but not providing "
                         f"a dhat-stream (that is needed, since it will be copied in place of the predictor).")
                return False

            if steps <= 0:
                self.err(f"Invalid number of steps: {steps}")
                return False

            # getting existing streams
            u_stream = self.known_streams[u_hash] if u_hash is not None else None
            du_stream = self.known_streams[du_hash] if du_hash is not None else None
            yhat_stream = self.known_streams[yhat_hash] if yhat_hash is not None else None
            dhat_stream = self.known_streams[dhat_hash] if dhat_hash is not None else None

            # getting right offsets for buffered streams
            offset_u = 0
            offset_du = 0
            offset_yhat = 0
            offset_dhat = 0
            if u_stream is not None and isinstance(u_stream, BufferedStream):
                if steps > len(u_stream):
                    self.err(f"Cannot process stream {u_stream} for {steps} steps, since it is is shorter")
                offset_u = u_stream.get_first_step_offset_given_current_step()
                self.buffered_streams_offsets[u_stream.get_hash()] = offset_u
            if du_stream is not None and isinstance(du_stream, BufferedStream):
                if steps > len(du_stream):
                    self.err(f"Cannot process stream {du_stream} for {steps} steps, since it is is shorter")
                offset_du = du_stream.get_first_step_offset_given_current_step()
                self.buffered_streams_offsets[du_stream.get_hash()] = offset_du
            if yhat_stream is not None and isinstance(yhat_stream, BufferedStream):
                if steps > len(yhat_stream):
                    self.err(f"Cannot process stream {yhat_stream} for {steps} steps, since it is is shorter")
                offset_yhat = yhat_stream.get_first_step_offset_given_current_step()
                self.buffered_streams_offsets[yhat_stream.get_hash()] = offset_yhat
            if dhat_stream is not None and isinstance(dhat_stream, BufferedStream):
                if steps > len(dhat_stream):
                    self.err(f"Cannot process stream {dhat_stream} for {steps} steps, since it is is shorter")
                offset_dhat = dhat_stream.get_first_step_offset_given_current_step()
                self.buffered_streams_offsets[dhat_stream.get_hash()] = offset_dhat

            # increasing the number of the last generated stream ("generated1", "generated2", ...)
            self.last_generated_stream_num += 1

            # creating new buffered streams to store the data received as inputs (for visualization purposes only)
            yhatdhat_stream = BufferedStream(use_static_descriptor=False)
            yhatdhat_stream.set_name("target" + str(self.last_generated_stream_num))
            yhatdhat_stream.set_creator(f"{self.name}")
            yhatdhat_stream.set_meta("Visualization purposes only: the yhat and dhat data used in a "
                                     "generation procedure")
            yhatdhat_stream.attributes[0] = yhat_stream.attributes[0] \
                if yhat_stream is not None else yhatdhat_stream.attributes[0]
            yhatdhat_stream.attributes[1] = dhat_stream.attributes[1] \
                if dhat_stream is not None else yhatdhat_stream.attributes[1]

            # creating a new buffered stream to store the data that will be generated
            yd_stream = BufferedStream(use_static_descriptor=False)
            yd_stream.set_name("generated" + str(self.last_generated_stream_num))
            yd_stream.set_creator(f"{self.name}")
            yd_stream.set_meta("Stream generated by the agent")
            yd_stream.attributes = self.model.attributes  # getting attributes from the model

            # storing a reference to the just generated streams
            self.known_streams[yhatdhat_stream.get_hash()] = yhatdhat_stream
            self.known_streams[yd_stream.get_hash()] = yd_stream
        else:

            # just getting all streams
            u_stream = self.known_streams[u_hash] if u_hash is not None else None
            du_stream = self.known_streams[du_hash] if du_hash is not None else None
            yhat_stream = self.known_streams[yhat_hash] if yhat_hash is not None else None
            dhat_stream = self.known_streams[dhat_hash] if dhat_hash is not None else None
            yhatdhat_stream = (
                self.known_streams)[Stream.build_hash("target" + str(self.last_generated_stream_num), self.name)]
            yd_stream = (
                self.known_streams)[Stream.build_hash("generated" + str(self.last_generated_stream_num), self.name)]

            # getting right offsets for buffered streams
            offset_u = 0
            offset_du = 0
            offset_yhat = 0
            offset_dhat = 0
            if u_stream is not None and isinstance(u_stream, BufferedStream):
                offset_u = self.buffered_streams_offsets[u_stream.get_hash()]
            if du_stream is not None and isinstance(du_stream, BufferedStream):
                offset_du = self.buffered_streams_offsets[du_stream.get_hash()]
            if yhat_stream is not None and isinstance(yhat_stream, BufferedStream):
                offset_yhat = self.buffered_streams_offsets[yhat_stream.get_hash()]
            if dhat_stream is not None and isinstance(dhat_stream, BufferedStream):
                offset_dhat = self.buffered_streams_offsets[dhat_stream.get_hash()]

        # streams at current time
        u = u_stream[u_stream.k + offset_u][0] if u_stream is not None else None
        du = du_stream[du_stream.k + offset_du][1] if du_stream is not None else None
        yhat = yhat_stream[yhat_stream.k + offset_yhat][0] if yhat_stream is not None else None
        dhat = dhat_stream[dhat_stream.k + offset_dhat][1] if dhat_stream is not None else None

        # generate output
        y, d = self.model(u=u, du=du,
                          y=yhat if skip_gen else None, d=dhat if skip_pred else None,
                          first=(k == 0))

        # buffer data to the streams
        yhatdhat_stream.append_data(yhat.cpu().detach() if yhat is not None and not skip_gen else None,
                                    dhat.cpu().detach() if dhat is not None and not skip_pred else None)
        yd_stream.append_data(y.cpu().detach(),
                              d.cpu().detach())

        # learn
        if (not skip_gen and yhat_stream is not None) or (not skip_pred and dhat_stream is not None):
            self.model.learn(y=y, yhat=yhat if not skip_gen else None,
                             d=d, dhat=dhat if not skip_pred else None)

        return True

    def __compare_streams(self, stream_a_hash: str, stream_b_hash: str,
                          what: str = "y", how: str = "mse", steps: int = 100) \
            -> tuple[float, bool]:
        """Loop on two -buffered- data streams, for comparison purposes, returning a value >= 0."""

        if stream_a_hash not in self.known_streams:
            self.err(f"Unknown stream (stream_a_hash): {stream_a_hash}")
            return -1., False

        if stream_b_hash not in self.known_streams:
            self.err(f"Unknown stream (stream_b_hash): {stream_b_hash}")
            return -1., False

        if what not in ["y", "d"]:
            self.err(f"Only two types of information can be compared: 'y' and 'd' (unknown: {what})")
            return -1., False

        if steps <= 0:
            self.err(f"Invalid number of steps: {steps}")
            return -1., False

        if how not in ["mse", "max"] and not how.startswith("geq"):
            self.err(f"Data can be compared by MSE, or by comparing the argmax ('max'), or comparing the number "
                     f"of corresponding bits (obtained by 'geqX', where 'X' is a number). Unknown: {how})")
            return -1., False

        stream_a = self.known_streams[stream_a_hash]
        stream_b = self.known_streams[stream_b_hash]
        z = 0 if what == "y" else 1

        # increasing the number of the last generated stream ("generated1", "generated2", ...)
        self.last_eval_stream_num += 1

        # creating new buffered streams to store the evaluated data (for visualization purposes only)
        eva_stream = BufferedStream(use_static_descriptor=False)
        eva_stream.set_name("eval" + str(self.last_eval_stream_num))
        eva_stream.set_creator(f"{self.name}")
        eva_stream.set_meta("Visualization purposes only: the evaluated data stream")
        eva_stream.attributes = stream_a.attributes
        eva_stream.set_first_step(stream_a.get_first_step())

        # creating a new buffered stream to store the expected/target data (for visualization purposes only)
        exp_stream = BufferedStream(use_static_descriptor=False)
        exp_stream.set_name("expect" + str(self.last_eval_stream_num))
        exp_stream.set_creator(f"{self.name}")
        exp_stream.set_meta("Visualization purposes only: the expected (target) data stream")
        exp_stream.attributes = stream_b.attributes
        exp_stream.set_first_step(stream_a.get_first_step())  # it must be "exp_stream" and "stream_a." (NOT stream_b.)

        # storing a reference to the just generated streams
        self.known_streams[eva_stream.get_hash()] = eva_stream
        self.known_streams[exp_stream.get_hash()] = exp_stream

        if not isinstance(stream_a, BufferedStream):
            self.err(f"Can only compare buffered streams and {stream_a_hash} is not buffered")
            return -1., False

        if not isinstance(stream_b, BufferedStream):
            self.err(f"Can only compare buffered streams and {stream_b_hash} is not buffered")
            return -1., False

        if steps > len(stream_a) or steps > len(stream_a):
            self.err(f"Cannot compare streams for {steps} steps, since at least one of them is shorter")

        a_first_k = stream_a.get_first_step()
        b_first_k = stream_b.get_first_step()

        # comparing data (averaging)
        o = 0.
        for k in range(0, steps):

            # signals or descriptors
            a = stream_a[a_first_k + k]
            b = stream_b[b_first_k + k]

            # checking
            if a[z] is None or b[z] is None:
                self.err("Cannot compare signals/descriptors if one or both of them are None")
                return -1., False

            # comparing
            o = o + self.model.compare(a[z], b[z], how)

            # storing (visualization purposes only)
            eva_stream.append_data(a[0], a[1], instantaneous=True)
            exp_stream.append_data(b[0], b[1], instantaneous=True)

        return o / steps, True

    def __complete_do(self, do_what: str, agent: Self, returned: bool):
        """Post action to run after at the end of a do_something call, to confirm it."""

        assert do_what in ["gen", "pred", "gen_and_pred", "learn_gen", "learn_pred", "learn_gen_and_pred"]

        if returned is True:

            # getting generated stream
            stream_hash = Stream.build_hash("generated" + str(self.last_generated_stream_num), self.name)

            if stream_hash not in self.known_streams:
                self.err(f"Unknown stream: {stream_hash}")
                self.failed_communicating_completion = do_what
                return False

            stream = self.known_streams[stream_hash]
            if (do_what == "gen" or do_what == "gen_and_pred") and isinstance(stream, BufferedStream):
                y_text, _ = stream.to_text(length=200)
                if y_text is not None:
                    self.out("Generated: \"" + y_text + "\"")

            # confirming
            if self.set_next_action(agent, "done_" + do_what,
                                    {"agent": self, "streams": {stream_hash: self.known_streams[stream_hash]}}):
                self.failed_communicating_completion = None
                return True
            else:
                self.err(f"Unable to confirm '{do_what}' to {agent.name}")
                self.failed_communicating_completion = do_what
                return False
        else:
            self.failed_communicating_completion = None
            return False
