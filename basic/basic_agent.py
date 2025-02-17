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
        self.target_agent = None  # another agent which is available for possible engagement
        self.engaged_agent = None  # agent involved in current interactions
        self.received_hash = None  # stream sent (e.g., generated) by another agent
        self.eval_result = None  # result of the last evaluation
        self.preferred_streams = []  # list of preferred streams
        self.cur_preferred_stream = 0  # id of the current preferred stream from the list
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
        for agent in self.known_agents.values():
            if agent.available and min_auth <= agent.authority <= max_auth:
                self.target_agent = agent
        return True

    def send_engagement(self):
        """Offer engagement ot another agent."""

        self.out(f"Sending engagement request to {self.target_agent.name}")
        if self.target_agent.behav.set_action_score("get_engagement", new_score='top'):
            self.target_agent.behav.set_buffer_param_value("agent", self)
            return True
        else:
            self.err(f"Unable to send engagement to {self.target_agent.name}")
            return False

    def get_engagement(self, agent: Self, min_auth: float, max_auth: float):
        """Receive engagement from another agent whose authority is in the specified range."""

        self.out(f"Getting engagement from {agent.name}")
        if agent.name not in self.known_agents:
            self.err(f"Unknown agent: {agent.name}")
            return False

        # confirming
        if self.available and min_auth <= agent.authority <= max_auth:
            if agent.behav.set_action_score("got_engagement", new_score='top'):
                agent.behav.set_buffer_param_value("agent", self)
                self.target_agent = None
                self.engaged_agent = agent
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
        if agent.name == self.target_agent.name:
            self.engaged_agent = agent
            self.target_agent = None
            self.available = False
            return True
        else:
            self.err(f"Unable to confirm engagement with {agent.name}")
            return False

    def gen(self, u_hash: str | None = None, du_hash: str | None = None, steps: int = 100) -> bool:
        """Generate a signal."""

        self.out(f"Generating signal")
        return self.__process_streams(u_hash=u_hash, du_hash=du_hash,
                                      yhat_hash=None, dhat_hash=None,
                                      skip_gen=False, skip_pred=True,
                                      steps=steps)

    def ask_gen(self, u_hash: str | None = None, du_hash: str | None = None, steps: int = 100):
        """Asking for generation."""

        if self.engaged_agent is None:
            self.err("Not engaged to any agents")
            return False

        self.out(f"Asking {self.engaged_agent.name} to generate signal")
        return self.__ask(for_what="gen", agent=self.engaged_agent,
                          u_hash=u_hash, du_hash=du_hash,
                          yhat_hash=None, dhat_hash=None,
                          steps=steps)

    def do_gen(self, u_hash: str | None = None, du_hash: str | None = None, steps: int = 100) -> bool:
        """Generate a signal."""

        if self.engaged_agent is None:
            self.err("Not engaged to any agents")
            return False

        ret = self.gen(u_hash, du_hash, steps)
        return self.__complete_do("gen", self.engaged_agent, ret)

    def done_gen(self, streams: dict):
        """Confirm generation."""

        if self.engaged_agent is None:
            self.err("Not engaged to any agents")
            return False

        self.out(f"Agent {self.engaged_agent.name} finished generation")
        return self.__done(streams)

    def pred(self, yhat_hash: str, steps: int = 100) -> bool:
        """Predict a descriptor."""

        self.out(f"Predicting descriptor")
        return self.__process_streams(u_hash=None, du_hash=None,
                                      yhat_hash=yhat_hash, dhat_hash=None,
                                      skip_gen=True, skip_pred=False,
                                      steps=steps)

    def ask_pred(self, yhat_hash: str, steps: int = 100):
        """Asking for prediction of a descriptor."""

        if self.engaged_agent is None:
            self.err("Not engaged to any agents")
            return False

        self.out(f"Asking {self.engaged_agent.name} to predict descriptor")
        return self.__ask(for_what="pred", agent=self.engaged_agent,
                          u_hash=None, du_hash=None,
                          yhat_hash=yhat_hash, dhat_hash=None,
                          steps=steps)

    def do_pred(self, yhat_hash: str, steps: int = 100) -> bool:
        """Predict a descriptor."""

        if self.engaged_agent is None:
            self.err("Not engaged to any agents")
            return False

        ret = self.pred(yhat_hash, steps)
        return self.__complete_do("pred", self.engaged_agent, ret)

    def done_pred(self, streams: dict):
        """Confirm prediction."""

        if self.engaged_agent is None:
            self.err("Not engaged to any agents")
            return False

        self.out(f"Agent {self.engaged_agent.name} finished predicting descriptor")
        return self.__done(streams)

    def gen_and_pred(self, u_hash: str | None = None, du_hash: str | None = None, steps: int = 100) -> bool:
        """Generate a signal and predict a descriptor."""

        self.out(f"Generating signal and predicting descriptor")
        return self.__process_streams(u_hash=u_hash, du_hash=du_hash,
                                      yhat_hash=None, dhat_hash=None,
                                      skip_gen=False, skip_pred=False,
                                      steps=steps)

    def ask_gen_and_pred(self, u_hash: str | None = None, du_hash: str | None = None, steps: int = 100):
        """Asking for generation and prediction of descriptor."""

        if self.engaged_agent is None:
            self.err("Not engaged to any agents")
            return False

        self.out(f"Asking {self.engaged_agent.name} to generate signal and predict descriptor")
        return self.__ask(for_what="gen_and_pred", agent=self.engaged_agent,
                          u_hash=u_hash, du_hash=du_hash,
                          yhat_hash=None, dhat_hash=None,
                          steps=steps)

    def do_gen_and_pred(self,
                        u_hash: str | None = None, du_hash: str | None = None, steps: int = 100) -> bool:
        """Generate a signal and predict a descriptor."""

        if self.engaged_agent is None:
            self.err("Not engaged to any agents")
            return False

        ret = self.gen_and_pred(u_hash, du_hash, steps)
        return self.__complete_do("gen_and_pred", self.engaged_agent, ret)

    def done_gen_and_pred(self, streams: dict):
        """Confirm generation and prediction."""

        if self.engaged_agent is None:
            self.err("Not engaged to any agents")
            return False

        self.out(f"Agent {self.engaged_agent.name} finished generation and prediction of descriptor")
        return self.__done(streams)

    def learn_gen(self, yhat_hash: str, u_hash: str | None = None, du_hash: str | None = None,
                  steps: int = 100) -> bool:
        """Learn to generate a signal."""

        self.out(f"Learning to generate signal")
        return self.__process_streams(u_hash=u_hash, du_hash=du_hash,
                                      yhat_hash=yhat_hash, dhat_hash=None,
                                      skip_gen=False, skip_pred=True,
                                      steps=steps)

    def ask_learn_gen(self, yhat_hash: str, u_hash: str | None = None, du_hash: str | None = None, steps: int = 100):
        """Asking for learning to generate."""

        if self.engaged_agent is None:
            self.err("Not engaged to any agents")
            return False

        self.out(f"Asking {self.engaged_agent.name} to learn to generate signal")
        return self.__ask(for_what="learn_gen", agent=self.engaged_agent,
                          u_hash=u_hash, du_hash=du_hash,
                          yhat_hash=yhat_hash, dhat_hash=None,
                          steps=steps)

    def do_learn_gen(self, yhat_hash: str,
                     u_hash: str | None = None, du_hash: str | None = None, steps: int = 100) -> bool:
        """Learn to generate a signal."""

        if self.engaged_agent is None:
            self.err("Not engaged to any agents")
            return False

        ret = self.learn_gen(yhat_hash, u_hash, du_hash, steps)
        return self.__complete_do("learn_gen", self.engaged_agent, ret)

    def done_learn_gen(self, streams: dict):
        """Confirm learning to generate."""

        if self.engaged_agent is None:
            self.err("Not engaged to any agents")
            return False

        self.out(f"Agent {self.engaged_agent.name} finished learning to generate")
        return self.__done(streams)

    def learn_pred(self, yhat_hash: str, dhat_hash: str, steps: int = 100) -> bool:
        """Learn to predict a descriptor."""

        self.out(f"Learning to predict descriptor")
        return self.__process_streams(u_hash=None, du_hash=None,
                                      yhat_hash=yhat_hash, dhat_hash=dhat_hash,
                                      skip_gen=True, skip_pred=False,
                                      steps=steps)

    def ask_learn_pred(self,
                       yhat_hash: str, dhat_hash: str,
                       steps: int = 100):
        """Asking for learning to predict descriptor."""

        if self.engaged_agent is None:
            self.err("Not engaged to any agents")
            return False

        self.out(f"Asking {self.engaged_agent.name} to learn to predict descriptor")
        return self.__ask(for_what="learn_pred", agent=self.engaged_agent,
                          u_hash=None, du_hash=None,
                          yhat_hash=yhat_hash, dhat_hash=dhat_hash,
                          steps=steps)

    def do_learn_pred(self, yhat_hash: str, dhat_hash: str, steps: int = 100) -> bool:
        """Learn to predict a descriptor."""

        if self.engaged_agent is None:
            self.err("Not engaged to any agents")
            return False

        ret = self.learn_pred(yhat_hash, dhat_hash, steps)
        return self.__complete_do("learn_pred", self.engaged_agent, ret)

    def done_learn_pred(self, streams: dict):
        """Confirm learning to predict descriptor."""

        if self.engaged_agent is None:
            self.err("Not engaged to any agents")
            return False

        self.out(f"Agent {self.engaged_agent.name} finished learning to predict descriptor")
        return self.__done(streams)

    def learn_gen_and_pred(self, yhat_hash: str, dhat_hash: str,
                           u_hash: str | None = None, du_hash: str | None = None, steps: int = 100) -> bool:
        """Learn to generate a signal and predict a descriptor."""

        self.out(f"Learning to generate signal and predict descriptor")
        return self.__process_streams(u_hash=u_hash, du_hash=du_hash,
                                      yhat_hash=yhat_hash, dhat_hash=dhat_hash,
                                      skip_gen=False, skip_pred=False,
                                      steps=steps)

    def ask_learn_gen_and_pred(self,
                               yhat_hash: str, dhat_hash: str,
                               u_hash: str | None = None, du_hash: str | None = None,
                               steps: int = 100):
        """Asking to learn to generate signal and predict descriptor."""

        if self.engaged_agent is None:
            self.err("Not engaged to any agents")
            return False

        self.out(f"Asking {self.engaged_agent.name} to learn to generate signal and to learn to predict descriptor")
        return self.__ask(for_what="learn_gen_and_pred", agent=self.engaged_agent,
                          u_hash=u_hash, du_hash=du_hash,
                          yhat_hash=yhat_hash, dhat_hash=dhat_hash,
                          steps=steps)

    def do_learn_gen_and_pred(self, yhat_hash: str, dhat_hash: str,
                              u_hash: str | None = None, du_hash: str | None = None, steps: int = 100) -> bool:
        """Learn to generate a signal and predict a descriptor."""

        if self.engaged_agent is None:
            self.err("Not engaged to any agents")
            return False

        ret = self.learn_gen_and_pred(yhat_hash, dhat_hash, u_hash, du_hash, steps)
        return self.__complete_do("learn_gen_and_pred", self.engaged_agent, ret)

    def done_learn_gen_pred(self, streams: dict):
        """Confirm learning to generate and learning to predict descriptor."""

        if self.engaged_agent is None:
            self.err("Not engaged to any agents")
            return False

        self.out(f"Agent {self.engaged_agent.name} finished learning to generate and learning to predict descriptor")
        return self.__done(streams)

    def eval(self, stream_hash: str, what: str, steps: int = 100) -> float:
        """Compare two signals."""

        assert what in ["y", "d"], "Only 'y' and 'd' are allowed for the 'what' argument"

        if stream_hash is not None and stream_hash == "<playlist>":
            stream_hash = self.preferred_streams[self.cur_preferred_stream]

        self.out(f"Comparing {self.received_hash} with {stream_hash} ({what})")
        self.eval_result = self.__compare_streams(stream_a_name=self.received_hash,
                                                  stream_b_name=stream_hash, what=what, steps=steps)
        return True if self.eval_result >= 0. else False

    def compare_eval(self, cmp: str, thres: float) -> bool:
        """After having completed an evaluation."""

        self.out(f"Checking if result {self.eval_result} {cmp} {thres}")
        if self.eval_result < 0. or self.eval_result > 1.:
            self.err(f"Invalid evaluation result: {self.eval_result}")
            return False

        if cmp not in ["<", ">", ">=", "<="]:
            self.err(f"Invalid comparison operator: {cmp}")
            return False

        if thres < 0. or thres > 1.:
            self.err(f"Invalid evaluation threshold: {thres} (it must be in [0, 1])")
            return False

        if cmp == "<" and self.eval_result < thres:
            return True
        elif cmp == "<=" and self.eval_result <= thres:
            return True
        elif cmp == ">" and self.eval_result > thres:
            return True
        elif cmp == ">=" and self.eval_result >= thres:
            return True
        else:
            self.err(f"The evaluation did not meet the expected outcome")
            return False

    def next_pref_stream(self):
        """Moves to the next stream in the list of preferred ones."""

        if len(self.preferred_streams) == 0:
            self.err(f"Cannot move to the next stream because the list of preferred streams is empty")
            return False

        self.cur_preferred_stream = (self.cur_preferred_stream + 1) % len(self.preferred_streams)
        suffix = ", warning: restarted" if self.cur_preferred_stream == 0 else ""
        self.out(f"Moving to the next preferred stream ({self.preferred_streams[self.cur_preferred_stream]}){suffix}")
        return True

    def check_pref_stream(self, what: str = "last") -> bool:
        """Check the current preferred stream."""

        valid = ['first', 'last', 'not_first', 'not_last']
        assert what in valid, f"The what argument can only be one of {valid}"

        self.out(f"Checking if the current preferred playlist item is the {what} one")
        if what == "first":
            return self.cur_preferred_stream == 0
        elif what == "last":
            return self.cur_preferred_stream == len(self.preferred_streams) - 1
        elif what == "not_first":
            return self.cur_preferred_stream != 0
        elif what == "not_last":
            return self.cur_preferred_stream != len(self.preferred_streams) - 1

    def set_pref_streams(self, stream_hashes: list[str]):
        """Fill a list with preferred streams."""

        self.out(f"Setting up a list of {len(stream_hashes)} preferred streams")
        self.cur_preferred_stream = 0
        self.preferred_streams = []
        for stream_hash in stream_hashes:
            self.preferred_streams.append(stream_hash)
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

        if u_hash is not None and u_hash == "<playlist>":
            u_hash = self.preferred_streams[self.cur_preferred_stream]
        if du_hash is not None and du_hash == "<playlist>":
            du_hash = self.preferred_streams[self.cur_preferred_stream]
        if yhat_hash is not None and yhat_hash == "<playlist>":
            yhat_hash = self.preferred_streams[self.cur_preferred_stream]
        if dhat_hash is not None and dhat_hash == "<playlist>":
            dhat_hash = self.preferred_streams[self.cur_preferred_stream]

        # triggering
        if for_what == "gen":
            if agent.behav.set_action_score("do_gen", new_score='top'):
                agent.behav.set_buffer_param_value("u_hash", u_hash)
                agent.behav.set_buffer_param_value("du_hash", du_hash)
                agent.behav.set_buffer_param_value("steps", steps)
                return True
            else:
                self.err(f"Unable to ask {agent.name} to generate")
                return False
        elif for_what == "pred":
            if agent.behav.set_action_score("do_pred", new_score='top'):
                agent.behav.set_buffer_param_value("yhat_hash", yhat_hash)
                agent.behav.set_buffer_param_value("steps", steps)
                return True
            else:
                self.err(f"Unable to ask {agent.name} to predict")
                return False
        elif for_what == "gen_and_pred":
            if agent.behav.set_action_score("do_gen_and_pred", new_score='top'):
                agent.behav.set_buffer_param_value("u_hash", u_hash)
                agent.behav.set_buffer_param_value("du_hash", dhat_hash)
                agent.behav.set_buffer_param_value("steps", steps)
                return True
            else:
                self.err(f"Unable to ask {agent.name} to generated and predict")
                return False
        elif for_what == "learn_gen":
            if agent.behav.set_action_score("do_learn_gen", new_score='top'):
                agent.behav.set_buffer_param_value("u_hash", u_hash)
                agent.behav.set_buffer_param_value("du_hash", du_hash)
                agent.behav.set_buffer_param_value("yhat_hash", yhat_hash)
                agent.behav.set_buffer_param_value("steps", steps)
                return True
            else:
                self.err(f"Unable to ask {agent.name} to learn to generate")
                return False
        elif for_what == "learn_pred":
            if agent.behav.set_action_score("do_learn_pred", new_score='top'):
                agent.behav.set_buffer_param_value("yhat_hash", yhat_hash)
                agent.behav.set_buffer_param_value("dhat_hash", dhat_hash)
                agent.behav.set_buffer_param_value("steps", steps)
                return True
            else:
                self.err(f"Unable to ask {agent.name} to learn to predict")
                return False
        elif for_what == "learn_gen_and_pred":
            if agent.behav.set_action_score("do_learn_gen_and_pred", new_score='top'):
                agent.behav.set_buffer_param_value("u_hash", u_hash)
                agent.behav.set_buffer_param_value("du_hash", du_hash)
                agent.behav.set_buffer_param_value("yhat_hash", yhat_hash)
                agent.behav.set_buffer_param_value("dhat_hash", dhat_hash)
                agent.behav.set_buffer_param_value("steps", steps)
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

        if u_hash is not None and u_hash == "<playlist>":
            u_hash = self.preferred_streams[self.cur_preferred_stream]
        if du_hash is not None and du_hash == "<playlist>":
            du_hash = self.preferred_streams[self.cur_preferred_stream]
        if yhat_hash is not None and yhat_hash == "<playlist>":
            yhat_hash = self.preferred_streams[self.cur_preferred_stream]
        if dhat_hash is not None and dhat_hash == "<playlist>":
            dhat_hash = self.preferred_streams[self.cur_preferred_stream]

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
        if skip_pred and dhat_hash is not None:
            self.err(f"Invalid request: you are asking to skip_pred by providing "
                     f"a dhat-stream (that would not be used).")
            return False

        if steps <= 0:
            self.err(f"Invalid number of steps: {steps}")
            return False

        # getting streams
        u_stream = self.known_streams[u_hash] if u_hash is not None else None
        du_stream = self.known_streams[du_hash] if du_hash is not None else None
        yhat_stream = self.known_streams[yhat_hash] if yhat_hash is not None else None
        dhat_stream = self.known_streams[dhat_hash] if dhat_hash is not None else None

        # creating new buffered streams to store the data received as inputs (for visualization purposes only)
        vis_udu_stream = BufferedStream()
        vis_udu_stream.set_name("vis-udu")
        vis_udu_stream.set_creator(f"{self.name}")
        vis_udu_stream.set_meta("Visualization purposes only: the y-part is the u-input used in a generation procedure,"
                                " while the d-part is the descriptor used an input")

        vis_yhatdhat_stream = BufferedStream()
        vis_yhatdhat_stream.set_name("vis-yhatdhat")
        vis_yhatdhat_stream.set_creator(f"{self.name}")
        vis_yhatdhat_stream.set_meta("Visualization purposes only: the yhat and dhat data used in a "
                                     "generation procedure")

        # creating a new buffered stream to store the data that will be generated
        yd_stream = BufferedStream()
        yd_stream.set_name("generated")
        yd_stream.set_creator(f"{self.name}")
        yd_stream.set_meta("Stream generated by the agent")

        # storing a reference to the just generated streams
        self.known_streams[vis_udu_stream.get_hash()] = vis_udu_stream
        self.known_streams[vis_yhatdhat_stream.get_hash()] = vis_yhatdhat_stream
        self.known_streams[yd_stream.get_hash()] = yd_stream

        # generating data
        for k in range(0, steps):

            # streams
            u = u_stream[k][0] if u_stream is not None else None
            du = du_stream[k][1] if du_stream is not None else None
            yhat = yhat_stream[k][0] if yhat_stream is not None else None
            dhat = dhat_stream[k][1] if dhat_stream is not None else None

            # generate output
            y, d = self.model(u=u, du=du,
                              y=yhat if skip_gen else None, d=dhat if skip_pred else None,
                              first=(k == 0))

            # buffer data to the streams
            vis_udu_stream.append_data(u.detach() if u is not None else None,
                                       du.detach() if du is not None else None)
            vis_yhatdhat_stream.append_data(yhat.detach() if yhat is not None else None,
                                            dhat.detach() if dhat is not None else None)
            yd_stream.append_data(y.detach(),
                                  d.detach())

            # learn
            if (not skip_gen and yhat_stream is not None) or (not skip_pred and dhat_stream is not None):
                self.model.learn(y=y, yhat=yhat, d=d, dhat=dhat)

        return True

    def __done(self, streams: dict):
        """Confirming generation, prediction, learning."""

        assert len(streams) == 1, f"Only one stream is expected (got {len(streams)})"

        # checking confirmation and saving streams
        for stream_hash, stream in streams.items():
            self.known_streams[stream_hash] = stream
            self.received_hash = stream_hash
            return True

    def __compare_streams(self, stream_a_name: str, stream_b_name: str, what: str = "y", steps: int = 100) -> float:
        """Loop on two data stream, for comparison purposes, returning a value in [0,1]."""

        if stream_a_name not in self.known_streams:
            self.err(f"Unknown stream (stream_a_name): {stream_a_name}")
            return -1.

        if stream_b_name not in self.known_streams:
            self.err(f"Unknown stream (stream_b_name): {stream_b_name}")
            return -1.

        if what not in ["y", "d"]:
            self.err(f"Only two types of information can be compared: 'y' and 'd' (unknown: {what})")
            return -1.

        if steps <= 0:
            self.err(f"Invalid number of steps: {steps}")
            return -1.

        stream_a = self.known_streams[stream_a_name]
        stream_b = self.known_streams[stream_b_name]

        z = 0 if what == "y" else 1

        # comparing data (averaging)
        o = 0.
        for k in range(0, steps):

            # signals or descriptors
            a = stream_a[k][z]
            b = stream_b[k][z]

            # checking
            if a is None or b is None:
                self.err("Cannot compare signals/descriptors if one or both of them are None")
                return -1.

            # comparing
            if z == 0:
                o += self.model.compare_y(a, b)
            else:
                o += self.model.compare_d(a, b)

        return o / steps

    def __complete_do(self, do_what: str, agent: Self, returned: bool):
        """Post action to run after at the end of a do_something call, to confirm it."""

        assert do_what in ["gen", "pred", "gen_and_pred", "learn_gen", "learn_pred", "learn_gen_and_pred"]

        if returned is True:

            # getting generated stream
            stream_hash = Stream.build_hash("generated", self.name)

            if stream_hash not in self.known_streams:
                self.err(f"Unknown stream: {stream_hash}")
                return False

            # confirming
            if agent.behav.set_action_score("done_" + do_what, new_score='top'):
                agent.behav.set_buffer_param_value("agent", self)
                agent.behav.set_buffer_param_value("streams", {stream_hash: self.known_streams[stream_hash]})
                return True
            else:
                self.err(f"Unable to confirm '{do_what}' to {agent.name}")
                return False
        else:
            return False
