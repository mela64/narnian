from narnian.server import Server
from narnian.streams import Stream
from basic.basic_streams import Square, CombSin
from basic.basic_agent import BasicAgent
from basic.basic_model import BasicModel
from basic.basic_hl_model import BasicHLModel
from basic.basic_environment import BasicEnvironment
import torch
from narnian.attributes import Attributes


class HFLACombSin(CombSin):
    def __init__(self):
        freqs = [0.276, 0.149, 0.053]
        coeffs = [0.4, 0.08, 0.08]
        super().__init__(f_cap=freqs, c_cap=coeffs, delta=0.1, order=3)
        self.attributes[1] = Attributes((3,), ['3sin', 'hf', 'la'], labeling_rule="geq0.5")
        self.static_d = torch.ones((1, 3))


class HFHACombSin(CombSin):
    def __init__(self):
        freqs = [0.276, 0.149, 0.053]
        coeffs = [1.4, 0.2, 0.2]
        super().__init__(f_cap=freqs, c_cap=coeffs, delta=0.1, order=3)
        self.attributes[1] = Attributes((3,), ['3sin', 'hf', 'ha'], labeling_rule="geq0.5")
        self.static_d = torch.ones((1, 3))


class LFLACombSin(CombSin):
    def __init__(self):
        freqs = [0.276, 0.149, 0.053]
        coeffs = [0.08, 0.08, 0.4]
        super().__init__(f_cap=freqs, c_cap=coeffs, delta=0.1, order=3)
        self.attributes[1] = Attributes((3,), ['3sin', 'lf', 'la'], labeling_rule="geq0.5")
        self.static_d = torch.ones((1, 3))


class LFHACombSin(CombSin):
    def __init__(self):
        freqs = [0.276, 0.149, 0.053]
        coeffs = [0.2, 0.2, 1.4]
        super().__init__(f_cap=freqs, c_cap=coeffs, delta=0.1, order=3)
        self.attributes[1] = Attributes((3,), ['3sin', 'lf', 'ha'], labeling_rule="geq0.5")
        self.static_d = torch.ones((1, 3))


class HFHASquare(Square):
    def __init__(self):
        super().__init__(freq=0.1, phase=0.5, delta=0.1, ampl=1.5)
        self.attributes[1] = Attributes((3,), ['square', 'hf', 'ha'], labeling_rule="geq0.5")
        self.static_d = torch.ones((1, 3))


class HFLASquare(Square):
    def __init__(self):
        super().__init__(freq=0.1, phase=0.5, delta=0.1, ampl=0.5)
        self.attributes[1] = Attributes((3,), ['square', 'hf', 'la'], labeling_rule="geq0.5")
        self.static_d = torch.ones((1, 3))


class LFHASquare(Square):
    def __init__(self):
        super().__init__(freq=0.03, phase=0.5, delta=0.1, ampl=1.5)
        self.attributes[1] = Attributes((3,), ['square', 'lf', 'ha'])
        self.static_d = torch.ones((1, 3))


class LFLASquare(Square):
    def __init__(self):
        super().__init__(freq=0.03, phase=0.5, delta=0.1, ampl=0.5)
        self.attributes[1] = Attributes((3,), ['square', 'lf', 'la'])
        self.static_d = torch.ones((1, 3))


# creating environment
env = BasicEnvironment("Feature generalization")

# adding streams to the environment
env.add_stream(Stream.create(name="3sinhfha", creator="envir", stream=HFHACombSin()))
env.add_stream(Stream.create(name="3sinhfla", creator="envir", stream=HFLACombSin()))
env.add_stream(Stream.create(name="3sinlfha", creator="envir", stream=LFHACombSin()))
env.add_stream(Stream.create(name="3sinlfla", creator="envir", stream=LFLACombSin()))
env.add_stream(Stream.create(name="squarehfha", creator="envir", stream=HFHASquare()))
env.add_stream(Stream.create(name="squarehfla", creator="envir", stream=HFLASquare()))
env.add_stream(Stream.create(name="squarelfha", creator="envir", stream=LFHASquare()))
env.add_stream(Stream.create(name="squarelfla", creator="envir", stream=LFLASquare()))

# modeling behaviour of the environment
env.behav.add_transit("init", "streams_enabled", action="enable_all_streams")
env.behav.add_transit("streams_enabled", "streams_sent", action="send_streams_to_all")
env.behav.add_transit("streams_sent", "ready", action="send_agents_to_all")
env.behav.set_state("init")
env.behav.save(f"{env.name}.json")
env.behav.save_pdf(f"{env.name}.pdf")

# creating teacher agent
ag = BasicAgent("teacher", model=BasicModel(attributes=env.shared_attributes, lr=0.), authority=1.0)
ag.behav.add_transit("init", "got_streams", action="get_streams")
ag.behav.add_transit("got_streams", "got_agents", action="get_agents")
ag.behav.add_transit("got_agents", "recording1", action="record", args={"stream_hash": "envir:3sinhfha", "steps": 1000})
ag.behav.add_transit("recording1", "recording2", action="record", args={"stream_hash": "envir:3sinhfla", "steps": 1000})
ag.behav.add_transit("recording2", "recording3", action="record", args={"stream_hash": "envir:3sinlfha", "steps": 1000})
ag.behav.add_transit("recording3", "recording4", action="record", args={"stream_hash": "envir:3sinlfla", "steps": 1000})
ag.behav.add_transit("recording4", "recording5", action="record", args={"stream_hash": "envir:squarehfha", "steps": 1000})
ag.behav.add_transit("recording5", "recording6", action="record", args={"stream_hash": "envir:squarehfla", "steps": 1000})
ag.behav.add_transit("recording6", "recording7", action="record", args={"stream_hash": "envir:squarelfha", "steps": 1000})
ag.behav.add_transit("recording7", "recording8", action="record", args={"stream_hash": "envir:squarelfla", "steps": 1000})
ag.behav.add_transit("recording8", "playlist_ready", action="set_pref_streams",
                     args={"stream_hashes": ["teacher:recorded1", "teacher:recorded2", "teacher:recorded3", "teacher:recorded4",
                                             "teacher:recorded5", "teacher:recorded6", "teacher:recorded7", "teacher:recorded8"],
                           "repeat": 2})
ag.behav.add_state_action("playlist_ready", action="find_agent_to_engage", args={"min_auth": 0.0, "max_auth": 0.0})
ag.behav.add_transit("playlist_ready", "student_found", action="send_engagement")
ag.behav.add_transit("student_found", "playlist_ready", action="nop")
ag.behav.add_transit("student_found", "student_engaged", action="got_engagement")
ag.behav.add_transit("student_engaged", "stream_shared", action="share_streams")
ag.behav.add_transit("stream_shared", "asked_learn", action="ask_learn_gen",
                     args={"du_hash": "<playlist>", "yhat_hash": "<playlist>", "dhat_hash": "<playlist>",
                           "ask_steps": 1000})
ag.behav.add_transit("asked_learn", "done_learn", action="done_learn_gen")
ag.behav.add_state_action("done_learn", action="next_pref_stream")
# todo qui l'ho messo come sandbox_buffered, sta facendo la playlist ma solo perchè volevo vedere se gli 8 segnali li riusciva a generare
ag.behav.add_transit("done_learn", "stream_shared", action="check_pref_stream", args={"what": "not_last_round"})
ag.behav.add_transit("done_learn", "ready_to_ask", action="check_pref_stream", args={"what": "last_round"})
# add a final unsupervised generation for each signal
ag.behav.add_transit("ready_to_ask", "asked_gen", action="ask_gen",
                     args={"du_hash": "<playlist>",  "dhat_hash": "<playlist>", "ask_steps": 1000})
ag.behav.add_transit("asked_gen", "done_gen", action="done_gen")
ag.behav.add_state_action("done_gen", action="next_pref_stream")
ag.behav.add_transit("done_gen", "ready_to_ask", action="check_pref_stream", args={"what": "not_first"})
ag.behav.add_transit("done_gen", "finished", action="check_pref_stream", args={"what": "first"})
ag.behav.set_state("init")
ag.behav.save(f"{ag.name}.json")
ag.behav.save_pdf(f"{ag.name}.pdf")

# adding agent to environment
env.add_agent(ag)

# creating student agent
# ag = BasicAgent("student", model=BasicModel(attributes=env.shared_attributes, lr=0.001), authority=0.0)
ag = BasicAgent("student", model=BasicHLModel(attributes=env.shared_attributes, lr=0.001,
                                              delta=0.1, cnu_memories=8), authority=0.0)
ag.behav.add_transit("init", "got_streams", action="get_streams")
ag.behav.add_transit("got_streams", "got_agents", action="get_agents")
ag.behav.add_transit("got_agents", "teacher_engaged", action="get_engagement", args={"min_auth": 1.0, "max_auth": 1.0})
ag.behav.add_transit("teacher_engaged", "got_teacher_streams", action="get_streams")
ag.behav.add_transit("got_teacher_streams", "learning", action="do_learn_gen")
ag.behav.add_transit("got_teacher_streams", "generated", action="do_gen")
ag.behav.add_transit("learning", "got_teacher_streams", action="nop")
ag.behav.add_transit("generated", "got_teacher_streams", action="nop")
ag.behav.set_state("init")
ag.behav.save(f"{ag.name}.json")
ag.behav.save_pdf(f"{ag.name}.pdf")

# adding agent to environment
env.add_agent(ag)

# printing
print(env)
for ag in env.agents.values():
    print(ag)

# creating server
Server(env=env, port=5002)

# running
env.run()
