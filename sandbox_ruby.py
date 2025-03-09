from narnian.server import Server
from narnian.streams import Stream
from basic.basic_agent import BasicAgent
from basic.basic_model import BasicModel
from basic.basic_streams import Sin, Square
from basic.basic_environment import BasicEnvironment


# creating environment
env = BasicEnvironment("Env", title="Ruby's Sandbox")

# adding streams to the environment
env.add_stream(Stream.create(name="sin", creator=env.name, stream=Sin(freq=0.159, phase=0.5, delta=0.1)))
env.add_stream(Stream.create(name="square", creator=env.name, stream=Square(freq=0.159, phase=0.5, delta=0.1)))

# modeling behaviour of the environment
env.behav.add_transit("init", "streams_enabled", action="enable_all_streams")
env.behav.add_transit("streams_enabled", "streams_sent", action="send_streams_to_all")
env.behav.add_transit("streams_sent", "ready", action="send_agents_to_all")

# creating teacher agent
ag = BasicAgent("Teacher", model=BasicModel(attributes=env.shared_attributes, lr=0.), authority=1.0)
ag.behav.add_transit("init", "got_streams", action="get_streams")
ag.behav.add_transit("got_streams", "got_agents", action="get_agents")
ag.behav.add_transit("got_agents", "recording1", action="record",
                     args={"stream_hash": env.name + ":sin", "steps": 20})
ag.behav.add_transit("recording1", "recording2", action="record",
                     args={"stream_hash": env.name + ":square", "steps": 20})
ag.behav.add_transit("recording2", "playlist_ready", action="set_pref_streams",
                     args={"stream_hashes": [ag.name + ":recorded1", ag.name + ":recorded2"], "repeat": 1})
ag.behav.add_state_action("playlist_ready", action="find_agent_to_engage", args={"min_auth": 0.0, "max_auth": 0.0})
ag.behav.add_transit("playlist_ready", "student_found", action="send_engagement")
ag.behav.add_transit("student_found", "playlist_ready", action="nop")
ag.behav.add_transit("student_found", "student_engaged", action="got_engagement")
ag.behav.add_transit("student_engaged", "stream_shared", action="share_streams")
ag.behav.add_transit("stream_shared", "asked_learn", action="ask_learn_gen",
                     args={"du_hash": "<playlist>", "yhat_hash": "<playlist>", "dhat_hash": "<playlist>",
                           "ask_steps": 20})
ag.behav.add_transit("asked_learn", "done_learn", action="done_learn_gen")
ag.behav.add_transit("done_learn", "asked_gen", action="ask_gen",
                     args={"du_hash": "<playlist>",  "dhat_hash": "<playlist>", "ask_steps": 10})
ag.behav.add_transit("asked_gen", "done_gen", action="done_gen")
ag.behav.add_state_action("done_gen", action="eval",
                          args={"stream_hash": "<playlist>", "what": "y", "how": "mse", "steps": 10})
ag.behav.add_transit("done_gen", "stream_shared", action="compare_eval", args={"cmp": ">", "thres": 0.5})
ag.behav.add_transit("done_gen", "good", action="compare_eval", args={"cmp": "<=", "thres": 0.5})
ag.behav.add_state_action("good", action="next_pref_stream")
ag.behav.add_transit("good", "stream_shared", action="check_pref_stream", args={"what": "not_first"})
ag.behav.add_transit("good", "finished", action="check_pref_stream", args={"what": "first"})

# adding agent to environment
env.add_agent(ag)

# creating student agent
ag = BasicAgent("Student", model=BasicModel(attributes=env.shared_attributes, lr=0.01), authority=0.0)
ag.behav.add_transit("init", "got_streams", action="get_streams")
ag.behav.add_transit("got_streams", "got_agents", action="get_agents")
ag.behav.add_transit("got_agents", "teacher_engaged", action="get_engagement", args={"min_auth": 1.0, "max_auth": 1.0})
ag.behav.add_transit("teacher_engaged", "got_teacher_streams", action="get_streams")
ag.behav.add_transit("got_teacher_streams", "learning", action="do_learn_gen")
ag.behav.add_transit("got_teacher_streams", "generated", action="do_gen")
ag.behav.add_transit("learning", "got_teacher_streams", action="nop")
ag.behav.add_transit("generated", "got_teacher_streams", action="nop")

# adding agent to environment
env.add_agent(ag)

# saving
env.save()

# printing
print(env)
for ag in env.agents.values():
    print(ag)

# creating server
Server(env=env, port=5001)

# running
env.run()
