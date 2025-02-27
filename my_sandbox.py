from narnian.server import Server
from narnian.streams import Stream, Random
from basic.basic_streams import Sin, Square
from basic.basic_agent import BasicAgent
from basic.basic_model import BasicModel
from basic.basic_environment import BasicEnvironment


# creating environment
env = BasicEnvironment("sandbox")

# registering streams
Stream.register(name="sin", stream_class=Sin, stream_class_args={"freq": 0.159, "phase": 0.5, "delta": 0.1})
Stream.register(name="square", stream_class=Square, stream_class_args={"freq": 0.159, "phase": 0.5, "delta": 0.1})
Stream.register(name="random", stream_class=Random, stream_class_args={"std": 1.0, "shape": (1,)})

# adding streams to the environment
env.add_stream(Stream.create(name="sin", creator="envir"))
env.add_stream(Stream.create(name="square", creator="envir"))
env.add_stream(Stream.create(name="random", creator="envir"))

# modeling behaviour of the environment
env.behav.add_transit("init", "streams_enabled", action="enable_all_streams", score=1.0)
env.behav.add_transit("streams_enabled", "streams_sent", action="send_streams_to_all", score=1.0)
env.behav.add_transit("streams_sent", "ready", action="send_agents_to_all", score=1.0)
env.behav.set_state("init")
env.behav.save(f"{env.name}.json")
env.behav.save_pdf(f"{env.name}.pdf")

# creating teacher agent
ag = BasicAgent("teacher", model=BasicModel(lr=0.), authority=1.0)
ag.behav.add_transit("init", "got_streams", action="get_streams")
ag.behav.add_transit("got_streams", "got_agents", action="get_agents", score=1.0)
ag.behav.add_transit("got_agents", "playlist_ready", action="set_pref_streams",
                     args={"stream_hashes": ["envir:sin", "envir:square"]}, score=1.0)
ag.behav.add_state_action("playlist_ready", action="find_agent_to_engage", args={"min_auth": 0.0, "max_auth": 0.0})
ag.behav.add_transit("playlist_ready", "student_found", action="send_engagement", score=1.0)
ag.behav.add_transit("student_found", "playlist_ready", action="nop", score=1.0)
ag.behav.add_transit("student_found", "student_engaged", action="got_engagement", score=0.0)
ag.behav.add_transit("student_engaged", "asked_learn", action="ask_learn_gen",
                     args={"du_hash": "<playlist>", "yhat_hash": "<playlist>", "ask_steps": 200}, score=1.0)
ag.behav.add_transit("asked_learn", "done_learn", action="done_learn_gen", score=0.0)
ag.behav.add_transit("done_learn", "asked_gen", action="ask_gen", args={"du_hash": "<playlist>", "ask_steps": 50})
ag.behav.add_transit("asked_gen", "done_gen", action="done_gen", score=0.0)
ag.behav.add_state_action("done_gen", action="eval", args={"stream_hash": "<playlist>", "what": "y", "steps": 50})
ag.behav.add_transit("done_gen", "student_engaged", action="compare_eval", args={"cmp": "<", "thres": 0.5}, score=1.0)
ag.behav.add_transit("done_gen", "good", action="compare_eval", args={"cmp": ">=", "thres": 0.5}, score=1.0)
ag.behav.add_state_action("good", action="next_pref_stream")
ag.behav.add_transit("good", "student_engaged", action="check_pref_stream", args={"what": "not_first"}, score=1.0)
ag.behav.add_transit("good", "finished", action="check_pref_stream", args={"what": "first"}, score=1.0)
ag.behav.set_state("init")
ag.behav.save(f"{ag.name}.json")
ag.behav.save_pdf(f"{ag.name}.pdf")

# adding agent to environment
env.add_agent(ag)

# creating student agent
ag = BasicAgent("student", model=BasicModel(lr=0.0001), authority=0.0)
ag.behav.add_transit("init", "got_streams", action="get_streams")
ag.behav.add_transit("got_streams", "got_agents", action="get_agents")
ag.behav.add_transit("got_agents", "teacher_engaged", action="get_engagement", args={"min_auth": 1.0, "max_auth": 1.0})
ag.behav.add_transit("teacher_engaged", "learning", action="do_learn_gen", score=0.0)
ag.behav.add_transit("teacher_engaged", "generated", action="do_gen", score=0.0)
ag.behav.add_transit("learning", "teacher_engaged", action="nop", score=1.0)
ag.behav.add_transit("generated", "teacher_engaged", action="nop", score=1.0)
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
Server(env=env)

# running
env.run()
