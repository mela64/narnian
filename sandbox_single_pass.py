from narnian.server import Server
from narnian.streams import Stream
from basic.basic_streams import Square, CombSin
from basic.basic_agent import BasicAgent
from basic.basic_model import BasicModel
from basic.basic_hl_model import BasicHLModel
from basic.basic_environment import BasicEnvironment


# creating environment
env = BasicEnvironment("Playlist memorization")

# adding streams to the environment
env.add_stream(Stream.create(name="3sin", creator="envir", stream=CombSin(f_cap=[0.1065, 0.1262, 0.0138],
                                                                          c_cap=[-0.3200, -0.4848,  0.1938],
                                                                          delta=0.1, order=3)))

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
ag.behav.add_transit("got_agents", "recording1", action="record", args={"stream_hash": "envir:3sin", "steps": 5000})
ag.behav.add_state_action("recording1", action="find_agent_to_engage", args={"min_auth": 0.0, "max_auth": 0.0})
ag.behav.add_transit("recording1", "student_found", action="send_engagement")
ag.behav.add_transit("student_found", "student_engaged", action="got_engagement")
ag.behav.add_transit("student_engaged", "stream_shared", action="share_streams")
ag.behav.add_transit("stream_shared", "asked_learn", action="ask_learn_gen",
                     args={"du_hash": "teacher:recorded1", "yhat_hash": "teacher:recorded1", "dhat_hash": "teacher:recorded1",
                           "ask_steps": 1500})
ag.behav.add_transit("asked_learn", "done_learn", action="done_learn_gen")
ag.behav.add_transit("done_learn", "asked_gen", action="ask_gen",
                     args={"du_hash": "teacher:recorded1", "dhat_hash": "teacher:recorded1", "ask_steps": 1500})
ag.behav.add_transit("asked_gen", "done_gen", action="done_gen")
ag.behav.add_transit("done_gen", "finished", action="nop")
ag.behav.set_state("init")
ag.behav.save(f"{ag.name}.json")
ag.behav.save_pdf(f"{ag.name}.pdf")

# adding agent to environment
env.add_agent(ag)

# creating student agent
# ag = BasicAgent("student", model=BasicModel(attributes=env.shared_attributes, lr=0.001), authority=0.0)
ag = BasicAgent("student", model=BasicHLModel(attributes=env.shared_attributes, lr=0.001,
                                              delta=0.1, cnu_memories=20), authority=0.0)
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
Server(env=env, port=5001)

# running
env.run()
