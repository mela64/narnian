from narnian.server import Server
from narnian.streams import Stream, ImageDataset
from basic.basic_agent import BasicAgent
from basic.basic_image_model import BasicImageModel
from basic.basic_environment import BasicEnvironment
from torchvision import transforms


# creating environment
env = BasicEnvironment("sandbox")


# adding streams to the environment
env.add_stream(Stream.create(name="animals", creator="envir",
                             stream=ImageDataset(image_dir="./data/animals", label_file_csv="./data/animals/labels.csv",
                                                 circular=True)))

# modeling behaviour of the environment
env.behav.add_transit("init", "streams_enabled", action="enable_all_streams")
env.behav.add_transit("streams_enabled", "streams_sent", action="send_streams_to_all")
env.behav.add_transit("streams_sent", "ready", action="send_agents_to_all")
env.behav.set_state("init")
env.behav.save(f"{env.name}.json")
env.behav.save_pdf(f"{env.name}.pdf")

# creating teacher agent
ag = BasicAgent("teacher", model=BasicImageModel(attributes=env.shared_attributes, lr=0.), authority=1.0)
ag.behav.add_transit("init", "got_streams", action="get_streams")
ag.behav.add_transit("got_streams", "got_agents", action="get_agents")
ag.behav.add_transit("got_agents", "playlist_ready", action="set_pref_streams",
                     args={"stream_hashes": ["envir:animals", "envir:animals"]})
ag.behav.add_state_action("playlist_ready", action="find_agent_to_engage", args={"min_auth": 0.0, "max_auth": 0.0})
ag.behav.add_transit("playlist_ready", "student_found", action="send_engagement")
ag.behav.add_transit("student_found", "playlist_ready", action="nop")
ag.behav.add_transit("student_found", "student_engaged", action="got_engagement")
ag.behav.add_transit("student_engaged", "asked_learn", action="ask_learn_pred",
                     args={"yhat_hash": "<playlist>", "dhat_hash": "<playlist>", "ask_steps": 200})
ag.behav.add_transit("asked_learn", "done_learn", action="done_learn_pred")
ag.behav.add_transit("done_learn", "asked_pred", action="ask_pred", args={"yhat_hash": "<playlist>", "ask_steps": 50})
ag.behav.add_transit("asked_pred", "done_pred", action="done_pred")
ag.behav.add_state_action("done_pred", action="eval", args={"stream_hash": "<playlist>", "what": "d", "steps": 50})
ag.behav.add_transit("done_pred", "student_engaged", action="compare_eval", args={"cmp": "<", "thres": 0.5})
ag.behav.add_transit("done_pred", "good", action="compare_eval", args={"cmp": ">=", "thres": 0.5})
ag.behav.add_state_action("good", action="next_pref_stream")
ag.behav.add_transit("good", "student_engaged", action="check_pref_stream", args={"what": "not_first"})
ag.behav.add_transit("good", "finished", action="check_pref_stream", args={"what": "first"})
ag.behav.set_state("init")
ag.behav.save(f"{ag.name}.json")
ag.behav.save_pdf(f"{ag.name}.pdf")

# adding agent to environment
env.add_agent(ag)

# creating student agent
ag = BasicAgent("student", model=BasicImageModel(attributes=env.shared_attributes, lr=0.0001), authority=0.0)
ag.behav.add_transit("init", "got_streams", action="get_streams")
ag.behav.add_transit("got_streams", "got_agents", action="get_agents")
ag.behav.add_transit("got_agents", "teacher_engaged", action="get_engagement", args={"min_auth": 1.0, "max_auth": 1.0})
ag.behav.add_transit("teacher_engaged", "learning", action="do_learn_pred")
ag.behav.add_transit("teacher_engaged", "predicted", action="do_pred")
ag.behav.add_transit("learning", "teacher_engaged", action="nop")
ag.behav.add_transit("predicted", "teacher_engaged", action="nop")
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
