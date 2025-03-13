import torch
from narnian.server import Server
from narnian.model import EmptyModel
from basic.basic_agent import BasicAgent
from narnian.streams import Stream, Tokens, BufferedStream
from basic.basic_token_model import BasicTokenModel
from basic.basic_environment import BasicEnvironment

# creating environment
env = BasicEnvironment("env", title="Cat Library")
device = torch.device("cpu")

# adding streams to the environment
env.add_stream(Stream.create(name="cats", creator=env.name,
                             stream=BufferedStream().wrap(Tokens(tokens_file_csv="data/cats/stream_of_words.csv"),
                                                          steps=998)))

# modeling behaviour of the environment
env.add_transit("init", "basic/behaviours/env_sharing_info.json", action="nop")

# creating the teacher agent "Dr. Green"
ag = BasicAgent("Dr. Green", model=EmptyModel(), authority=1.0)

# getting generic info from the environment
ag.add_transit("init", "basic/behaviours/getting_from_env.json", action="nop")

# preparing exam
ag.add_transit("got_agents", "exam_prepared", action="record",
               args={"stream_hash": env.name + ":cats", "steps": 998})

# engaging students, teaching and, afterward, evaluating students
ag.add_transit("exam_prepared", "basic/behaviours/teach-playlist_eval-recorded1_gen.json",
               action="set_pref_streams", args={"stream_hashes": [env.name + ":cats"], "repeat": 100},
               wildcards={"<agent_name>": ag.name, "<learn_steps>": 998, "<eval_steps>": 998, "<cmp_thres>": 0.4})

# adding agent to environment
env.add_agent(ag)

# creating student agent named Mario
ag = BasicAgent("Mario", model=BasicTokenModel(attributes=env.shared_attributes, lr=1.0, device=device, seed=42),
                authority=0.0)

# in principle, he is like Dr. Green...
ag.behave_as(env.agents["Dr. Green"])

# ...however, he is not ready yet to prepare exams and teach
ag.wait_for_actions(ag, "got_agents", "exam_prepared", wait=True)

# generic behaviour of a student who listens to the requests from the teacher
ag.add_transit("got_agents", "basic/behaviours/listening_to_teacher.json", action="get_engagement",
               args={"min_auth": 1.0, "max_auth": 1.0})

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
