import torch
from narnian.server import Server
from narnian.model import EmptyModel
from basic.basic_agent import BasicAgent
from narnian.streams import Stream, Tokens
from basic.basic_token_model import BasicTokenModel
from basic.basic_environment import BasicEnvironment

# creating environment
env = BasicEnvironment("env", title="Cat Library")
device = torch.device("cpu")

# adding streams to the environment
env.add_stream(Stream.create(name="cats", creator=env.name,
                             stream=Tokens(tokens_file_csv="data/cats/stream_of_words.csv")))

# modeling behaviour of the environment
env.add_transit("init", "./basic/behaviours/env_sharing_info.json", action="nop")

# creating the teacher agent "Dr. Green"
ag = BasicAgent("Dr. Green", model=EmptyModel(), authority=1.0)

# getting generic info from the environment
ag.add_transit("init", "basic/behaviours/getting_from_env.json", action="nop")

# preparing exam
ag.add_transit("got_agents", "exam_prepared", action="record",
               args={"stream_hash": env.name + ":cats", "steps": 1000})

# engaging students, teaching and, afterward, evaluating students
ag.add_transit("exam_prepared", "basic/behaviours/teach-playlist_eval-recorded1_gen.json",
               action="set_pref_streams", args={"stream_hashes": [env.name + ":cats"]},
               wildcards={"<agent_name>": ag.name, "<learn_steps>": 1000, "<eval_steps>": 1000, "<cmp_thres>": 0.4})

# promoting students that were positively evaluated
ag.add_transit("some_good", "promote", action="set_authority", args={"agent": "<valid_cmp>", "auth": 1.0})

# freeing students
ag.add_transit("promote", "habilitate", action="send_disengagement")

# telling promoted students that is time to teach
ag.add_transit("habilitate", "done_teaching", action="wait_for_actions",
               args={"agent": "<valid_cmp>", "from_state": "got_agents", "to_state": "exam_prepared", "wait": False})

# adding agent to environment
env.add_agent(ag)

# creating student agent named Mario
ag = BasicAgent("Mario", model=BasicTokenModel(attributes=env.shared_attributes, lr=0.01, device=device),
                authority=0.0)

# in principle, he is like Dr. Green...
ag.behave_as(env.agents["Dr. Green"])

# ...however, he is not ready yet to prepare exams and teach
ag.wait_for_actions(ag, "got_agents", "exam_prepared", wait=True)

# generic behaviour of a student who listens to the requests from the teacher
ag.add_transit("got_agents", "./basic/behaviours/listening_to_teacher.json", action="get_engagement",
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

