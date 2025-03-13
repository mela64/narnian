import torch
from narnian.server import Server
from narnian.model import EmptyModel
from basic.basic_agent import BasicAgent
from narnian.streams import Stream, ImageDataset
from basic.basic_image_model import BasicImageModel, BasicImageModelCNU
from basic.basic_environment import BasicEnvironment

# creating environment
env = BasicEnvironment("Env", title="Animal School")
device = torch.device("cpu")

# adding streams to the environment
env.add_stream(Stream.create(name="albatross", creator=env.name,
                             stream=ImageDataset(image_dir="data/animals",
                                                 label_file_csv="data/animals/c1_skip_10i.csv")))
env.add_stream(Stream.create(name="cheetah", creator=env.name,
                             stream=ImageDataset(image_dir="data/animals",
                                                 label_file_csv="data/animals/c2_skip_10i.csv")))
env.add_stream(Stream.create(name="giraffe", creator=env.name,
                             stream=ImageDataset(image_dir="data/animals",
                                                 label_file_csv="data/animals/c3_skip_10i.csv")))
env.add_stream(Stream.create(name="all", creator=env.name,
                             stream=ImageDataset(image_dir="data/animals",
                                                 label_file_csv="data/animals/first3c_10i.csv")))

# modeling behaviour of the environment
env.add_transit("init", "basic/behaviours/env_sharing_info.json", action="nop")

# creating the teacher agent "Dr. Green"
ag = BasicAgent("Dr. Green", model=EmptyModel(), authority=1.0)

# getting generic info from the environment
ag.add_transit("init", "basic/behaviours/getting_from_env.json", action="nop")

# preparing exam
ag.add_transit("got_agents", "exam_prepared", action="record",
               args={"stream_hash": env.name + ":all", "steps": 30})

# engaging students, teaching and, afterward, evaluating students
ag.add_transit("exam_prepared", "basic/behaviours/teach-playlist_eval-recorded1_pred.json",
               action="set_pref_streams", args={"stream_hashes": [env.name + ":albatross",
                                                                  env.name + ":cheetah",
                                                                  env.name + ":giraffe"]},
               wildcards={"<agent_name>": ag.name, "<learn_steps>": 40, "<eval_steps>": 30, "<cmp_thres>": 0.4})

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
ag = BasicAgent("Mario", model=BasicImageModelCNU(attributes=env.shared_attributes, mem_units=5,
                                                  lr=0.01956, lr_head=0.08583, device=device, seed=42), authority=0.0)

# in principle, he is like Dr. Green...
ag.behave_as(env.agents["Dr. Green"])

# ...however, he is not ready yet to prepare exams and teach
ag.wait_for_actions(ag, "got_agents", "exam_prepared", wait=True)

# generic behaviour of a student who listens to the requests from the teacher
ag.add_transit("got_agents", "./basic/behaviours/listening_to_teacher.json", action="get_engagement",
               args={"min_auth": 1.0, "max_auth": 1.0})

# when the teacher will send the student back home
ag.add_transit("teacher_engaged", "got_agents", action="get_disengagement")

# adding agent to environment
env.add_agent(ag)

# creating another student agent named Luigi
ag = BasicAgent("Luigi", model=BasicImageModel(attributes=env.shared_attributes,
                                               lr=0.007223, device=device, seed=42), authority=0.0)

# he really acts like Mario
ag.behave_as(env.agents["Mario"])

# adding agent to environment
env.add_agent(ag)

# printing
print(env)
for ag in env.agents.values():
    print(ag)

# creating server
Server(env=env)

# running
env.run(steps=450)
