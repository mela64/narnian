import torch
from narnian.server import Server
from narnian.model import EmptyModel
from basic.basic_agent import BasicAgent
from narnian.streams import Stream, BufferedStream
from basic.basic_environment import BasicEnvironment
from basic.basic_model_gen4all_hl import BasicHLModel
from basic.basic_streams import (SmoothHFHA, SmoothHFLA, SmoothLFHA, SmoothLFLA,
                                 SquareHFHA, SquareHFLA, SquareLFHA, SquareLFLA)


# creating environment
env = BasicEnvironment("env", title="Signal School")
device = torch.device("cpu")

# adding streams to the environment
env.add_stream(Stream.create(name="smoHfHa", creator=env.name, stream=BufferedStream().wrap(SmoothHFHA(), steps=1000)))
env.add_stream(Stream.create(name="smoHfLa", creator=env.name, stream=BufferedStream().wrap(SmoothHFLA(), steps=1000)))
env.add_stream(Stream.create(name="smoLfHa", creator=env.name, stream=BufferedStream().wrap(SmoothLFHA(), steps=1000)))
env.add_stream(Stream.create(name="smoLfLa", creator=env.name, stream=BufferedStream().wrap(SmoothLFLA(), steps=1000)))
env.add_stream(Stream.create(name="squHfHa", creator=env.name, stream=BufferedStream().wrap(SquareHFHA(), steps=1000)))
env.add_stream(Stream.create(name="squHfLa", creator=env.name, stream=BufferedStream().wrap(SquareHFLA(), steps=1000)))
env.add_stream(Stream.create(name="squLfHa", creator=env.name, stream=BufferedStream().wrap(SquareLFHA(), steps=1000)))
env.add_stream(Stream.create(name="squLfLa", creator=env.name, stream=BufferedStream().wrap(SquareLFLA(), steps=1000)))

# modeling behaviour of the environment
env.add_transit("init", "basic/behaviours/env_sharing_info.json", action="nop")

# creating the teacher agent "Dr. Green"
ag = BasicAgent("Dr. Green", model=EmptyModel(), authority=1.0)

# getting generic info from the environment
ag.add_transit("init", "basic/behaviours/getting_from_env.json", action="nop")

# engaging students, teaching and, afterward, evaluating students
ag.add_transit("got_contacts", "basic/behaviours/teach-playlist_eval-playlist-lastrep_looped_gen.json",
               action="set_pref_streams", args={"stream_hashes": [env.name + ":smoHfHa", env.name + ":smoHfLa",
                                                                  env.name + ":smoLfHa", env.name + ":smoLfLa",
                                                                  env.name + ":squHfHa", env.name + ":squHfLa",
                                                                  env.name + ":squLfHa"],
                                                "repeat": 3+1},
               wildcards={"<agent_name>": ag.name, "<learn_steps>": 1000, "<eval_steps>": 1000, "<cmp_thres>": 0.2})

# testing generalization
ag.add_transit("finished_work", "hard_exam_in_progress", action="ask_gen",
               args={"du_hash": env.name + ":squLfLa",  "dhat_hash": env.name + ":squLfLa", "ask_steps": 1000})
ag.add_transit("hard_exam_in_progress", "eval_time_again", action="done_gen")
ag.add_state_action("eval_time_again", action="eval",
                    args={"stream_hash": env.name + ":squLfLa", "what": "y", "how": "mse", "steps": 1000})
ag.add_transit("eval_time_again", "very_good", action="compare_eval", args={"cmp": "<=", "thres": 0.2})
ag.add_transit("eval_time_again", "not_good", action="nop")

# adding agent to environment
env.add_agent(ag)

# creating student agent named Mario
ag = BasicAgent("Mario", model=BasicHLModel(attributes=env.shared_attributes, device=device,
                                            delta=0.1, cnu_memories=20, seed=42), authority=0.0)

# getting generic info from the environment
ag.add_transit("init", "basic/behaviours/getting_from_env.json", action="nop")

# generic behaviour of a student who listens to the requests from the teacher
ag.add_transit("got_contacts", "./basic/behaviours/listening_to_teacher.json", action="get_engagement",
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
env.run(checkpoints="sandbox_signal_school_checkpoints.json")
