import torch
from narnian.server import Server
from narnian.streams import Stream
from narnian.model import EmptyModel
from basic.basic_agent import BasicAgent
from basic.basic_streams import Sin, Square
from basic.basic_environment import BasicEnvironment
from basic.basic_model_gen4all_pred4all_gd import BasicModel

# creating environment
env = BasicEnvironment("env", title="Sample Sandbox")  # named "env", but it also has an extended title
device = torch.device("cpu")

# adding streams to the environment: a sinusoidal wave and a square wave, named "sin" and "square", created by "env"
env.add_stream(Stream.create(name="sin", creator=env.name, stream=Sin(freq=0.06, phase=0.5, delta=0.1)))
env.add_stream(Stream.create(name="square", creator=env.name, stream=Square(freq=0.06, ampl=0.5, phase=0.5, delta=0.1)))

# modeling behaviour of the environment
# it is a 3-state FSM, init --a--> streams_enabled --b--> streams_sent --c--> ready,
# where actions a, b, c, are enable_all_streams, send_streams_to_all, send_agents_to_all, respectively
env.add_transit("init", "streams_enabled", action="enable_all_streams")
env.add_transit("streams_enabled", "streams_shared", action="send_streams_to_all")
env.add_transit("streams_shared", "contacts_shared", action="send_agents_to_all")

# creating teacher-like agent named Dr. Green
ag = BasicAgent("Dr. Green", model=EmptyModel(), authority=1.0)  # empty model, it is just a manager

# modeling behaviour of the teacher agent Dr. Green
# the first part is just a set of reactions to what the environment does
ag.add_transit("init", "got_streams", action="get_streams")
ag.add_transit("got_streams", "got_contacts", action="get_agents")

# then, the teacher records 500 steps for the 2 streams of the environment, and
# creates a playlist composed of the two recordings
ag.add_transit("got_contacts", "recording1", action="record",
               args={"stream_hash": env.name + ":sin", "steps": 500})
ag.add_transit("recording1", "recording2", action="record",
               args={"stream_hash": env.name + ":square", "steps": 500})
ag.add_transit("recording2", "looking_for_students", action="set_pref_streams",
               args={"stream_hashes": [ag.name + ":recorded1", ag.name + ":recorded2"], "repeat": 1})

# the teacher looks for students (a student features a 0-level of authority in this example), shares the playlist,
# and asks the student to learn a signal from it (both generation and descriptor-prediction)
ag.add_state_action("looking_for_students", action="find_agent_to_engage", args={"min_auth": 0., "max_auth": 0.})
ag.add_transit("looking_for_students", "student_found", action="send_engagement")
ag.add_transit("student_found", "looking_for_students", action="nop")
ag.add_transit("student_found", "student_engaged", action="got_engagement")
ag.add_transit("student_engaged", "teaching_time", action="share_streams")
ag.add_transit("teaching_time", "lecture_in_progress", action="ask_learn_gen_and_pred",
               args={"du_hash": "<playlist>", "yhat_hash": "<playlist>", "dhat_hash": "<playlist>", "ask_steps": 500})
ag.add_transit("lecture_in_progress", "student_finished_following", action="done_learn_gen_and_pred")
ag.add_transit("student_finished_following", "exam_in_progress", action="ask_gen_and_pred",
               args={"du_hash": "<playlist>", "ask_steps": 500})
ag.add_transit("exam_in_progress", "student_finished_exam", action="done_gen_and_pred")

# the teacher evaluates the student, asking to generate the considered signal of the playlist
# (in this example, we do not check for the quality of the descriptor-prediction process, that was indeed subject to
# learning as well)
# if the evaluation is not satisfactory (MSE > 0.05), the teacher asks again to learn, otherwise it moves to the next
# element of the playlist and asks the student to learn the new signal
ag.add_state_action("student_finished_exam", action="eval",
                    args={"stream_hash": "<playlist>", "what": "y", "how": "mse", "steps": 500})
ag.add_transit("student_finished_exam", "good", action="compare_eval", args={"cmp": "<=", "thres": 0.05})
ag.add_transit("student_finished_exam", "teaching_time", action="nop", args={"message": "Insufficient"})
ag.add_state_action("good", action="next_pref_stream")
ag.add_transit("good", "teaching_time", action="check_pref_stream", args={"what": "not_first"})

# when the playlist ends, we are done
ag.add_transit("good", "finished", action="check_pref_stream", args={"what": "first"})

# adding Dr. Green, the teacher-agent, to environment
env.add_agent(ag)

# creating a student agent named Mario
ag = BasicAgent("Mario", model=BasicModel(attributes=env.shared_attributes, lr=0.001, device=device, seed=4),
                authority=0.0)  # Mario has a basic neural networks, based on a pair of state-space models

# modeling behaviour of the teacher agent Dr. Green
# the first part is just a set of reactions to what the environment does
ag.add_transit("init", "got_streams", action="get_streams")
ag.add_transit("got_streams", "got_contacts", action="get_agents")

# then, student Mario responds to a request from a teacher, and it simply follows its indications about learning and
# making predictions
ag.add_transit("got_contacts", "teacher_engaged", action="get_engagement", args={"min_auth": 1., "max_auth": 1.})
ag.add_transit("teacher_engaged", "got_more_streams", action="get_streams")
ag.add_transit("got_more_streams", "teacher_engaged", action="nop")
ag.add_transit("teacher_engaged", "done_learning", action="do_learn_gen_and_pred")
ag.add_transit("done_learning", "teacher_engaged", action="nop")
ag.add_transit("teacher_engaged", "done", action="do_gen_and_pred")
ag.add_transit("done", "teacher_engaged", action="nop")

# adding agent Mario to environment
env.add_agent(ag)

# printing
print(env)
for ag in env.agents.values():
    print(ag)

# saving the environment (just an example) in a folder named "output" (this is actually the default name)
env.save(where="output")

# creating server: this will allow us to check what is going on, and it will immediately pause the environment
Server(env=env)

# running the environment
env.run()
