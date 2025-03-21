<br />
<div align="center">
  <img src="imgs/narniancats.jpg" alt="Logo" width="500" height="500">
  <h2 align="center">NARNIAN - NAtuRe iNspired Intelligent AgeNts</h2>

  <p align="center" style="font-size: 1.1em; font-style: italic;">
    A new paradigm for simulating evolving ecosystems of intelligent agents that learn over time
  </p>
</div>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) ![PyPI - Python Version](https://img.shields.io/pypi/pyversions/django)


## What is it?

The NARNIAN project promotes a nature-inspired framework that simulates an evolving ecosystem of agents that learn over time by interacting with each other and with humans. Designed for flexibility and extensibility, it enables researchers and developers to study dynamic learning and teaching processes in a customizable simulation world, that might be directly linked to "reality". Agents capture, generate, and share real-time data streams and adapt their capabilities (their *Model*) in function of the information to which they are exposed. Their interactions patterns are handled by an architectural component called *Behavior*.

## Dictionary of Key Concepts

- **Agents**  
  NARNIAN is a dynamic world populated by Agents, where an Agent is an entity that can interact with the environment and with the other agents living in it. Agents can be human or artificial, and they can have different levels of intelligence and different learning capabilities. Agents can be controlled by the user, or they can be controlled by the framework itself. In the current implementation, each agent is represented by its Finite State Machine (FSM) which describe its *Behavior*, and by a neural *Model* that determines how it reacts to perceptual stimuli. The *Model* is capable of *generation* and of prediction of *descriptors* of the generated or given data.

- **Streams**  
  Real-time data flows representing *sensory information* and *descriptors* from the environment.

- **Environment**  
  The simulation world where agents and streams coexist. The enviroment can be a physical world, a virtual world, or a combination of both. The environment can be static or dynamic, and it can have different levels of complexity. The environment can be controlled by the user, or it can be controlled by the framework itself. 

---

## Installation

Install a python enviroment ensuring than you have python 3.10 or higher.

Using [Anaconda](https://www.anaconda.com/products/individual) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html): 


```bash
conda create -n narnian python=3.10
conda activate narnian
```

Clone the repository:

```bash
git clone https://github.com/mela64/narnian
cd narnian
```

> [!IMPORTANT]
> Install graphviz for visualizing the FSMs (not strictly required for running narnian simulations):
> ```bash
> sudo apt install graphviz # For Sudoers
> conda install -c conda-forge graphviz # Otherwise
> ```

Install the dependecies with pip:

```bash
pip install -r requirements.txt
```

Try one of the examples:

```bash
python sandbox_example.py # A simple example of a Narnian simulation.
```

---

### Project Structure

Narnian is structured as follows, where we emphasized with square brakets those files that will be extended when creating an instance of NARNIAN environment (see below):

```
narnian/
├── narnian/
│   ├── __init__.py            # Package initializer
│   ├── [agents.py]            # A NARNIAN agent, with its basic skills in communication and interaction
│   ├── attributes.py          # Manages attributes and labels for streams
│   ├── [environment.py]       # Defines the environment where agents and streams interact
│   ├── fsm.py                 # Implements Finite State Machines for agent behaviors
│   ├── [model.py]             # Contains the structure of the core models of the agents (generators, predictors)
│   ├── server.py              # Manages a server which makes the NARNIAN world accessible through a web app
│   ├── [streams.py]           # Handles real-time data streams      
```

#### Basic Instance
A basic instance of the Narnian framework is available in the **basic** folder. It contains an example of a simulated environment where student-like agents named Mario and Luigi and a teacher-like agent named Dr. Greeen coexist. The teacher exploits information from the environment to teach the students. After one or more lectures, the students are evaluated on their capabilities, and, in some cases, the teacher may decide if a student is ready to become a new teacher or if it needs more assistance.

```
narnian/
├── basic/
│   ├── utils/
│   │   ├── hl_utils.py
│   ├── [basic_agent.py]       # An agent living in a basic environment
│   ├── [basic_environment.py] # The basic enviroment 
│   ├── [basic_model_*.py]     # Neural models that can be used in the basic envirnment
│   ├── [basic_streams.py]     # Streams that can be used in the basic environment 
```

### Examples

The root folder of the repository includes example simulations that showcase how to run the basic:

#### Animal School
Animal School is a simulation scenario (what we call "sandbox") built using the basic framework that mimics an educational ecosystem. In this scenario, a virtual "school" is created where a teacher agent instructs and evaluates student agents using real-time data streams composed of animal images.

At its core, the scenario emulates a classroom setting where:

- **A Teacher Agent ("Dr. Green")** collects general information from the environment, prepares an exam using the aggregate data stream, and then embarks on a teaching process. The teacher sets preferred image streams for focused instruction and evaluates student performance.

- **Student Agents ("Mario" and "Luigi")** adopt the teacher’s behavior patterns, engage with the provided data, and respond to teaching instructions. Their interactions simulate a typical learning process where they are guided, assessed, and eventually promoted based on their performance.

To execute the Animal School sandbox, run the following command:

```bash
python sandbox_animal_school.py
```

It will start the server for monitoring
the simulation in a web browser. Follow the hyperlink provided in the terminal to access the web interface.

#### Cat Library
Cat Library is a simulation scenario ("sandbox") where a digital "library" of textual data forms the basis for an educational ecosystem. In this scenario, agents interact with a stream of tokens—words related to cats. The simulation emulates a teaching and learning environment where the teacher agent compiles a "book" from the token stream, and the student agent learns from that prepared content.

To execute the Cat Library sandbox, run the following command:

```bash
python sandbox_cat_library.py
```

#### Signal School
Signal School sandbox is a dynamic ecosystem where multiple streams of scalar signal are processed to evaluate learning, generalization, and predictive performance. The simulation involves a teacher agent orchestrating a teaching-evaluation loop over diverse scalar signals, while a student agent adapts and learns to generalize across these signals.

To execute the Signal School sandbox, run the following command:

```bash
python sandbox_signal_school.py
```


### References
<div align="center">
  <img src="imgs/caicat.png" alt="Logo" width="60" height="60" style="vertical-align: middle;">
  <h3 style="display: inline; margin-left: 10px;"><a href='https://cai.diism.unisi.it/'>Collectionless AI</a></h3>
</div>

Collectionless AI Team - &copy; Stefano Melacci (2025)