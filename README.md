
<!-- PROJECT LOGO -->
<br />
<div align="center">
  <img src="imgs/its_cat_community.jpg" alt="Logo" width="500" height="500">
  <h2 align="center">NARNIAN - NAtuRe iNspired Intelligent AgeNts</h3>

  <p align="center" style="font-size: 1.1em; font-style: italic;">
    A new paradigm for simulating evolving ecosystems of intelligent agents
  </p>
</div>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) ![PyPI - Python Version](https://img.shields.io/pypi/pyversions/django)


## What is it?

Narnian is a nature-inspired framework that simulates an evolving ecosystem of intelligent agents. Designed for flexibility and extensibility, it enables researchers and developers to study dynamic learning and teaching processes in a customizable simulation world. Agents capture real-time data streams, adapt their behavior using Finite State Machines (FSMs), and can even evolve from students into teachers as they gain new capabilities.

## Dictionary of Key Concepts

- **Agents**  
  Narnian is roled by Agents, an agent is an entity that can interact with the environment and with the other agents living in it. Agents can be human or artificial, and they can have different levels of intelligence and different learning capabilities. Agents can be controlled by the user, or they can be controlled by the framework itself. An agent can be a student, a teacher, or any other entity that can interact with the environment. Each agent is represented by its Finite State Machine (FSM) which role its behavior.

- **Streams**  
  Real-time data flows representing sensory informations or descriptors from the environment.

- **Environment**  
  The simulation world where agents and streams coexist. The enviroment can be a physical world, a virtual world, or a combination of both. The environment can be static or dynamic, and it can have different levels of complexity. The environment can be controlled by the user, or it can be controlled by the framework itself. 

---

## Installation

Install a python enviroment ensuring than you have python 3.8 or higher.

Using [Anaconda](https://www.anaconda.com/products/individual) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html): 

```bash
conda create -n narnian python=3.8
conda activate narnian
```

Clone the repository:

```bash
git clone https://github.com/mela64/narnian
cd narnian
```

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

Narnian is structured as follows:

```
narnian/
├── narnian/
│   ├── __init__.py         # Package initializer
│   ├── **agents.py**           # Manages an agent's behavior, communication, and interactions
│   ├── attributes.py       # Manages attributes and labels for streams
│   ├── **environment.py**      # Defines the environment where agents and streams interact
│   ├── fsm.py              # Implements Finite State Machines for agent behaviors
│   ├── model.py            # Contains the core simulation models
│   ├── server.py           # Manages a server which makes the narnian world accessible through a web app
│   ├── **streams.py**          # Handles real-time data streams      
```

#### Basic Instance
A basic instance of the Narnian framework is available in the **./basic** folder. It contains an example of a simulated environment where agents named students and a teacher coexist. The teacher records information from the environment and then teaches the students using that information. After a learning run, the students are evaluated on their capabilities, and the teacher decides if a student is ready to become a teacher or if it needs more assistance with several runs.

```
narnian/
├── basic/
│   ├── utils/
│   │   ├── hl_utils.py
│   ├── **basic_agent.py**           # Basic's enviroment realisation of an agent
│   ├── **basic_environment.py**      # The basic enviroment 
│   ├── **basic_model_*.py**            # Models defined in the basic instance of narnian
│   ├── **basic_streams.py**          # Handles real-time data streams for basic instance   
```

### Examples

The root folder of the repository includes example simulations that showcase how to run the basic instance:

#### Animal School
Animal School is a simulation scenario built using the basic framework that mimics an educational ecosystem. In this scenario, a virtual "school" is created where a teacher agent instructs and evaluates student agents using real-time data streams composed of animal images.

At its core, the scenario emulates a classroom setting where:

- **A Teacher Agent ("Dr. Green")** collects general information from the environment, prepares an exam using the aggregate data stream, and then embarks on a teaching process. The teacher sets preferred image streams for focused instruction and evaluates student performance.

- **Student Agents ("Mario" and "Luigi")** adopt the teacher’s behavior patterns, engage with the provided data, and respond to teaching instructions. Their interactions simulate a typical learning process where they are guided, assessed, and eventually promoted based on their performance.

To execute the Animal School simulation, run the following command:

```bash
python sandbox_animal_school.py
```

It will start the server for monitoring
the simulation in a web browser. Follow the hyperlink provided in the terminal to access the web interface.

#### Cat Library
Cat Library is a simulation scenario where a digital "library" of textual data forms the basis for an educational ecosystem. In this scenario, agents interact with a stream of tokens—words or phrases related to cats. The simulation emulates a teaching and learning environment where the teacher agent compiles a "book" from the token stream, and the student agent learns from that prepared content.

To execute the Cat Library simulation, run the following command:

```bash
python sandbox_cat_library.py
```

#### Signal School
Signal School models a dynamic educational ecosystem where multiple signal streams are processed to evaluate learning, generalization, and predictive performance. The simulation involves a teacher agent orchestrating a teaching-evaluation loop over diverse signal datasets, while a student agent adapts and learns to generalize across these signals.

To execute the Signal School simulation, run the following command:

```bash
python sandbox_signal_school.py
```


### References

- [bla bla](https://arxiv.org/abs)

