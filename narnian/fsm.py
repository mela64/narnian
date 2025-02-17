import json
from collections.abc import Iterable
import graphviz
import torch
import inspect


class FiniteStateMachine:
    DEBUG = True

    def __init__(self, actionable: object, policy: str = "sampling"):
        self.initial_state = None
        self.prev_state = None
        self.state = None
        self.states = {}
        self.transitions = {}
        self.actionable = actionable
        self.param_buffer = {}  # dictionary of pre-buffered parameter names and their current value
        self.action_to_params = {}  # list of parameter names for each action
        self.policy = policy
        self.last_action_tuple = None  # last executed action and its arguments (and score)
        self.running_action_tuple = None  # action that is being executed and its arguments (and score)

        assert self.policy in ["uniform", "sampling"], \
            f"Invalid policy: {self.policy}, it must be in ['uniform', 'sampling']"

    def __str__(self):

        fsm_data = {
            'initial_state': self.initial_state,
            'state': self.state,
            'prev_state': self.prev_state,
            'policy': self.policy,
            'state_actions': {
                state: str([state_action_tuple[0].__name__, state_action_tuple[1]])
                for state, state_action_tuple in self.states.items() if state_action_tuple is not None
            },
            'transitions': {
                from_state: {
                    to_state: str([[action.__name__, args, score]
                                   for (action, args, score) in action_tuple])
                    for to_state, action_tuple in to_states.items()
                }
                for from_state, to_states in self.transitions.items()
            },
            'last_action': str([self.last_action_tuple[0].__name__,
                                self.last_action_tuple[1],
                                self.last_action_tuple[2]]) if self.last_action_tuple is not None else None,
            'running_action': str([self.running_action_tuple[0].__name__,
                                   self.running_action_tuple[1],
                                   self.running_action_tuple[2]]) if self.running_action_tuple is not None else None,
            'param_buffer': self.param_buffer
        }
        json_string = json.dumps(fsm_data, indent=4)
        json_string = json_string.replace("\"[", "[")
        json_string = json_string.replace("]\"", "]")
        json_string = json_string.replace("'", "\"")
        json_string = json_string.replace("None", "null")
        return json_string

    def set_actionable(self, obj: object):
        """Set the object where actions should be found (as methods)."""

        self.actionable = obj

    def add_state_action(self, state: str, action: str = None, args: dict | None = None):
        """Add an action (inner action) to an existing state (or creating it from scratch, if not existent)."""

        if args is None:
            args = {}

        if action is not None:
            action_callable = self.action_name_to_callable(action)

            # saving user-specified parameters
            sig = inspect.signature(action_callable)
            for param_name, param_value in args.items():
                if param_name not in sig.parameters.keys():
                    raise ValueError(f"Unknown parameter for action {action_callable.__name__}: " + str(param_name))
            params = [param_name for param_name in sig.parameters.keys()]
            defaults = {param.name: param.default for param in sig.parameters.values() if
                        param.default is not inspect.Parameter.empty}
            self.action_to_params[action_callable.__name__] = (params, defaults)

            # saving operation
            self.states[state] = (action_callable, args)
        else:
            self.states[state] = None

    def get_state(self) -> str:
        """Returns the current state of the FSM."""

        return self.state

    def reset_state(self):
        """Go back to the initial state of the FSM."""

        self.state = self.initial_state
        self.prev_state = None

    def get_states(self) -> Iterable[str]:
        """Get all the states of the FSM."""

        return self.transitions.keys()

    def set_state(self, state: str):
        """Set the current state."""

        if state in self.transitions:
            self.prev_state = self.state
            self.state = state
            if self.initial_state is None:
                self.initial_state = state
        else:
            raise ValueError("Unknown state: " + str(state))

    def add_transit(self, from_state: str, to_state: str,
                    action: str, args: dict | None = None, score: float = 1.0):
        """Define a transition between two states with an associated action."""

        assert 0 <= score <= 1.0, f"Invalid transition score: {score}"

        if from_state not in self.transitions:
            self.add_state_action(from_state, action=None)
            self.transitions[from_state] = {}
        if to_state not in self.transitions:
            self.add_state_action(to_state, action=None)
            self.transitions[to_state] = {}
        if args is None:
            args = {}

        action_callable = self.action_name_to_callable(action)

        # saving user-specified parameters
        sig = inspect.signature(action_callable)
        for param_name, param_value in args.items():
            if param_name not in sig.parameters.keys():
                raise ValueError(f"Unknown parameter for action {action_callable.__name__}: " + str(param_name))
        params = [param_name for param_name in sig.parameters.keys()]
        defaults = {param.name: param.default for param in sig.parameters.values() if
                    param.default is not inspect.Parameter.empty}
        self.action_to_params[action_callable.__name__] = (params, defaults)

        if to_state not in self.transitions[from_state]:
            self.transitions[from_state][to_state] = [(action_callable, args, score)]
        else:
            if (action_callable, args, score) in self.transitions[from_state][to_state]:
                raise ValueError(f"Repeated transition from {from_state} to {to_state}: "
                                 f"{(action, args, score)}")
            self.transitions[from_state][to_state].append((action_callable, args, score))

    def set_buffer_param_value(self, param_name: str, param_value):
        """Add a new variable to the FSM."""

        self.param_buffer[param_name] = param_value

    def get_buffer_param_value(self, param_name: str):
        """Get a variable from the FSM."""

        return self.param_buffer[param_name]

    def action_name_to_callable(self, action_name: str):
        """Get a function from its name."""

        if self.actionable is not None:
            action_fcn = getattr(self.actionable, action_name)
        else:
            action_fcn = globals()[action_name] if action_name in globals() else None
        if action_fcn is None:
            raise ValueError("Cannot find function/method: " + str(action_name))
        return action_fcn

    def act_states(self):
        """Apply actions that do not trigger a state transition."""

        state_action_tuple = self.states[self.state]
        if state_action_tuple is not None:
            action, args = state_action_tuple

            # picking up the inner-state-action parameters, considering their current buffered values and the given ones
            actual_params = self.__get_actual_params(action.__name__, args)

            # call inner state action
            if actual_params is not None:
                action(**actual_params)

    def act_transitions(self):

        # collecting list of feasible actions, scores, etc. (from the current state)
        actions_list = []
        args_list = []
        scores_list = []
        to_state_list = []

        for to_state, action_tuples in self.transitions[self.state].items():
            for i, (action, args, score) in enumerate(action_tuples):
                if (not isinstance(score, tuple | list) and score > 0.) or \
                        (isinstance(score, tuple | list) and score[0] > 0.):
                    actions_list.append(action)
                    args_list.append(args)
                    to_state_list.append(to_state)
                    if not isinstance(score, tuple | list):  # if "score" is a tuple, it means that includes a tmp value
                        scores_list.append(score)
                    else:
                        scores_list.append(score[0])
                if isinstance(score, tuple | list):
                    action_tuples[i] = (action, args, score[1])  # restore original

        # using the selected policy to decide what action to apply
        while len(actions_list) > 0:

            if self.policy == "uniform":
                a = torch.randint(0, len(actions_list), [1]).item()

            elif self.policy == "sampling":
                s = torch.tensor(scores_list, dtype=torch.float)
                a = torch.multinomial(s, 1, replacement=True).item()
            else:
                raise ValueError(f"Unknown policy: {self.policy}")

            # picking up the actual parameters, considering their current buffered values and the given ones
            actual_params = self.__get_actual_params(actions_list[a].__name__, args_list[a])

            # debug only
            _debug_printed = False

            # call action
            if actual_params is not None:
                ret = actions_list[a](**actual_params)
            else:
                if FiniteStateMachine.DEBUG:
                    print("   [DEBUG FSM] Tried and failed (missing actual param): " + actions_list[a].__name__)
                    _debug_printed = True
                ret = False

            # if the action failed, another one will be sampled
            if not ret:
                if FiniteStateMachine.DEBUG:
                    if not _debug_printed:
                        print("   [DEBUG FSM] Tried and failed (failed execution): " + actions_list[a].__name__)
                del actions_list[a]
                del args_list[a]
                del scores_list[a]
                del to_state_list[a]
            else:

                # transition
                self.prev_state = self.state
                self.state = to_state_list[a]

                # clearing action-related arguments
                for param in actual_params.keys():
                    if param in self.param_buffer.keys():
                        del self.param_buffer[param]
                return

    def clear_param_buffer(self):
        self.param_buffer = {}

    def set_action_score(self, action_name: str, args: dict | None = None,
                         new_score: float | str = "top", from_state: str | None = None, to_state: str | None = None):
        """Change the score of an action, if the action exists."""

        if from_state is None:
            from_state = self.state

        if from_state not in self.transitions:
            return False

        if to_state is not None and to_state not in self.transitions[from_state]:
            return False

        assert isinstance(new_score, float) or isinstance(new_score, str), f"Invalid score: {new_score}"
        assert isinstance(new_score, float) or new_score in ["top", "bottom", "eq_max", "eq_min", "eq_middle"], \
            f"Invalid score: {new_score}"

        to_states = self.transitions[from_state].keys() if to_state is None else [to_state]

        for to_state in to_states:
            for i, (_action, _args, _score) in enumerate(self.transitions[from_state][to_state]):
                if _action.__name__ == action_name and (args is None or _args == args):
                    if isinstance(new_score, float):
                        self.transitions[from_state][to_state][i] = (_action, _args, (new_score, _score))
                    elif isinstance(new_score, str):
                        _max = _score
                        _min = _score
                        _mean = 0.
                        _n = 0
                        for _to_state, _action_tuples in self.transitions[from_state].items():
                            for (_, _, __score) in _action_tuples:
                                _max = max(__score, _max)
                                _min = min(__score, _min)
                                _mean += __score
                                _n += 1
                        _mean /= _n

                        if new_score == 'eq_max':
                            self.transitions[from_state][to_state][i] = (_action, _args, (_max, _score))
                        elif new_score == 'eq_min':
                            self.transitions[from_state][to_state][i] = (_action, _args, (_min, _score))
                        elif new_score == 'eq_mean':
                            self.transitions[from_state][to_state][i] = (_action, _args, (_mean, _score))
                        elif new_score == 'bottom':
                            self.transitions[from_state][to_state][i] = (_action, _args, (0., _score))
                        elif new_score == 'top':
                            for _to_state, _action_tuples in self.transitions[from_state].items():
                                for j, (__action, __args, __score) \
                                        in enumerate(_action_tuples):
                                    if (__action.__name__ == action_name and
                                            (args is None or __args == args)):
                                        self.transitions[from_state][_to_state][j] = \
                                            (__action, __args, (1., __score))
                                    else:
                                        self.transitions[from_state][_to_state][j] = \
                                            (__action, __args, (0., __score))
                    else:
                        raise ValueError(f"Invalid score: {new_score}")
                    return True
        return False

    def save(self, filename: str):
        """Save the FSM to a JSON file."""

        with (open(filename, 'w') as file):
            file.write(str(self))

    def load(self, filename: str):
        """Load the FSM state from a JSON file and resolve actions."""

        with open(filename, 'r') as file:
            fsm_data = json.load(file)

        self.initial_state = fsm_data['initial_state']
        self.state = fsm_data['state']
        self.prev_state = fsm_data['prev_state']
        self.policy = fsm_data['policy']
        self.param_buffer = fsm_data['param_buffer']
        self.last_action_tuple = (self.action_name_to_callable(fsm_data['last_action'][0]),
                                  fsm_data['last_action'][1],
                                  fsm_data['last_action'][2])
        self.running_action_tuple = (self.action_name_to_callable(fsm_data['running_action'][0]),
                                     fsm_data['running_action'][1],
                                     fsm_data['running_action'][2])

        self.states = {}
        for state, (action_name, args) in fsm_data['states'].items():
            self.add_state_action(state, action=action_name, args=args)

        self.transitions = {}
        for from_state, to_states in fsm_data['transitions'].items():
            for to_state, action_tuples in to_states.items():
                for action_name, args, score in action_tuples:
                    self.add_transit(from_state, to_state, action_name, args, score)

    def save_pdf(self, filename: str):
        """Save the FSM in GraphViz format, drawn on a PDF file."""

        graph = graphviz.Digraph()

        graph.attr('node', fontsize='8')
        for state, state_action_tuple in self.states.items():
            if state_action_tuple is not None:
                action, args = state_action_tuple
                s = "("
                for i, (k, v) in enumerate(args.items()):
                    s += str(k) + "=" + (str(v) if not isinstance(v, str) else ("'" + v + "'"))
                    if i < len(args) - 1:
                        s += ", "
                s += ")"
                label = action.__name__ + s
                if len(label) > 40:
                    tokens = label.split(" ")
                    z = ""
                    i = 0
                    done = False
                    while i < len(tokens):
                        z += (" " if i > 0 else "") + tokens[i]
                        if not done and i < (len(tokens) - 1) and len(z + tokens[i + 1]) > 40:
                            z += "\n    "
                            done = True
                        i += 1
                    label = z
                suffix = "\n" + label
            else:
                suffix = ""
            if state == self.initial_state:
                graph.attr('node', shape='diamond')
            else:
                graph.attr('node', shape='oval')
            if state == self.state:
                graph.attr('node', style='filled')
            else:
                graph.attr('node', style='solid')
            graph.node(state, state + suffix)

        for from_state, to_states in self.transitions.items():
            for to_state, action_tuples in to_states.items():
                for action, args, score in action_tuples:
                    s = "("
                    for i, (k, v) in enumerate(args.items()):
                        s += str(k) + "=" + (str(v) if not isinstance(v, str) else ("'" + v + "'"))
                        if i < len(args) - 1:
                            s += ", "
                    s += ")"
                    label = action.__name__ + s + ":" + str(score)
                    if len(label) > 40:
                        tokens = label.split(" ")
                        z = ""
                        i = 0
                        done = False
                        while i < len(tokens):
                            z += (" " if i > 0 else "") + tokens[i]
                            if not done and i < (len(tokens) - 1) and len(z + tokens[i + 1]) > 40:
                                z += "\n"
                                done = True
                            i += 1
                        label = z
                    graph.edge(from_state, to_state, label=" " + label + " ", fontsize='8')

        # Render the graph to a PDF
        if filename.lower().endswith(".pdf"):
            filename = filename[0:-4]
        graph.render(filename, format='pdf', cleanup=True)

    def __get_actual_params(self, action_name: str, args: dict):
        actual_params = {}
        params, defaults = self.action_to_params[action_name]
        for param_name in params:
            if param_name in args:
                actual_params[param_name] = args[param_name]
            elif param_name in self.param_buffer:
                actual_params[param_name] = self.param_buffer[param_name]
            elif param_name in defaults:
                actual_params[param_name] = defaults[param_name]
            else:
                if FiniteStateMachine.DEBUG:
                    print(f"   [DEBUG FSM] Getting actual params for {action_name}; missing param: " + str(param_name))
                return None
        return actual_params
