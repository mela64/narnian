import json
import torch
import inspect
import graphviz
from collections.abc import Iterable


class FiniteStateMachine:
    DEBUG = True

    def __init__(self, actionable: object, policy: str = "sampling"):
        self.initial_state = None
        self.prev_state = None
        self.limbo_state = None
        self.state = None
        self.states = {}
        self.transitions = {}
        self.source_state_and_action_to_id = {}
        self.actionable = actionable
        self.param_buffer = {}  # dictionary of pre-buffered parameter names and their current value
        self.action_to_params = {}  # list of parameter names for each action
        self.policy = policy
        self.action = None  # action that is being executed and its arguments
        self.action_step = 0
        self.__act_transitions_status = None
        self.__tot_states = 0
        self.__tot_actions = 0

        assert self.policy in ["uniform", "sampling"], \
            f"Invalid policy: {self.policy}, it must be in ['uniform', 'sampling']"

    def __str__(self):

        fsm_data = {
            'initial_state': self.initial_state,
            'state': self.state,
            'prev_state': self.prev_state,
            'limbo_state': self.limbo_state,
            'policy': self.policy,
            'state_actions': {
                state: str([state_action_tuple[0].__name__ if state_action_tuple[0] is not None else None,
                            state_action_tuple[1]])
                for state, state_action_tuple in self.states.items() if state_action_tuple is not None
            },
            'transitions': {
                from_state: {
                    to_state: str([[action.__name__, args, wait, act_id]
                                   for (action, args, wait, act_id) in action_tuple])
                    for to_state, action_tuple in to_states.items()
                }
                for from_state, to_states in self.transitions.items()
            },
            'action': str([self.action[0].__name__,
                           self.action[1], self.action[2]]) if self.action is not None else None,
            'action_step': self.action_step,
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

    def add_state_action(self, state: str, action: str = None, args: dict | None = None, state_id: int | None = None):
        """Add an action (inner action) to an existing state (or creating it from scratch, if not existent)."""

        if args is None:
            args = {}
        if state_id is None:
            state_id = self.__tot_states

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
            self.states[state] = (action_callable, args, state_id)
            self.__tot_states += 1
        else:
            self.states[state] = (None, None, state_id)
            self.__tot_states += 1

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
                    action: str, args: dict | None = None, wait: bool = False, act_id: int | None = None):
        """Define a transition between two states with an associated action."""

        if from_state not in self.transitions:
            self.add_state_action(from_state, action=None)
            self.transitions[from_state] = {}
        if to_state not in self.transitions:
            self.add_state_action(to_state, action=None)
            self.transitions[to_state] = {}
        if args is None:
            args = {}
        if act_id is None:
            act_id = self.__tot_actions

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
            self.transitions[from_state][to_state] = [(action_callable, args, wait, act_id)]
            self.__tot_actions += 1
        else:
            if (action_callable, args, wait) in self.transitions[from_state][to_state]:
                raise ValueError(f"Repeated transition from {from_state} to {to_state}: "
                                 f"{(action, args, wait)}")
            self.transitions[from_state][to_state].append((action_callable, args, wait, act_id))
            self.__tot_actions += 1

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

        if self.state is not None:  # when in the middle of an action, the state is None
            state_action_tuple = self.states[self.state]
            if state_action_tuple is not None and state_action_tuple[0] is not None:
                action, args, _ = state_action_tuple

                # getting inner-state-action parameters, considering their current buffered values and the given ones
                actual_params = self.__get_actual_params(action.__name__, args)

                # call inner state action
                if actual_params is not None:
                    action(**actual_params)

    def act_transitions(self):

        # collecting list of feasible actions, wait flags, etc. (from the current state)
        if self.__act_transitions_status is None:
            actions_list = []
            args_list = []
            waits_list = []
            to_state_list = []
            actions_ids = []

            for to_state, action_tuples in self.transitions[self.state].items():
                for i, (action, args, wait, act_id) in enumerate(action_tuples):
                    if (not isinstance(wait, tuple | list) and wait is False) or \
                            (isinstance(wait, tuple | list) and wait[0] is False):
                        actions_list.append(action)
                        args_list.append(args)
                        to_state_list.append(to_state)
                        actions_ids.append(act_id)
                        if not isinstance(wait, tuple | list):  # if "wait" is a tuple, it includes a tmp value
                            waits_list.append(wait)
                        else:
                            waits_list.append(wait[0])  # tmp value
                    if isinstance(wait, tuple | list):
                        action_tuples[i] = (action, args, wait[1], act_id)  # replace tmp with original (restore)

            if len(actions_list) > 0:
                self.__act_transitions_status = {
                    'actions_list': actions_list,
                    'args_list': args_list,
                    'waits_list': waits_list,
                    'to_state_list': to_state_list,
                    'actions_ids': actions_ids,
                    'idx': 0
                }
        else:

            # reloading the already computed set of actions, wait flags, etc. (when in the middle of an action)
            actions_list = self.__act_transitions_status['actions_list']
            args_list = self.__act_transitions_status['args_list']
            waits_list = self.__act_transitions_status['waits_list']
            to_state_list = self.__act_transitions_status['to_state_list']
            actions_ids = self.__act_transitions_status['actions_ids']

        # using the selected policy to decide what action to apply
        while len(actions_list) > 0:

            # debug only
            _debug_printed = False

            if self.action is None:

                # naive policy: take the first action that is ready (i.e., not waiting)
                idx = -1
                for i, wait in enumerate(waits_list):
                    if wait is False:
                        idx = i
                        break
                if idx == -1:
                    break

                # picking up the actual parameters, considering their current buffered values and the given ones
                actual_params = self.__get_actual_params(actions_list[idx].__name__, args_list[idx])

                # saving current action
                self.limbo_state = self.state
                self.state = None
                self.action = (actions_list[idx], actual_params, actions_ids[idx])
                self.__act_transitions_status['idx'] = idx

            # references
            action = self.action[0]
            actual_params = self.action[1]
            idx = self.__act_transitions_status['idx']

            # call action
            if actual_params is not None:
                if 'steps' in actual_params:  # multistep action
                    ret = action(**actual_params)
                    if ret:
                        self.action_step += 1
                        if self.action_step < actual_params['steps']:
                            return False  # early stop without clearing the action or changing state, prev_state, etc...
                else:
                    ret = action(**actual_params)  # instantaneous action
            else:
                if FiniteStateMachine.DEBUG:
                    print("   [DEBUG FSM] Tried and failed (missing actual param): " + action.__name__)
                    _debug_printed = True
                ret = False

            # clearing action
            self.action = None
            self.action_step = 0

            # clearing action-related arguments
            if actual_params is not None:
                for param in actual_params.keys():
                    if param in self.param_buffer.keys():
                        del self.param_buffer[param]

            # if the action failed, another one will be sampled
            if not ret:
                self.state = self.limbo_state
                self.limbo_state = None
                if FiniteStateMachine.DEBUG:
                    if not _debug_printed:
                        print("   [DEBUG FSM] Tried and failed (failed execution): " + action.__name__)
                del actions_list[idx]
                del args_list[idx]
                del waits_list[idx]
                del to_state_list[idx]
                del actions_ids[idx]

                if len(actions_list) == 0:
                    self.__act_transitions_status = None
                return False
            else:

                # transition
                self.prev_state = self.limbo_state
                self.state = to_state_list[idx]
                self.limbo_state = None
                self.__act_transitions_status = None
                return True

    def clear_param_buffer(self):
        self.param_buffer = {}

    def set_next_action(self, action_name: str, args: dict | None = None,
                        from_state: str | None = None, to_state: str | None = None):
        """Forces the next action (temporary setting all the others to a 'wait' status), if the action exists."""

        if from_state is None:
            from_state = self.state

        if from_state not in self.transitions:
            return False

        if to_state is not None and to_state not in self.transitions[from_state]:
            return False

        to_states = self.transitions[from_state].keys() if to_state is None else [to_state]

        for to_state in to_states:
            for i, (_action, _args, _wait, _act_id) in enumerate(self.transitions[from_state][to_state]):
                if _action.__name__ == action_name and (args is None or _args == args):
                    for _to_state, _action_tuples in self.transitions[from_state].items():
                        for j, (__action, __args, __wait, __act_id) \
                                in enumerate(_action_tuples):
                            if (__action.__name__ == action_name and
                                    (args is None or __args == args)):
                                self.transitions[from_state][_to_state][j] = \
                                    (__action, __args, (False, __wait), __act_id)  # un-block action (make it ready!)
                            else:
                                self.transitions[from_state][_to_state][j] = \
                                    (__action, __args, (True, __wait), __act_id)  # block action
                    return True
        return False

    def wait_for_all_action_that_start_with(self, prefix):
        """Forces the wait flag to all actions whose name start with a given prefix."""

        for state, to_states in self.transitions.items():
            for to_state, action_tuples in to_states.items():
                for i, (action, args, wait, act_id) in enumerate(action_tuples):
                    if action.__name__.startswith(prefix):
                        wait = True
                        action_tuples[i] = (action, args, wait, act_id)

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
        self.limbo_state = fsm_data['limbo_state']
        self.policy = fsm_data['policy']
        self.param_buffer = fsm_data['param_buffer']
        self.action = (self.action_name_to_callable(fsm_data['action'][0]),
                       fsm_data['action'][1] if fsm_data['action'] is not None else None)
        self.action_step = fsm_data['action_step']

        self.states = {}
        for state, (action_name, args, state_id) in fsm_data['states'].items():
            self.add_state_action(state, action=action_name, args=args, state_id=state_id)

        self.transitions = {}
        for from_state, to_states in fsm_data['transitions'].items():
            for to_state, action_tuples in to_states.items():
                for action_name, args, wait, act_id in action_tuples:
                    self.add_transit(from_state, to_state, action_name, args, wait, act_id)

    def to_graphviz(self):
        """Encode the FSM in GraphViz format."""

        graph = graphviz.Digraph()
        graph.attr('node', fontsize='8')
        for state, state_action_tuple in self.states.items():
            action, args, state_id = state_action_tuple
            if action is not None:
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
            graph.node(state, state + suffix, _attributes={'id': "node" + str(state_id)})

        for from_state, to_states in self.transitions.items():
            for to_state, action_tuples in to_states.items():
                for action, args, wait, act_id in action_tuples:
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
                                z += "\n"
                                done = True
                            i += 1
                        label = z
                    graph.edge(from_state, to_state, label=" " + label + " ", fontsize='8',
                               _attributes={'id': "edge" + str(act_id)})
        return graph

    def save_pdf(self, filename: str):
        """Save the FSM in GraphViz format, drawn on a PDF file."""

        if filename.lower().endswith(".pdf"):
            filename = filename[0:-4]
        self.to_graphviz().render(filename, format='pdf', cleanup=True)

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
