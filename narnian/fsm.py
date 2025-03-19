import os
import json
import copy
import inspect
import graphviz
from typing_extensions import Self
from collections.abc import Iterable


class FiniteStateMachine:
    DEBUG = False

    def __init__(self, actionable: object, wildcards: dict[str, str] | None = None):
        self.initial_state = None
        self.prev_state = None
        self.limbo_state = None
        self.state = None
        self.states = {}
        self.transitions = {}
        self.source_state_and_action_to_id = {}
        self.actionable = actionable
        self.wildcards = wildcards if wildcards is not None else {}
        self.param_buffer = {}  # dictionary of pre-buffered parameter names and their current value
        self.action_to_params = {}  # list of parameter names for each action
        self.action = None  # action that is being executed and its arguments
        self.action_step = 0
        self.__act_transitions_status = None
        self.__tot_states = 0
        self.__tot_actions = 0

    def __str__(self):

        fsm_data = {
            'initial_state': self.initial_state,
            'state': self.state,
            'prev_state': self.prev_state,
            'limbo_state': self.limbo_state,
            'state_actions': {
                state: [state_action_tuple[0].__name__ if state_action_tuple[0] is not None else None,
                        state_action_tuple[1], state_action_tuple[2]]
                for state, state_action_tuple in self.states.items() if state_action_tuple is not None
            },
            'transitions': {
                from_state: {
                    to_state: [[action.__name__, args, wait, act_id]
                               for (action, args, wait, act_id) in action_tuple]
                    for to_state, action_tuple in to_states.items()
                }
                for from_state, to_states in self.transitions.items()
            },
            'action': [self.action[0].__name__,
                       self.action[1], self.action[2]] if self.action is not None else None,
            'action_step': self.action_step,
            'param_buffer': self.param_buffer
        }

        def custom_serializer(obj):
            if not isinstance(obj, (int, str, float, bool, list, tuple, dict, set)):
                return "_non_basic_type_removed_"
            else:
                return obj

        json_string = json.dumps(fsm_data, indent=4, default=custom_serializer)
        return json_string

    def set_actionable(self, obj: object):
        """Set the object where actions should be found (as methods)."""

        self.actionable = obj

    def set_wildcards(self, wildcards: dict[str, str | float | int] | None):
        """Set the dictionary of wildcards used during the loading process."""

        self.wildcards = wildcards if wildcards is not None else {}

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

            for k, v in args.items():
                if isinstance(v, str):
                    for wildcard_from, wildcard_to in self.wildcards.items():
                        if wildcard_from == v and not isinstance(wildcard_to, str):
                            args[k] = wildcard_to
                        elif wildcard_from in v:
                            args[k] = v.replace(wildcard_from, wildcard_to)

            # saving operation
            self.states[state] = (action_callable, args, state_id)
            self.__tot_states += 1
        else:
            self.states[state] = (None, None, state_id)
            self.__tot_states += 1

        if len(self.states) == 1:
            self.set_state(state)

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

        if state in self.transitions or state in self.states:
            self.prev_state = self.state
            self.state = state
            if self.initial_state is None:
                self.initial_state = state
        else:
            raise ValueError("Unknown state: " + str(state))

    def add_transit(self, from_state: str, to_state: str,
                    action: str, args: dict | None = None, wait: bool | list[bool, bool] = False,
                    act_id: int | None = None):
        """Define a transition between two states with an associated action."""

        # fixing (when loading from JSON, "wait" is a list, but the whole code expects a tuple)
        if isinstance(wait, list):
            wait = tuple(wait)

        # plugging a previously loaded FSM
        if os.path.exists(to_state):
            file_name = to_state
            fsm = FiniteStateMachine(self.actionable).load(file_name)

            # first, we avoid name clashes, renaming already-used-state-names in original_name~1 (or ~2, or ~3, ...)
            fsm_states = list(fsm.states.keys())  # keep it as a list, since the dictionary will change
            for state in fsm_states:
                renamed_state = state
                i = 1
                while renamed_state in self.states or (i > 1 and renamed_state in fsm.states.keys()):
                    renamed_state = state + "." + str(i)
                    i += 1

                if fsm.initial_state == state:
                    fsm.initial_state = renamed_state
                if fsm.prev_state == state:
                    fsm.prev_state = renamed_state
                if fsm.state == state:
                    fsm.state = renamed_state
                if fsm.limbo_state == state:
                    fsm.limbo_state = renamed_state

                fsm.states[renamed_state] = fsm.states[state]
                if renamed_state != state:
                    del fsm.states[state]
                fsm.transitions[renamed_state] = fsm.transitions[state]
                if renamed_state != state:
                    del fsm.transitions[state]

                for to_states in fsm.transitions.values():
                    if state in to_states:
                        to_states[renamed_state] = to_states[state]
                        if renamed_state != state:
                            del to_states[state]

            # saving
            initial_state_was_set = self.initial_state is not None
            state_was_set = self.state is not None

            # include actions/states from another FSM
            self.include(fsm)

            # adding a transition to the initial state of the given FSM
            self.add_transit(from_state=from_state, to_state=fsm.initial_state, action=action, args=args,
                             wait=wait, act_id=None)

            # restoring
            self.initial_state = from_state if not initial_state_was_set else self.initial_state
            self.state = from_state if not state_was_set else self.state
            return

        # adding a new transition
        if from_state not in self.transitions:
            if from_state not in self.states:
                self.add_state_action(from_state, action=None)
            self.transitions[from_state] = {}
        if to_state not in self.transitions:
            if to_state not in self.states:
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

        for k, v in args.items():
            if isinstance(v, str):
                for wildcard_from, wildcard_to in self.wildcards.items():
                    if wildcard_from == v and not isinstance(wildcard_to, str):
                        args[k] = wildcard_to
                    elif wildcard_from in v:
                        args[k] = v.replace(wildcard_from, wildcard_to)

        if to_state not in self.transitions[from_state]:
            self.transitions[from_state][to_state] = [(action_callable, args, wait, act_id)]
            self.__tot_actions += 1
        else:
            if (action_callable, args, wait) in self.transitions[from_state][to_state]:
                raise ValueError(f"Repeated transition from {from_state} to {to_state}: "
                                 f"{(action, args, wait)}")
            self.transitions[from_state][to_state].append((action_callable, args, wait, act_id))
            self.__tot_actions += 1

    def include(self, fsm, make_a_copy=False):

        # adding states before adding transitions, so that we also add inner state actions, if any
        for _state, (_action_callable, _args, _) in fsm.states.items():
            if _args is not None:
                _args = copy.deepcopy(_args)
                for k, v in _args.items():
                    if isinstance(v, str):
                        for wildcard_from, wildcard_to in self.wildcards.items():
                            if wildcard_from == v and not isinstance(wildcard_to, str):
                                _args[k] = wildcard_to
                            elif wildcard_from in v:
                                _args[k] = v.replace(wildcard_from, wildcard_to)
            self.add_state_action(state=_state,
                                  action=_action_callable.__name__ if _action_callable is not None else None,
                                  args=_args, state_id=None)

        # copy all the transitions of the FSM
        for _from_state, _to_states in fsm.transitions.items():
            for _to_state, _action_tuples in _to_states.items():
                for (_action_callable, _args, _wait, _) in _action_tuples:
                    if _args is not None:
                        _args = copy.deepcopy(_args)
                        for k, v in _args.items():
                            if isinstance(v, str):
                                for wildcard_from, wildcard_to in self.wildcards.items():
                                    if wildcard_from == v and not isinstance(wildcard_to, str):
                                        _args[k] = wildcard_to
                                    elif wildcard_from in v:
                                        _args[k] = v.replace(wildcard_from, wildcard_to)
                    self.add_transit(from_state=_from_state, to_state=_to_state, action=_action_callable.__name__,
                                     args=_args, wait=_wait, act_id=None)

        if make_a_copy:
            self.state = fsm.state
            self.prev_state = fsm.state
            self.initial_state = fsm.initial_state
            self.limbo_state = fsm.limbo_state

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
        dest_state = None
        while len(actions_list) > 0:

            # debug only
            _debug_printed = False

            # it there was an already selected action (for example a multistep action), then continue with it,
            # otherwise, select a new one following a certain policy (actually, first-come first-served)
            if self.action is None:

                # naive policy: take the first action that is ready (i.e., not waiting)
                idx = -1
                for i, wait in enumerate(waits_list):
                    if wait is False:
                        idx = i
                        if dest_state is None:
                            dest_state = to_state_list[i]  # saving the dest state of the fist action
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
                        if self.action_step > 1:
                            return False  # early stop without clearing the action or changing state, prev_state, etc...
                else:
                    ret = action(**actual_params)  # instantaneous action
            else:
                if FiniteStateMachine.DEBUG:
                    print("   [DEBUG FSM] Tried and failed (missing actual param): " + action.__name__)
                    _debug_printed = True
                ret = False

            # clearing selected action, since it either completed or fully failed
            self.action = None
            self.action_step = 0

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
            else:

                # transition
                self.prev_state = self.limbo_state
                self.state = to_state_list[idx]
                self.limbo_state = None
                self.__act_transitions_status = None

                # replacing temporary wait tuple with the original wait value
                for to_state, action_tuples in self.transitions[self.state].items():
                    for i, (action, args, wait, act_id) in enumerate(action_tuples):
                        if isinstance(wait, tuple | list):
                            action_tuples[i] = (action, args, wait[1], act_id)  # replace tmp with original (restore)

                # clearing all action-related arguments
                self.clear_param_buffer()

                return True  # action done!

        # no actions were applied
        return False

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
                        for j, (__action, __args, __wait, __act_id) in enumerate(_action_tuples):

                            if isinstance(__wait, tuple):
                                return False  # some actions were already temporarily un-blocked or blocked!

                            if (__action.__name__ == action_name and
                                    (args is None or __args == args)):
                                self.transitions[from_state][_to_state][j] = \
                                    (__action, __args, (False, __wait), __act_id)  # un-block action (make it ready!)
                            else:
                                self.transitions[from_state][_to_state][j] = \
                                    (__action, __args, (True, __wait), __act_id)  # block all other actions
                    return True
        return False

    def wait_for_all_actions_that_start_with(self, prefix):
        """Forces the wait flag to all actions whose name start with a given prefix."""

        for state, to_states in self.transitions.items():
            for to_state, action_tuples in to_states.items():
                for i, (action, args, wait, act_id) in enumerate(action_tuples):
                    if action.__name__.startswith(prefix):
                        wait = True
                        action_tuples[i] = (action, args, wait, act_id)

    def wait_for_actions(self, from_state: str, to_state: str, wait: bool = True):
        """Forces the wait flag to a specific action."""

        if from_state not in self.transitions or to_state not in self.transitions[from_state]:
            return False

        for i, (action, args, _wait, act_id) in enumerate(self.transitions[from_state][to_state]):
            self.transitions[from_state][to_state][i] = (action, args, wait, act_id)
        return True

    def save(self, filename: str):
        """Save the FSM to a JSON file."""

        with (open(filename, 'w') as file):
            file.write(str(self))

    def load(self, filename: str) -> Self:
        """Load the FSM state from a JSON file and resolve actions."""

        with open(filename, 'r') as file:
            fsm_data = json.load(file)

        self.initial_state = fsm_data['initial_state']
        self.state = fsm_data['state']
        self.prev_state = fsm_data['prev_state']
        self.limbo_state = fsm_data['limbo_state']
        self.param_buffer = fsm_data['param_buffer']
        self.action = (self.action_name_to_callable(fsm_data['action'][0]),
                       fsm_data['action'][1]) if fsm_data['action'] is not None else None
        self.action_step = fsm_data['action_step']

        self.states = {}
        for state, (action_name, args, state_id) in fsm_data['state_actions'].items():
            if args is not None:
                for k, v in args.items():
                    if isinstance(v, str):
                        for wildcard_from, wildcard_to in self.wildcards.items():
                            if wildcard_from == v and not isinstance(wildcard_to, str):
                                args[k] = wildcard_to
                            elif wildcard_from in v:
                                args[k] = v.replace(wildcard_from, wildcard_to)
            self.add_state_action(state, action=action_name, args=args, state_id=state_id)

        self.transitions = {}
        for from_state, to_states in fsm_data['transitions'].items():
            for to_state, action_tuples in to_states.items():
                for action_name, args, wait, act_id in action_tuples:
                    if args is not None:
                        for k, v in args.items():
                            if isinstance(v, str):
                                for wildcard_from, wildcard_to in self.wildcards.items():
                                    if wildcard_from == v and not isinstance(wildcard_to, str):
                                        args[k] = wildcard_to
                                    elif wildcard_from in v:
                                        args[k] = v.replace(wildcard_from, wildcard_to)
                    self.add_transit(from_state, to_state, action_name, args, wait, act_id)

        return self

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

    def print_actions(self, state: str | None = None):
        state = (self.state if self.state is not None else self.limbo_state) if state is None else state
        for to_state, action_tuples in self.transitions[state].items():
            if action_tuples is None:
                print(f"{state}, no actions")
            for i, (action, args, wait, act_id) in enumerate(action_tuples):
                print(f"{state} -> {to_state}, action: {action.__name__}, args: {args}, wait: {wait}, act_id: {act_id}")

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
