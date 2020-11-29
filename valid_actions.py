from pddlsim.external.liblapkt import Planner


class ValidActions():

    """
    ValidActions creates the provider for the actual valid action.
    It has a fallback if LAPKT isn't available to the python implementation, note that that implementation neither efficient nor stable
    """

    def __init__(self, parser, pddl, perception):
        self.problem = pddl.problem_path
        self.provider = PythonValidActions(parser, perception)

    def get(self, state=None):
        return self.provider.get(state)

    def on_action(self, action_sig):
        self.provider.on_action(action_sig)


class TrackedSuccessorValidActions():

    """
    Use the TrackedSuccessor to query for valid actions at the current state
    This successor is tracked because LAPKT needs to keep track of the state
    """

    def __init__(self, domain_path, problem_path):
        self.task = Planner()
        self.task.load(domain_path, problem_path)
        self.task.setup()
        self.sig_to_index = dict()
        for i in range(0, self.task.num_actions()):
            self.sig_to_index[self.task.get_action_signature(i)] = i

    def get(self):
        return map(str.lower, self.task.next_actions_from_current())

    def on_action(self, action_signature):
        """
        This is called by the SimulatorServices to notify that an action has been selected
        It is necessary because TrackedSuccessors keeps track of it's own state
        """
        self.task.proceed_with_action(
            self.sig_to_index[action_signature.upper()])


class PythonValidActions():

    """
    Python implemention for valid actions
    This is significantly less efficient than the TrackedSuccessor version
    """

    def __init__(self, parser, perception):
        self.parser = parser
        self.perception = perception

    def get(self, state=None):
        if state is None:
            current_state = self.perception.get_state()
        else:
            current_state = state
        possible_actions = []
        for (name, action) in self.parser.actions.items():
            for candidate in self.get_valid_candidates_for_action(current_state, action):
                possible_actions.append((action, action.action_string(candidate)))
        return possible_actions

    def join_candidates(self, previous_candidates, new_candidates, p_indexes, n_indexes):
        shared_indexes = p_indexes.intersection(n_indexes)
        if previous_candidates is None:
            return new_candidates
        result = []
        for c1 in previous_candidates:
            for c2 in new_candidates:
                if all([c1[idx] == c2[idx] for idx in shared_indexes]):
                    merged = c1[:]
                    for idx in n_indexes:
                        merged[idx] = c2[idx]
                    result.append(merged)
        return result

    def indexed_candidate_to_dict(self, candidate, index_to_name):
        return {name[0]: candidate[idx] for idx, name in index_to_name.items()}

    def on_action(self, action_sig):
        pass

    def get_valid_candidates_for_action(self, state, action):
        '''
        Get all the valid parameters for a given action for the current state of the simulation
        '''
        objects = dict()
        signatures_to_match = {
            name: (idx, t) for idx, (name, t) in enumerate(action.signature)}
        index_to_name = {idx: name for idx,
                         name in enumerate(action.signature)}
        candidate_length = len(signatures_to_match)
        found = set()
        candidates = None
        # copy all preconditions
        for precondition in sorted(action.precondition, key=lambda x: len(state[x.name])):
            thruths = state[precondition.name]
            if len(thruths) == 0:
                return []
            # map from predicate index to candidate index
            dtypes = [(name, 'object') for name in precondition.signature]
            reverse_map = {idx: signatures_to_match[pred][0] for idx, pred in enumerate(
                precondition.signature)}
            indexes = reverse_map.values()
            overlap = len(found.intersection(indexes)) > 0
            precondition_candidates = []
            for entry in thruths:
                candidate = [None] * candidate_length
                for idx, param in enumerate(entry):
                    candidate[reverse_map[idx]] = param
                precondition_candidates.append(candidate)

            candidates = self.join_candidates(
                candidates, precondition_candidates, found, indexes)
            # print( candidates)
            found = found.union(indexes)

        return [self.indexed_candidate_to_dict(c, index_to_name) for c in candidates]