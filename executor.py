import cPickle as pickle
import random
from operator import itemgetter

from pddlsim.executors.executor import Executor
import pddlsim.planner as planner
from perceiver import Perceiver
from valid_actions import PythonValidActions


class PlanDispatcher(Executor):
    def __init__(self, services):
        super(PlanDispatcher, self).__init__()
        self.steps = planner.make_plan(services.pddl.domain_path, services.pddl.problem_path)

    def next_action(self):
        if len(self.steps) > 0:
            return self.steps.pop(0).lower()
        return None


class MyExecutor:
    def __init__(self):
        self.times_chosen_action = {}
        self.services = None
        self.steps = []
        self.q_dict = {}
        self.task = None
        self.start_state = None
        self.sorted_keys = None
        self.perceiver = None
        self.planner = None
        self.actions = None

    def initialize(self, services):
        self.services = services
        self.actions = PythonValidActions(services.parser, services.perception)
        self.perceiver = Perceiver(services.perception)
        try:
            filename = services.parser.problem_name + '.txt'
            print filename
            with open(filename, 'rb') as load_file:
                self.q_dict = pickle.load(load_file)
            self.task = self.execute_policy
            self.start_state = self.services.perception.get_state()
            self.sorted_keys = self.start_state.keys()
            self.sorted_keys.sort()
        except IOError:
            self.steps = planner.make_plan(services.pddl.domain_path, services.pddl.problem_path)
            self.planner = PlanDispatcher(services)
            self.task = self.planner.next_action

    def next_action(self):
        try:
            return self.task()
        except:
            state = self.services.perception.get_state()
            actions = self.actions.get(state)
            return random.choice(actions)[1]

    def execute_policy(self):
        if self.services.goal_tracking.reached_all_goals():
            return None
        state = self.services.perception.get_state()
        self.perceiver.perceive(state)

        if random.uniform(0, 1) > 0.75:
            # choose action with more options
            action_tuple = max(self.actions.get(state), key=itemgetter(1))
            action = action_tuple[1]
        else:
            dict_values = list(self.q_dict[self.perceiver.current_state].items())
            action = max(dict_values, key=itemgetter(1))[0]

        return action
