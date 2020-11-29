# Dor Amrani 205927916
import random as rand
import atexit
import cPickle as pickle

from collections import OrderedDict

from perceiver import Perceiver
from valid_actions import PythonValidActions
from operator import itemgetter

from pddlsim.parser_independent import Literal, ProbabilisticAction


class Learner:
    def __init__(self):
        self.learn_rate = 0.6
        self.discount = 0.3
        self.exploration_rate = 0.7
        self.last_state_goals = 0
        self.services = None
        self.actions = None
        self.perceiver = None
        self.last_action = None
        self.q_dict = None
        self.goals = set()
        self.goals_achieved = set()
        self.route = OrderedDict()
        self.times_chosen_action = {}
        self.state_action_space = {}

    def initialize(self, services):
        self.perceiver = Perceiver(services.perception)
        self.services = services
        self.actions = PythonValidActions(services.parser, services.perception)
        self.deterministic_or_not()
        filename = services.parser.problem_name + '.txt'
        self.init_dictionaries(filename)
        self.parse_goals(services.parser.goals)
        atexit.register(self.save_to_files, filename)

    def next_action(self):
        if self.services.goal_tracking.reached_all_goals():
            self.award_route(len(self.goals))
            value = self.q_dict[self.perceiver.last_state][self.last_action]
            self.update_table(value, len(self.goals))
            return None
        state = self.services.perception.get_state()
        self.perceiver.perceive(state)

        if self.perceiver.current_state not in self.q_dict:
            # first time in domain or state in domain
            self.new_state(state)

        if self.perceiver.last_state is None:
            # not first time in domain but first run in episode
            self.first_action()
            return self.last_action

        try:
            self.choose_action(state)
        except:
            self.award_route(-1)
            actions = self.actions.get(state)
            self.last_action = rand.choice(actions)[1]
            self.perceiver.switch()
            self.action_chosen()
        return self.last_action

    def new_state(self, state):
        possible_actions = self.actions.get(state)

        self.q_dict[self.perceiver.current_state] = \
            {action_name: 0 for action, action_name in possible_actions}
        self.times_chosen_action[self.perceiver.current_state] = \
            {action_name: 0 for action, action_name in possible_actions}
        self.state_action_space[self.perceiver.current_state] = len(possible_actions)

    def choose_action(self, state):
        explore_vs_exploit = rand.uniform(0, 1)
        if explore_vs_exploit > self.exploration_rate:
            # balance between exploration and exploitation
            self.explore(state)
        else:
            self.choose_max(state)
        # self.choose_max(state)
        self.action_chosen()

    def get_reward(self, state):
        new_goals = {goal for goal in self.goals_achieved}
        for goal in self.goals:
            if self.services.parser.test_condition(goal, state):
                new_goals.add(goal)

        num_of_new_goals = len(new_goals)
        num_of_total_goals = len(self.goals_achieved)
        if num_of_new_goals == num_of_total_goals:
            num_of_new_goals = -0.5
        elif num_of_new_goals < num_of_total_goals:
            num_of_new_goals = -1
            self.award_route(-1)
        else:
            # new goal was attained add to state
            self.award_route(1)
        self.goals_achieved = new_goals

        return num_of_new_goals

    def update_table(self, max_val, reward):
        # update table
        current_value = self.q_dict[self.perceiver.last_state][self.last_action]
        self.q_dict[self.perceiver.last_state][self.last_action] = \
            (1 - self.learn_rate) * current_value + \
            self.learn_rate * (reward + self.discount * max_val)

    def explore(self, state):
        # randomly select available action
        possible_actions = self.actions.get(state)
        current_action = rand.choice(possible_actions)[1]

        # get its table quality and update table
        action_quality = self.q_dict[self.perceiver.current_state][current_action]
        reward = self.get_reward(state)
        self.update_table(action_quality, reward)

        # prepare next iteration
        self.perceiver.switch()
        self.last_action = current_action

    def get_max_action(self):
        dict_values = list(self.q_dict[self.perceiver.current_state].items())
        max_action_quality = max(dict_values, key=itemgetter(1))

        return max_action_quality

    def parse_goals(self, goals):
        for goal in goals:
            if isinstance(goal, Literal):
                self.goals.add(goal)
            else:
                self.parse_goals(goal.parts)

    def deterministic_or_not(self):
        actions_dict = self.services.parser.actions
        for action_name in self.services.parser.actions:
            if isinstance(actions_dict[action_name], ProbabilisticAction):
                return
        exit(128)

    def min_chosen_action(self, state):
        possible_actions = self.actions.get(state)
        considered_action = possible_actions[0][1]
        times_chosen = self.times_chosen_action[self.perceiver.current_state][considered_action]

        for action, action_name in possible_actions[1:]:
            times_chosen_other = self.times_chosen_action[self.perceiver.current_state][action_name]

            if times_chosen_other < times_chosen:
                considered_action = action_name
                times_chosen = times_chosen_other

        reward = self.get_reward(state)
        action_quality_in_table = self.q_dict[self.perceiver.current_state][considered_action]
        self.update_table(action_quality_in_table, reward)
        self.last_action = considered_action
        self.perceiver.switch()

    def action_chosen(self):
        try:
            self.times_chosen_action[self.perceiver.last_state][self.last_action] += 1
        except KeyError:
            self.times_chosen_action[self.perceiver.last_state][self.last_action] = 1
        self.route[self.perceiver.last_state] = self.last_action

    def first_action(self):
        max_action = self.get_max_action()
        self.perceiver.switch()
        self.last_action = max_action[0]
        self.action_chosen()

    def award_route(self, goals_achieved):
        # the more actions the agent has means he has more
        # control over the state means it is probably has the better expectancy
        step = 1.0
        route_len = len(self.route)
        for state in self.route:
            action = self.route[state]
            num_of_actions = float(self.state_action_space[state])
            times_action_was_chosen = self.times_chosen_action[state][action]

            val = goals_achieved * num_of_actions / times_action_was_chosen
            self.q_dict[state][action] += step * val / route_len
            step += 1.0

    def choose_max(self, state):
        reward = self.get_reward(state)

        max_action = self.get_max_action()
        self.update_table(max_action[1], reward)

        self.last_action = max_action[0]
        self.perceiver.switch()

    def init_dictionaries(self, filename):
        q_dict = OrderedDict()
        try:
            with open(filename, 'rb') as load_file:
                q_dict = pickle.load(load_file)
        except IOError:
            pass
        self.q_dict = q_dict
        for state in q_dict:
            self.times_chosen_action[state] = {}
            self.state_action_space[state] = len(q_dict[state])
            for action in state:
                self.times_chosen_action[state][action] = 0

    def save_to_files(self, filename):
        with open(filename, 'wb') as save_file:
            pickle.dump(self.q_dict, save_file)

    def get_action_quality(self, state, action):
        try:
            return self.q_dict[state][action]
        except KeyError:
            try:
                self.q_dict[state][action] = 0
            except KeyError:
                self.q_dict[state] = {action: 0}
            return 0
