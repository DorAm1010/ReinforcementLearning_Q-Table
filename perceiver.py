import hashlib
from collections import OrderedDict


class Perceiver:
    def __init__(self, perceptor):
        self.last_state = None
        self.perceptor = perceptor
        self.current_state = None
        self.keys = None
        self.sorted_keys(perceptor.get_state())
        self.perceive(perceptor.get_state())

    def perceive(self, state):
        current_state = OrderedDict()

        for key in self.keys:
            values = list(state[key])
            values.sort()
            current_state[key] = values

        val = hashlib.sha1(str(current_state))
        self.current_state = val.hexdigest()

    def sorted_keys(self, state):
        given_information = {key: state[key] for key in state}
        self.keys = given_information.keys()
        self.keys.sort()

    def switch(self):
        self.last_state = self.current_state
