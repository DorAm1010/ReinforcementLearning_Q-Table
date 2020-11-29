# Dor Amrani 205927916
import sys

from executor import MyExecutor
from pddlsim.local_simulator import LocalSimulator
from learner import Learner


def main(argv):
    domain_path = argv[1]
    problem_path = argv[2]
    if argv[0] == "-L":
        agent = Learner()
    else:
        agent = MyExecutor()
    print LocalSimulator().run(domain_path, problem_path, agent)


if __name__ == "__main__":
    main(sys.argv[1:])
