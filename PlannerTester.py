import numpy as np
import pandas as pd

class PlannerTester(object):
    """Test Frame for a planner"""

    def runTest(self, planner, student, num_iterations, length, verbose):
        avg_status_list = []

        for i in range(num_iterations):
            student.reset()
            # First question
            skill = planner.getFirstQuestion()
            result = student.simulationRun(skill)
            # The rest questions
            for j in range(length - 1):
                if verbose >= 2:
                    print 'o',
                skill = planner.getNextQuestion(skill, result)
                result = student.simulationRun(skill)
            if verbose >= 2:
                    print 'o'

            avg_status_list.append(np.mean(student.getPrediction()))

            if verbose >= 1:
                print "iteration " + str(i) + ": " + str(avg_status_list[-1])

        return avg_status_list

    def evaluate(self, planner, student, num_iterations, length, verbose=0):
        return pd.DataFrame(self.runTest(planner, student, num_iterations, length, verbose)).describe()