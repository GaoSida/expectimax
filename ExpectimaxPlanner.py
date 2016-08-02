import copy
import random
import numpy as np

class MaxNode(object):
    """Max Node in Expectimax Tree"""
    def __init__(self, num_children_dict, student, last_skill=-1, last_result=-1):
        self.num_children = num_children_dict['Max']
        self.num_children_dict = num_children_dict
        self.student = student.getCopy()        # Each Node has a copy of student status

        # Root node will have -1 for last skill and result.
        # If not root, then run the student model one step.
        if last_skill != -1 and last_result != -1:
            self.student.fixResultRun(last_skill, last_result)
        self.p_correct_skill = self.student.getPrediction()

        self.children = list()          # Leaf Node won't have children
        self.children_values = list()

    def getMax(self, max_depth):
        if max_depth == 0:
            return np.mean(self.p_correct_skill)
        # First time to arrive this node, and max_depth != 0, then develop children
        if len(self.children) == 0:
            for i in range(self.num_children):
                self.children.append(ExpectNode(self.num_children_dict, self.student,
                    last_skill=i, p_children=[1 - self.p_correct_skill[i], self.p_correct_skill[i]]))
                # only count depth on Max Nodes
                self.children_values.append(self.children[-1].getExpect(max_depth - 1))
        else:   # Children have developed, then just get the value of children
            for i in range(self.num_children):
                self.children_values[i] = self.children[i].getExpect(max_depth - 1)

        return np.max(self.children_values)

    # This method is only called on root nodes
    def getArgMax(self, max_depth):
        if max_depth == 0:      # 0-step look ahead means do nothing
            return random.randint(0, self.num_children - 1)
        else:
            self.getMax(max_depth)
            return np.argmax(self.children_values)

    # This method is only called on root nodes.
    def reset(self):
        self.student.reset()
        self.p_correct_skill = self.student.getPrediction()
        self.children = list()
        self.children_values = list()

class ExpectNode(object):
    """ExpectNode in Expectimax Tree"""
    def __init__(self, num_children_dict, student, last_skill, p_children):
        # ExpectNode always have children
        self.children = list()
        self.children_values = list()
        self.num_children = num_children_dict['Expect']
        for i in range(self.num_children):
            self.children.append(MaxNode(num_children_dict, student, 
                                         last_skill=last_skill, last_result=i))
            self.children_values.append(0)
        
        self.p_children = p_children

    def getExpect(self, max_depth):
        expectation = 0
        for i in range(self.num_children):
            self.children_values[i] = self.children[i].getMax(max_depth)
            expectation += self.children_values[i] * self.p_children[i]
        return expectation
        

class ExpectimaxPlanner(object):
    """Planner based on expectimax Tree search"""
    def __init__(self, num_skills, num_results, student, max_depth):
        self.num_children_dict = {'Max': num_skills, 'Expect': num_results}
        self.max_depth = max_depth
        self.root = MaxNode(self.num_children_dict, student)

    def getFirstQuestion(self):
        # reset
        self.root.reset()
        return self.root.getArgMax(self.max_depth)

    def getNextQuestion(self, last_skill, last_result):
        self.root = self.root.children[last_skill].children[last_result]
        return self.root.getArgMax(self.max_depth)
        
        
        