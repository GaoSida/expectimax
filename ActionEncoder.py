import numpy as np

class ActionEncoder(object):
    """Encode action into vectors: one-hot or compressed sensing"""
    
    def __init__(self, num_skills, scheme='one-hot'):
        self.num_skills = num_skills
        self.scheme = scheme
        if scheme == 'one-hot':
            self.num_actions = num_skills * 2

    def getActionVector(self, skill, correct):
        if self.scheme == 'one-hot':
            vector = np.zeros([1, self.num_actions])
            vector[0][2 * skill + correct] = 1
            return vector