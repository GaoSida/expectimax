import random
import numpy as np

class RandomPlanner(object):
    """randomly give next question/skill"""
    def __init__(self, num_skills):
        self.num_skills = num_skills

    def getFirstQuestion(self):
        return random.randint(0, self.num_skills - 1)

    def getNextQuestion(self, last_skill, last_result):
        return random.randint(0, self.num_skills - 1)

class BlockPlanner(object):
    """
    Block by block planner. Move on to the next skill after mastery.
    Mastery means getting 3 questions correct in a row.
    """
    def __init__(self, num_skills, mastery=3):
        self.mastery = mastery
        self.num_skills = num_skills
        self.master_counter = 0         # current combo
        self.master_through = -1        # last mastered skill

    def getFirstQuestion(self):
        # reset
        self.master_counter = 0
        self.master_through = -1
        return 0

    def getNextQuestion(self, last_skill, last_result):
        if self.master_through != self.num_skills - 1:
            self.master_counter += last_result
            if self.master_counter == self.mastery:
                self.master_through += 1
                self.master_counter = 0

        if self.master_through == self.num_skills - 1:
            return random.randint(0, self.num_skills - 1)
        else:
            return self.master_through + 1
        