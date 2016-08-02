import random
import copy
import tensorflow as tf
import pandas as pd

class DKTStudent(object):
    """Student powered by DKT."""
    def __init__(self, encoder=None, dkt_graph=None, model_file='', copy_from=None):
        if copy_from == None:
            self.encoder = encoder
            self.graph = dkt_graph
            self.session = tf.Session(graph=self.graph.graph)
            self.graph.saver.restore(self.session, model_file)
            self.current_output, self.current_state, self.current_status = \
                self.session.run([self.graph.initial_output, self.graph.initial_state, \
                             self.graph.initial_status])
        else:
            self.encoder = copy_from.encoder
            self.graph = copy_from.graph
            self.session = copy_from.session
            self.current_output = copy.deepcopy(copy_from.current_output)
            self.current_state = copy.deepcopy(copy_from.current_state)
            self.current_status = copy.deepcopy(copy_from.current_status)

    def fixResultRun(self, skill, result):
        input_vector = self.encoder.getActionVector(skill, result)
        feed_dict = dict()
        feed_dict[self.graph.current_state] = self.current_state
        feed_dict[self.graph.current_output] = self.current_output
        feed_dict[self.graph.current_input] = input_vector

        self.current_output, self.current_state, self.current_status = \
            self.session.run([self.graph.next_output, self.graph.next_state, \
                         self.graph.next_status], feed_dict=feed_dict)

    def simulationRun(self, skill):
        p_correct = self.current_status[0, skill]
        if random.random() < p_correct:
            result = 1
        else:
            result = 0
        self.fixResultRun(skill, result)
        return result

    def getPrediction(self):
        #  For DKT, status == prediction, just flattened
        return self.current_status.reshape(-1)

    def reset(self):
        self.current_output, self.current_state, self.current_status = \
            self.session.run([self.graph.initial_output, self.graph.initial_state, \
                         self.graph.initial_status])

    def getCopy(self):
        return DKTStudent(copy_from=self)


class BKTStudent(object):
    """Student powered by BKT"""
    def __init__(self, model_file='', copy_from=None):
        if copy_from == None:
            bkt_model = pd.read_csv(model_file)
            self.num_skills = len(bkt_model)
            self.p_prior = bkt_model['prior'].values
            self.current_status = copy.deepcopy(self.p_prior)
            self.p_transition = bkt_model['p_transition'].values
            self.p_guess = bkt_model['p_guess'].values
            self.p_slip = bkt_model['p_slip'].values
            prereqs_list = bkt_model['prerequisites'].values
            self.prerequisites = dict()
            for i in range(self.num_skills):
                self.prerequisites[i] = set()
                prereqs = prereqs_list[i].split()
                for skill in prereqs:
                    if int(skill) >= 0:
                        self.prerequisites[i].add(int(skill))
            del bkt_model
        else:
            self.num_skills = copy_from.num_skills
            self.p_prior = copy_from.p_prior
            self.current_status = copy.deepcopy(copy_from.current_status)
            self.p_transition = copy_from.p_transition
            self.p_guess = copy_from.p_guess
            self.p_slip = copy_from.p_slip
            self.prerequisites = copy_from.prerequisites

    def fixResultRun(self, skill, result):
        # BKT / Classic HMM Filtering
        p_last_know = self.current_status[skill]
        p_last_notknow = 1 - p_last_know
        
        if result == 0:
            p_last_know_result = self.p_slip[skill]
            p_last_notknow_result = 1 - self.p_guess[skill]
        else:
            p_last_know_result = 1 - self.p_slip[skill]
            p_last_notknow_result = self.p_guess[skill]

        post_last_know_weight = p_last_know * p_last_know_result
        post_last_notknow_weight = p_last_notknow * p_last_notknow_result
        p_post_last_know = post_last_know_weight / (post_last_know_weight + post_last_notknow_weight)
        
        p_able_to_learn = 1
        for k in self.prerequisites[skill]:
            p_able_to_learn = p_able_to_learn * self.current_status[k]
        
        self.current_status[skill] = p_post_last_know + \
                (1 - p_post_last_know) * p_able_to_learn * self.p_transition[skill]

    def simulationRun(self, skill):
        p_correct = self.current_status[skill] * (1 - self.p_slip[skill]) + \
                    (1 - self.current_status[skill]) * self.p_guess[skill]
        if random.random() < p_correct:
            result = 1
        else:
            result = 0
        self.fixResultRun(skill, result)
        return result

    def getPrediction(self):
        # For BKT, prediction != status, because of slip and guess
        return self.current_status * (1 - self.p_slip) + (1 - self.current_status) * self.p_guess

    def reset(self):
        self.current_status = copy.deepcopy(self.p_prior)

    def getCopy(self):
        return BKTStudent(copy_from=self)
        