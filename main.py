import functools
import itertools
import math
import multiprocessing as mp
import pickle
import time

import matplotlib.pyplot as plt
import nashpy as nash
import numpy as np
import pandas as pd
import scipy as sp
import tqdm
from scipy.stats import entropy
from sklearn import linear_model


def find_nash_equilibrium(from_states, estimated_q_function, row_num):
    token_pos = from_states.iloc[row_num, 0]
    budget_A = from_states.iloc[row_num, 1]
    budget_B = from_states.iloc[row_num, 2]
    matrixGame = nash.Game(estimated_q_function[token_pos, budget_A, budget_B, :, :])
    equilibriums = matrixGame.support_enumeration(non_degenerate = False, tol = 0)
    max_entropy = -np.inf
    policy_A = None
    policy_B = None
    for eqs in equilibriums:
        curr_policy_B = eqs[0]
        curr_policy_A = eqs[1]
        curr_entropy = entropy(curr_policy_A) + entropy(curr_policy_B)
        if max_entropy < curr_entropy:
            max_entropy = curr_entropy
            policy_A = curr_policy_A
            policy_B = curr_policy_B
    return [policy_A, policy_B]


class Alesia():
    def __init__(self, budget, token_space):
        self.budget = budget
        self.token_space = token_space
        self.terminate = False
        
        # A tuple (token_pos, budget_A, budget_B)
        # token_pos can go from 0 to token_space + 1. 
        # budget_A and budget_B can go from 0 to budget
        self.state = (math.floor(self.token_space / 2) , self.budget, self.budget)

        self.t = 0

        self.state_transition_dist = self.get_state_transition_dist()
        self.expected_reward = self.get_expected_reward()
        self.initial_state_dist = np.zeros((self.token_space + 2, self.budget + 1, self.budget + 1))
        self.initial_state_dist[self.state[0], self.state[1], self.state[2]] = 1

    def reset(self):
        # Full budget for player A and B, token_space is set to be the middle
        # Fair game if token_space is odd, favour to player A (go to index 0) if even
        self.state = (math.floor(self.token_space / 2) , self.budget, self.budget)
        self.terminate = False
        self.t = 0
        return self.state

    @staticmethod
    def check_termination(token_pos, token_space, budget_A, budget_B):
        if token_pos == 0 or token_pos == token_space + 1:
            return True
        elif budget_A == 0 and budget_B == 0:
            return True
        else:
            return False

    @staticmethod
    def get_reward(token_pos, budget_A, budget_B, action_A, action_B, token_space):  
        if token_pos == 0:
            return -1, -1
        if token_pos == token_space + 1:
            return 1, 1
        if action_A is None or action_B is None:
            return 0, 0
        if action_B > action_A:
            if token_pos == token_space:
                return 1, 1

            lower_point = np.linspace(start = 0, stop = 0.5, num = budget_B, endpoint = False)
            lower_point = lower_point[len(lower_point) - (action_B - action_A)]
            return np.random.uniform(low = lower_point, high = 0.5, size = 1)[0], 0.5 * (0.5 + lower_point)
            
        elif action_A > action_B:
            if token_pos == 1:
                return -1, -1

            higher_point = np.linspace(start = 0, stop = -0.5, num = budget_A, endpoint = False)
            higher_point = higher_point[len(higher_point) - (action_A - action_B)]
            return np.random.uniform(low = -0.5, high = higher_point, size = 1)[0], 0.5 * (-0.5 + higher_point)
        
        else:
            if action_A == 0 and action_B == 0:
                return 0, 0
            else:
                higher_point = np.linspace(start = 0, stop = -0.5, num = budget_A, endpoint = False)
                higher_point = higher_point[len(higher_point) - action_A]
                lower_point = np.linspace(start = 0, stop = 0.5, num = budget_B, endpoint = False)
                lower_point = lower_point[len(lower_point) - action_B]
                return np.random.uniform(low = higher_point, high = lower_point, size = 1)[0], 0.5 * (higher_point + lower_point)

    def get_expected_reward(self):
        print("Finding expected reward function...")
        result = np.zeros((self.token_space + 2, self.budget + 1, self.budget + 1, self.budget + 1, self.budget + 1))
        with np.nditer(result, flags = ["multi_index"], op_flags = ["readwrite"]) as it:
            for x in tqdm.tqdm(it, total = result.size):
                action_space = Alesia.get_action_space(it.multi_index[1], it.multi_index[2])
                action_A = it.multi_index[3]
                action_B = it.multi_index[4]
                if not action_A in action_space[0] :
                    action_A = None
                if not action_B in action_space[1]:
                    action_B = None 
                _, expected_reward = Alesia.get_reward(it.multi_index[0], it.multi_index[1], it.multi_index[2], action_A, action_B, self.token_space)
                x[...] = expected_reward
        return result

    @staticmethod
    def get_state_transition(token_space, total_budget, token_pos, budget_A, budget_B, action_A, action_B):
        # Assume that action_A and action_B are valid, that is curr_state[1] - action_A >= 0 and curr_state[2] - action_B >= 0
        # Also, self.terminate is False
        # Returns a 3d transition matrix, (token_pos, budget_A, budget_B)
        result = np.zeros((token_space + 2, total_budget + 1, total_budget + 1))
        if action_A == action_B:
            result[token_pos, budget_A - action_A, budget_B - action_B] = 1
        elif action_A > action_B:
            result[token_pos - 1, budget_A - action_A, budget_B - action_B] = 1
        else:
            result[token_pos + 1, budget_A - action_A, budget_B - action_B] = 1
        return result
    
    def get_state_transition_dist(self):
        ## 8 dimension array in order of (from_token_pos, from_budget_A, from_budget_B, to_token_pos, to_budget_A, to_budget_B, action_A, action_B)
        print("Retrieving actual transitional distribution...")
        result = np.zeros((self.token_space + 2, self.budget + 1, self.budget + 1, self.token_space + 2, self.budget + 1, self.budget + 1, self.budget + 1, self.budget + 1))
        from_state_action = result[:, :, :, 0, 0, 0, :, :]
        with np.nditer(from_state_action, flags = ["multi_index"], op_flags = ["readwrite"]) as it:
            for _ in tqdm.tqdm(it, total = from_state_action.size):
                action_space = Alesia.get_action_space(it.multi_index[1], it.multi_index[2])
                action_A = it.multi_index[3]
                action_B = it.multi_index[4]
                if not action_A in action_space[0] :
                    action_A = None
                if not action_B in action_space[1]:
                    action_B = None
                done = Alesia.check_termination(it.multi_index[0], self.token_space, it.multi_index[1], it.multi_index[2])
                if not done:
                    if action_A is None or action_B is None:
                        result[it.multi_index[0], it.multi_index[1], it.multi_index[2], it.multi_index[0], it.multi_index[1], it.multi_index[2], it.multi_index[3], it.multi_index[4]] = 1
                    else:
                        result[it.multi_index[0], it.multi_index[1], it.multi_index[2], :, :, :, it.multi_index[3], it.multi_index[4]] = Alesia.get_state_transition(self.token_space, self.budget, it.multi_index[0], it.multi_index[1], it.multi_index[2], action_A, action_B)
                else:
                    result[it.multi_index[0], it.multi_index[1], it.multi_index[2], it.multi_index[0], it.multi_index[1], it.multi_index[2], it.multi_index[3], it.multi_index[4]] = 1
        return result 

    @staticmethod
    def get_action_space(budget_A, budget_B):
        action_space_A = list(range(1, budget_A + 1))
        if budget_A == 0:
            action_space_A = [0]
        action_space_B = list(range(1, budget_B + 1))
        if budget_B == 0:
            action_space_B = [0]
        return [action_space_A, action_space_B]  

    @staticmethod
    def get_token_pos_space(token_space):
        return list(range(0, token_space + 2)) 

    @staticmethod
    def state_sampler(num_samples, token_space, total_budget, transition_distribution, replace = False):
        token_pos_space = Alesia.get_token_pos_space(token_space)
        budget_A_space = list(range(0, total_budget + 1))
        budget_B_space = list(range(0, total_budget + 1))

        idx = np.arange(len(token_pos_space) * len(budget_A_space) * len(budget_B_space))
        sampled_state_idx = np.random.choice(idx, size = num_samples, replace = replace, p = np.reshape(transition_distribution, -1))
        
        idx = np.reshape(idx, (len(token_pos_space), len(budget_A_space), len(budget_B_space)))
        result = []
        for i in sampled_state_idx:
            coordinate = np.where(idx == i)
            sampled_token_pos = token_pos_space[coordinate[0][0]]
            sampled_budget_A = budget_A_space[coordinate[1][0]]
            sampled_budget_B = budget_B_space[coordinate[2][0]]
            result.append((sampled_token_pos, sampled_budget_A, sampled_budget_B))
        return result

    def step(self, action_A, action_B):
        if self.state is None:
            raise Exception('step() used before calling reset()')
        action_space = Alesia.get_action_space(self.state[1], self.state[2])
        if not action_A in action_space[0] :
            action_A = 0
        if not action_B in action_space[1]:
            action_B = 0

        done = Alesia.check_termination(self.state[0], self.token_space, self.state[1], self.state[2])
        self.terminate = done

        reward, _ = Alesia.get_reward(self.state[0], self.state[1], self.state[2], action_A, action_B, self.token_space)

        if not done:
            token_budget_state_space = Alesia.get_state_transition(self.token_space, self.budget, self.state[0], self.state[1], self.state[2], action_A, action_B)
            self.state = Alesia.state_sampler(1, self.token_space, self.budget, token_budget_state_space)[0]

        self.t = self.t + 1
        return done, reward, self.state, {}

    def sample_from_env(self, num_samples):

        print("Sampling from enviornment...")

        # This sampling method is not based on trajectory
        sampled_data = pd.DataFrame(columns=["from_token_pos", "from_budget_A", "from_budget_B", "action_A", "action_B", "to_token_pos", "to_budget_A", "to_budget_B", "reward"])
        
        ## Independently sample states and bugets, also uniformly, can be changed
        sampled_token_pos = np.random.randint(low = 1, high = self.token_space + 1, size = num_samples)
        sampled_budget_A = np.random.randint(low = 1, high = self.budget + 1, size = num_samples)
        sampled_budget_B = np.random.randint(low = 1, high = self.budget + 1, size = num_samples)
        
        for i in tqdm.tqdm(range(0, num_samples)):
            curr_sampled_token_pos = sampled_token_pos[i]
            curr_sampled_budget_A = sampled_budget_A[i]
            curr_sampled_budget_B = sampled_budget_B[i]
            action_space = Alesia.get_action_space(curr_sampled_budget_A, curr_sampled_budget_B)
            curr_sampled_action_A = np.random.choice(action_space[0], size = 1)[0]
            curr_sampled_action_B = np.random.choice(action_space[1], size = 1)[0]
            token_budget_state_space = Alesia.get_state_transition(self.token_space, self.budget, curr_sampled_token_pos, curr_sampled_budget_A, curr_sampled_budget_B, curr_sampled_action_A, curr_sampled_action_B)
            curr_sampled_to_state = Alesia.state_sampler(1, self.token_space, self.budget, token_budget_state_space)[0]
            curr_sampled_reward, _ = Alesia.get_reward(curr_sampled_token_pos, curr_sampled_budget_A, curr_sampled_budget_B, curr_sampled_action_A, curr_sampled_action_B, self.token_space)
            sample = pd.DataFrame([[curr_sampled_token_pos, curr_sampled_budget_A, curr_sampled_budget_B, curr_sampled_action_A, curr_sampled_action_B, curr_sampled_to_state[0], curr_sampled_to_state[1], curr_sampled_to_state[2], curr_sampled_reward]], columns = ["from_token_pos", "from_budget_A", "from_budget_B", "action_A", "action_B", "to_token_pos", "to_budget_A", "to_budget_B", "reward"])
            sampled_data = pd.concat([sampled_data, sample], axis = 0)
        return sampled_data        


class Agent():


    def __init__(self, game_env, gamma, num_iter_k, num_sample_n, num_iter_q, initial_q = None, initial_policy_A=None, estimate_prob_transition = "logistic", initial_w = None):
        self.game_env = game_env
        self.gamma = gamma
        self.num_iter_k = num_iter_k
        self.num_sample_n = num_sample_n
        self.num_iter_q = num_iter_q
        
        ## 5 dimension array in order of (token_pos, budget_A, budget_B, action_A, action_B)
        self.estimated_reward_function = None
        ## 8 dimension array in order of (from_token_pos, from_budget_A, from_budget_B, to_token_pos, to_budget_A, to_budget_B, action_A, action_B)
        self.estimated_transition_function = None
        ## 4 dimension array in order of (action_A, from_token_pos, from_budget_A, from_budget_B)
        if initial_policy_A is None:
            print("Initializting policies of A, B...")
            self.policy_A = np.zeros((game_env.budget + 1, game_env.token_space + 2, game_env.budget + 1, game_env.budget + 1))
            from_budget_idx = np.arange((game_env.budget + 1) * (game_env.budget + 1)).reshape((game_env.budget + 1, game_env.budget + 1))
            with np.nditer(from_budget_idx, flags = ["multi_index"], op_flags = ["readwrite"]) as it:
                for _ in tqdm.tqdm(it, total = from_budget_idx.size):
                    curr_budget_A = it.multi_index[0]
                    curr_budget_B = it.multi_index[1]
                    action = Alesia.get_action_space(curr_budget_A, curr_budget_B)
                    self.policy_A[action[0][0], :, curr_budget_A, curr_budget_B] = 1
        else:
            self.policy_A = initial_policy_A

        ## 6 dimension array in order of (action_B, from_token_pos, from_budget_A, from_budget_B)
        self.policy_B = self.policy_A
        self.initial_policy_A = self.policy_A
        self.initial_policy_B = self.policy_B
        self.optimal_policy_A = self.policy_A
        self.optimal_policy_B = self.policy_B

        ## 2 dimension arrau in order of (token_pos * budget_A * budget_B = dimension_of_to_states, #(token_pos, budget_A, budget_B, action_A, action_B) = 5)
        if estimate_prob_transition == "value" and initial_w is None:
            print("Initializing weight w...")            
            from_token_pos = np.arange(game_env.token_space + 2)
            from_budget_A = np.arange(game_env.budget + 1)
            from_budget_B = np.arange(game_env.budget + 1)
            action_A = np.arange(game_env.budget + 1)
            action_B = np.arange(game_env.budget + 1)
            synthetic_from_data = pd.MultiIndex.from_product([from_token_pos, from_budget_A, from_budget_B, action_A, action_B], names = ["from_token_pos", "from_budget_A", "from_budget_B", "action_A", "action_B"]).to_frame()
            synthetic_to_data = np.repeat(np.arange((game_env.token_space + 2) * (game_env.budget + 1) * (game_env.budget + 1)), (game_env.budget + 1) * (game_env.budget + 1))

            trained_model = linear_model.LogisticRegression(n_jobs = 11, multi_class = "multinomial", fit_intercept = False, solver = "lbfgs", max_iter = 5000).fit(synthetic_from_data, synthetic_to_data)
            self.w = trained_model.coef_
        else:
            self.w = initial_w
        self.estimate_prob_transition = estimate_prob_transition

        ## 5 dimension array in order of (token_pos, budget_A, budget_B, action_A, action_B)
        if initial_q is None:
            self.q_function = np.zeros((game_env.token_space + 2, game_env.budget + 1, game_env.budget + 1, game_env.budget + 1, game_env.budget + 1))
            self.q_function[0, ...] = -1
            self.q_function[-1, ...] = 1
        else:
            self.q_function = initial_q
        self.optimal_q_function = self.q_function
        ## 3 dimension array in order of (token_pos, budget_A, budget_B)
        self.value_function = Agent.estimate_value_function_from_q_function(self.q_function, self.policy_A, self.policy_B)
        self.optimal_value_function = Agent.estimate_value_function_from_q_function(self.optimal_q_function, self.optimal_policy_A, self.optimal_policy_B)


    @staticmethod
    def estimate_reward_distribution(training_set, game_env):
        print("Estimating reward distribution...")

        state_action_pair = training_set[["from_token_pos", "from_budget_A", "from_budget_B", "action_A", "action_B"]]
        reward = training_set['reward']
        trained_model = linear_model.LinearRegression().fit(X = state_action_pair, y = reward)
        
        estimated_reward_function = np.zeros((game_env.token_space + 2, game_env.budget + 1, game_env.budget + 1, game_env.budget + 1, game_env.budget + 1))
        
        with np.nditer(estimated_reward_function, flags = ["multi_index"], op_flags = ["readwrite"]) as it:
            for x in tqdm.tqdm(it, total = estimated_reward_function.size):
                regressor = pd.DataFrame([list(it.multi_index)], columns = state_action_pair.columns)
                x[...] = trained_model.predict(regressor)[0]

        return estimated_reward_function


    @staticmethod
    def estimate_transition_distribution_value(training_set, estimated_value_function, game_env, prev_w):   
        ## estimated_value_function is (token_pos * budget_A * budget_B)

        print("Estimating transition value distribution...")
        from_state_action_pair = training_set[["from_token_pos", "from_budget_A", "from_budget_B", "action_A", "action_B"]]
        to_states = training_set[["to_token_pos", "to_budget_A", "to_budget_B"]]

        def cost_func(w):
            
            ## We need to flatten everything to speed things up
            ## w is (token_pos * budget_A * budget_B = dimension_of_to_states, #(token_pos, budget_A, budget_B, action_A, action_B) = 5)
            w = w.reshape(((game_env.token_space + 2) * (game_env.budget + 1) * (game_env.budget + 1), 5))
                        
            total_costs = 0
            for i in range(0, len(to_states.index)):
                state_action_pair = from_state_action_pair.iloc[i].to_numpy()
                to_state = to_states.iloc[i].to_numpy()
                all_state_linear_predictor = np.apply_along_axis(lambda x: np.dot(x, state_action_pair), 1, w)
                all_state_prob = np.exp(all_state_linear_predictor - sp.special.logsumexp(all_state_linear_predictor))
                expected_value_function = np.dot(all_state_prob, estimated_value_function.flatten())
                actual_value_function = estimated_value_function[to_state[0], to_state[1], to_state[2]]
                total_costs += (expected_value_function - actual_value_function) ** 2
            total_costs = total_costs / len(to_states.index)
            return total_costs

        def cost_func_jac(w):
            ## We need to flatten everything to speed things up
            ## w is (token_pos * budget_A * budget_B = dimension_of_to_states, #(token_pos, budget_A, budget_B, action_A, action_B) = 5)
            w = w.reshape(((game_env.token_space + 2) * (game_env.budget + 1) * (game_env.budget + 1), 5))
            
            change = np.zeros(((game_env.token_space + 2) * (game_env.budget + 1) * (game_env.budget + 1), 5))
            for i in range(0, len(to_states.index)):
                state_action_pair = from_state_action_pair.iloc[i].to_numpy()
                to_state = to_states.iloc[i].to_numpy()
                to_state_idx = np.arange((game_env.token_space + 2) * (game_env.budget + 1) * (game_env.budget + 1)).reshape((game_env.token_space + 2, game_env.budget + 1, game_env.budget + 1))[to_state[0], to_state[1], to_state[2]]

                all_state_linear_predictor = np.apply_along_axis(lambda x: np.dot(x, state_action_pair), 1, w).flatten()
                
                ## Vector of shape (token_pos * budget_A * budget_B,)
                all_state_prob = np.exp(all_state_linear_predictor - sp.special.logsumexp(all_state_linear_predictor))
                
                ## Number
                expected_value_function = np.dot(all_state_prob, estimated_value_function.flatten())
                ## Number
                actual_value_function = estimated_value_function[to_state[0], to_state[1], to_state[2]]

                cov_estimated_actual = np.sum(np.dot(np.multiply(all_state_prob, estimated_value_function.flatten()).reshape(-1, 1), state_action_pair.reshape(1, -1)), axis = 0) - expected_value_function * np.sum(np.dot(all_state_prob.reshape(-1, 1), state_action_pair.reshape(1, -1)), axis = 0)
                change[to_state_idx, :] = change[to_state_idx, :] + (expected_value_function - actual_value_function) * cov_estimated_actual
            change = 2 * change / len(to_states.index)
            return change.flatten()

        optimization_result = sp.optimize.minimize(cost_func, prev_w.flatten(), method = "CG", jac = cost_func_jac)
        new_w = optimization_result.x.reshape(((game_env.token_space + 2) * (game_env.budget + 1) * (game_env.budget + 1), 5))

        ## 8 dimension array in order of (from_token_pos, from_budget_A, from_budget_B, to_token_pos, to_budget_A, to_budget_B, action_A, action_B)
        estimated_transition_dist = np.zeros((game_env.token_space + 2, game_env.budget + 1, game_env.budget + 1, game_env.token_space + 2, game_env.budget + 1, game_env.budget + 1, game_env.budget + 1, game_env.budget + 1))

        from_state_action_idx = estimated_transition_dist[:, :, :, 0, 0, 0, :, :]
        it = np.nditer(from_state_action_idx, flags = ["multi_index"], op_flags = ["readwrite"])
        for _ in tqdm.tqdm(it, total = from_state_action_idx.size):
            state_action_pair = np.array(list(it.multi_index))
            all_state_linear_predictor = np.apply_along_axis(lambda x: np.dot(x, state_action_pair), 1, new_w).flatten()
            all_state_prob = np.exp(all_state_linear_predictor - sp.special.logsumexp(all_state_linear_predictor))
            estimated_transition_dist[it.multi_index[0], it.multi_index[1], it.multi_index[2], :, :, :, it.multi_index[3], it.multi_index[4]] = all_state_prob.reshape((game_env.token_space + 2, game_env.budget + 1, game_env.budget + 1))
        
        return estimated_transition_dist, new_w


    @staticmethod
    def estimate_transition_distribution_logistic(training_set, estimated_value_function, game_env):   
        print("Estimating transition distribution...")
        from_state_action_pair = training_set[["from_token_pos", "from_budget_A", "from_budget_B", "action_A", "action_B"]]
        to_states = training_set[["to_token_pos", "to_budget_A", "to_budget_B"]]

        ## Here we use multionmial logistic to model the transition probability, this method does not consider difference in value function.
        
        state_index_dict = np.arange((game_env.token_space + 2) * (game_env.budget + 1) * (game_env.budget + 1)).reshape((game_env.token_space + 2, game_env.budget + 1, game_env.budget + 1))
        to_states_index = np.zeros(len(to_states.index))
        
        for i in range(0, len(to_states.index)):
            to_states_index[i] = state_index_dict[to_states.iloc[i, 0], to_states.iloc[i, 1], to_states.iloc[i, 2]]
        to_states_index = pd.Series(to_states_index)

        trained_model = linear_model.LogisticRegression(multi_class = "multinomial", solver = "lbfgs", max_iter = 5000).fit(from_state_action_pair, to_states_index)

        ## 8 dimension array in order of (from_token_pos, from_budget_A, from_budget_B, to_token_pos, to_budget_A, to_budget_B, action_A, action_B)
        estimated_transition_dist = np.zeros((game_env.token_space + 2, game_env.budget + 1, game_env.budget + 1, game_env.token_space + 2, game_env.budget + 1, game_env.budget + 1, game_env.budget + 1, game_env.budget + 1))

        from_state_action_idx = estimated_transition_dist[:, :, :, 0, 0, 0, :, :]
        it = np.nditer(from_state_action_idx, flags = ["multi_index"], op_flags = ["readwrite"])
        predicted_classes = [int(x) for x in trained_model.classes_]
        for _ in tqdm.tqdm(it, total = from_state_action_idx.size):
            regressor = pd.DataFrame([list(it.multi_index)], columns = from_state_action_pair.columns)
            predicted_probs = trained_model.predict_proba(regressor)
            to_probs = np.zeros((game_env.token_space + 2) * (game_env.budget + 1) * (game_env.budget + 1))
            to_probs[predicted_classes] = predicted_probs
            to_probs = to_probs.reshape((game_env.token_space + 2, game_env.budget + 1, game_env.budget + 1))
            estimated_transition_dist[it.multi_index[0], it.multi_index[1], it.multi_index[2], :, :, :, it.multi_index[3], it.multi_index[4]] = to_probs
        return estimated_transition_dist

    
    @staticmethod
    def estimate_q_function(q_0, estimated_reward_function, estimated_transition_function, policy_A, num_iter_q, gamma):
        print("Estimating Q function...")
        curr_q = q_0
        curr_iter = 1

        pbar = tqdm.tqdm(total = num_iter_q, initial = 1)
        while curr_iter < num_iter_q:
            ##  Q function :(token_pos, budget_A, budget_B, action_A, action_B)
            ##  policy A : (action_A, token_pos, budget_A, budget_B)
            ## mod_policy_A : (token_pos, budget_A, budget_B, action_A, 1)
            mod_policy_A = np.moveaxis(policy_A, 0, -1)[..., np.newaxis]
            ## q_policy_B : (token_pos, budget_A, budget_B, action_B)
            q_policy_B = np.sum(np.multiply(curr_q, mod_policy_A), axis = 3)
            ## value_func : (token_pos, budget_A, budget_B), action_B maximizes value
            value_func = np.max(q_policy_B, axis = 3)

            ## Update q_func, apply bellman equation
            ## estimated_reward_function (token_pos, budget_A, budget_B, action_A, action_B)
            ## estimated_transition_function (from_token_pos, from_budget_A, from_budget_B, to_token_pos, to_budget_A, to_budget_B, action_A, action_B)
            mod_value_func = value_func[:, :, :, np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
            new_q = estimated_reward_function + gamma * np.sum(np.multiply(estimated_transition_function, mod_value_func), axis = (3, 4, 5))

            curr_q = new_q
            curr_iter = curr_iter + 1
            pbar.update(1)
        pbar.close()            
            
        return curr_q


    @staticmethod
    def estimate_value_function_from_q_function(q_function, policy_A, policy_B):
        ## q_function  : (token_pos, budget_A, budget_B, action_A, action_B)
        ## policy_A : (action_A, token_pos, budget_A, budget_B)
        ## policy_B : (action_B, token_pos, budget_A, budget_B)
        ## value_function : (token_pos, budget_A, budget_B)

        ## mod_policy_A : (token_pos, budget_A, budget_B, action_A, 1)
        mod_policy_A = np.moveaxis(policy_A, 0, -1)[..., np.newaxis]
        ## q_policy_B : (token_pos, budget_A, budget_B, action_B)
        q_policy_B = np.sum(np.multiply(q_function, mod_policy_A), axis = 3)
        ## mod_policy_B : (token_pos, budget_A, budget_B, action_B)
        mod_policy_B = np.moveaxis(policy_B, 0, -1)
        value_function = np.sum(np.multiply(q_policy_B, mod_policy_B), axis = 3)
        return value_function


    @staticmethod
    def find_optimal_policies(estimated_q_function, game_env, num_cores = 11):
        print("Finding min-max equilibrium policy...")
        ## Q function :(token_pos, budget_A, budget_B, action_A, action_B)
        ## Policy A is 4 dimension (action_A, from_token_pos, from_budget_A, from_budget_B), policy A minimizes target
        ## Policy B is 4 dimension (action_B, from_token_pos, from_budget_A, from_budget_B), policy B maximizes target


        policy_A = np.zeros((game_env.budget + 1) * (game_env.token_space + 2) * (game_env.budget + 1) * (game_env.budget + 1)).reshape((game_env.budget + 1, game_env.token_space + 2, game_env.budget + 1, game_env.budget + 1))
        policy_B = np.zeros((game_env.budget + 1) * (game_env.token_space + 2) * (game_env.budget + 1) * (game_env.budget + 1)).reshape((game_env.budget + 1, game_env.token_space + 2, game_env.budget + 1, game_env.budget + 1))
        
        state_space = np.arange(game_env.token_space + 2)
        action_space_A = np.arange(game_env.budget + 1)
        action_space_B = np.arange(game_env.budget + 1)

        from_states = pd.MultiIndex.from_product([state_space, action_space_A, action_space_B], names=["token_pos", "budget_A", "budget_B"]).to_frame()

        with mp.Pool(processes = num_cores) as p:
            result = list(tqdm.tqdm(p.imap(functools.partial(find_nash_equilibrium, from_states, estimated_q_function), range(len(from_states.index))), total = len(from_states.index)))

        for i in range(len(from_states.index)):
            token_pos = from_states.iloc[i, 0]
            budget_A = from_states.iloc[i, 1]
            budget_B = from_states.iloc[i, 2]
            policy_A[:, token_pos, budget_A, budget_B] = result[i][0]
            policy_B[:, token_pos, budget_A, budget_B] = result[i][1]
        return policy_A, policy_B


    @staticmethod
    def sample_from_policy(policy_A, policy_B, game_env, max_try=10):
        curr_token_pos = game_env.state[0]
        curr_budget_A = game_env.state[1]
        curr_budget_B = game_env.state[2]
        curr_action_space = Alesia.get_action_space(curr_budget_A, curr_budget_B)
        
        sample_action_A_success = False
        sample_action_B_success = False

        curr_sampled_action_A = 0
        try_time = 0
        while (not sample_action_A_success) and (try_time < max_try):
            curr_sampled_action_A = np.random.choice(np.arange(game_env.budget + 1), size = 1, p = policy_A[:, curr_token_pos, curr_budget_A, curr_budget_B])[0]
            if curr_sampled_action_A in curr_action_space[0]:
                sample_action_A_success = True
            try_time += 1   
        if not sample_action_A_success:
            curr_sampled_action_A = np.random.choice(curr_action_space[0], 1)[0]             
        
        curr_sampled_action_B = 0
        try_time = 0
        while (not sample_action_B_success) and (try_time < max_try):
            curr_sampled_action_B = np.random.choice(np.arange(game_env.budget + 1), size = 1, p = policy_B[:, curr_token_pos, curr_budget_A, curr_budget_B])[0]
            if curr_sampled_action_B in curr_action_space[1]:
                sample_action_B_success = True
            try_time += 1   
        if not sample_action_B_success:
            curr_sampled_action_B = np.random.choice(curr_action_space[1], 1)[0] 
        return curr_sampled_action_A, curr_sampled_action_B
        

    def make_action(self):
        ## This is the main function
        recorded_q_functions = []
        recorded_policy_A = []
        recorded_policy_B = []
        recorded_estimated_transition_matrix = []
        recorded_estimated_reward_function = []

        for k in range(0, self.num_iter_k):
            print("Number of iteration: " + str(k))

            ## Draw samples by interacting with the environment
            training_set = self.game_env.sample_from_env(self.num_sample_n)

            ## Estimate reward function
            self.estimated_reward_function = Agent.estimate_reward_distribution(training_set, self.game_env)
            recorded_estimated_reward_function.append(self.estimated_reward_function)
            ## Estimate transition probability
            if self.estimate_prob_transition == "logistic":
                self.estimated_transition_function = Agent.estimate_transition_distribution_logistic(training_set, self.value_function, self.game_env)
            else:
                self.estimated_transition_function, self.w = Agent.estimate_transition_distribution_value(training_set, self.value_function, self.game_env, self.w)
            recorded_estimated_transition_matrix.append(self.estimated_transition_function)

            ## Estimate Value function and Q function
            self.q_function = Agent.estimate_q_function(self.q_function, self.estimated_reward_function, self.estimated_transition_function, self.policy_A, self.num_iter_q, self.gamma)
            recorded_q_functions.append(self.q_function)
            self.optimal_q_function = Agent.estimate_q_function(self.optimal_q_function, self.game_env.expected_reward, self.game_env.state_transition_dist, self.optimal_policy_A, self.num_iter_q, self.gamma)

            self.policy_A, self.policy_B = Agent.find_optimal_policies(self.q_function, self.game_env)
            recorded_policy_A.append(self.policy_A)
            recorded_policy_B.append(self.policy_B)
            self.optimal_policy_A, self.optimal_policy_B = Agent.find_optimal_policies(self.optimal_q_function, self.game_env)

            self.value_function = Agent.estimate_value_function_from_q_function(self.q_function, self.policy_A, self.policy_B)
            self.optimal_value_function = Agent.estimate_value_function_from_q_function(self.optimal_q_function, self.optimal_policy_A, self.optimal_policy_B)

        action_A, action_B = Agent.sample_from_policy(self.policy_A, self.policy_B, self.game_env)

        record = {"timestamp" : self.game_env.t, "recorded_q_functions" : recorded_q_functions, "recorded_policy_A" : recorded_policy_A, "recorded_policy_B" : recorded_policy_B, "recorded_estimated_transition_matrix" : recorded_estimated_transition_matrix, "recorded_estimated_reward_function" : recorded_estimated_reward_function, "state_transition_dist" : self.game_env.state_transition_dist, "initial_state_dist" : self.game_env.initial_state_dist, "initial_policy_A" : self.initial_policy_A, "initial_policy_B": self.initial_policy_B, "optimal_q_function" : self.optimal_q_function}
        with open("recorded_statistics at time " + str(self.game_env.t) + ".pkl", "wb") as f:
            pickle.dump(record, f,  protocol = pickle.HIGHEST_PROTOCOL)
        return action_A, action_B


def apply_bellman_equation(num_iter_q, gamma, estimated_reward_function, estimated_transition_function, estimated_q_function, policy_A, policy_B): 
            curr_q = estimated_q_function
            for _ in range(num_iter_q):
                curr_value_func = Agent.estimate_value_function_from_q_function(curr_q, policy_A, policy_B)
                mod_value_func = curr_value_func[:, :, :, np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
                curr_q = estimated_reward_function + gamma * np.sum(np.multiply(estimated_transition_function, mod_value_func), axis = (3, 4, 5))
            return curr_q


def find_errors(estimated_q_function_k, estimated_q_function_kp1, policy_A_k, policy_B_k, gamma, estimated_transition_function_kp1, transition_distribution, estimated_reward_function_k, num_iter_q):            
        epsilon_k = apply_bellman_equation(num_iter_q, gamma, estimated_reward_function_k, estimated_transition_function_kp1, estimated_q_function_k, policy_A_k, policy_B_k) - estimated_q_function_kp1
        epsilon_k_l2_norm = np.sqrt(np.sum(np.power(epsilon_k, 2)))

        S_k = apply_bellman_equation(num_iter_q, gamma, estimated_reward_function_k, estimated_transition_function_kp1, estimated_q_function_k, policy_A_k, policy_B_k) - apply_bellman_equation(num_iter_q, gamma, estimated_reward_function_k, transition_distribution, estimated_q_function_k, policy_A_k, policy_B_k)
        S_k_l2_norm = np.sqrt(np.sum(np.power(S_k, 2)))

        ## transition matrix: 8 dimension array in order of (from_token_pos, from_budget_A, from_budget_B, to_token_pos, to_budget_A, to_budget_B, action_A, action_B)
        ## value function: 3 dimension array in order of (from_token_pos, from_budget_A, from_budget_B)
        estimated_value_function_k = Agent.estimate_value_function_from_q_function(estimated_q_function_k, policy_A_k, policy_B_k)
        e_k = np.multiply(transition_distribution - estimated_transition_function_kp1, estimated_value_function_k[:, :, :, np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis])
        e_k_l2_norm = np.sqrt(np.sum(np.power(e_k, 2)))
        return epsilon_k_l2_norm, S_k_l2_norm, e_k_l2_norm


def find_error_weighted_norm(estimated_q_function_k, optimal_q_function, initial_state_dist, initial_policy_A, initial_policy_B):
    ## initial_state_dist: (token_pos, budget_A, budget_B)
    ## initial_policy_A: (action_A, token_pos, budget_A, budget_B)
    ## initial_policy_B: (action_B, token_pos, budget_A, budget_B)
    ## initial_state_action_dist : (token_pos, budget_A, budget_B, action_A, action_B)
    initial_state_action_dist = np.zeros(estimated_q_function_k.shape)
    with np.nditer(initial_state_dist, flags = ["multi_index"], op_flags = ["readwrite"]) as it:
        for _ in it:
            curr_token_pos = it.multi_index[0]
            curr_budget_A = it.multi_index[1]
            curr_budget_B = it.multi_index[2]
            cond_policy_A = initial_policy_A[:, curr_token_pos, curr_budget_A, curr_budget_B]
            cond_policy_B = initial_policy_B[:, curr_token_pos, curr_budget_A, curr_budget_B]
            initial_state_action_dist[curr_token_pos, curr_budget_A, curr_budget_B, :, :] = np.outer(cond_policy_A, cond_policy_B)
    
    weighted_l1_diff_q_func = np.sum(np.multiply(np.abs(optimal_q_function - estimated_q_function_k), initial_state_action_dist))
    return weighted_l1_diff_q_func


def find_error_upper_bound(gamma, c_bar, epsilon_k_l2_norms, S_k_l2_norms, num_iter_k, num_iter_q, Rmax):
    upper_bound =  (2 * gamma / ((1 - gamma) ** 2)) * (c_bar * np.max(np.array(epsilon_k_l2_norms) + np.array(S_k_l2_norms)) + 2 * (gamma ** (num_iter_k * num_iter_q)) * Rmax)
    return upper_bound


def collect_statistics(terminate_time, num_iter_k, cbar = 1, Rmax = 1):
    lhs_l1_norm = []
    rhs_upper_bound = []
    epsilon_k_l2_norms = []
    S_k_l2_norms = []
    e_k_l2_norms = []
    for t in range(0, terminate_time):
        with open("recorded_statistics at time " + str(t) + ".pkl", "rb") as f:
            record = pickle.load(f)
        recorded_q_functions = record["recorded_q_functions"]
        recorded_policy_A = record["recorded_policy_A"]
        recorded_policy_B = record["recorded_policy_B"]
        recorded_estimated_transition_matrix = record["recorded_estimated_transition_matrix"]
        recorded_estimated_reward_function = record["recorded_estimated_reward_function"]
        state_transition_dist = record["state_transition_dist"]
        initial_state_dist = record["initial_state_dist"]
        initial_policy_A = record["initial_policy_A"]
        initial_policy_B = record["initial_policy_B"]
        optimal_q_function = record["optimal_q_function"]
        for k in range(0, num_iter_k - 1):
            estimated_q_function_k = recorded_q_functions[k]
            estimated_q_function_kp1 = recorded_q_functions[k + 1]
            policy_A_k = recorded_policy_A[k]
            policy_B_k = recorded_policy_B[k]
            estimated_transition_function_kp1 = recorded_estimated_transition_matrix[k + 1]
            estimated_reward_function_k = recorded_estimated_reward_function[k]
            epsilon_k_l2_norm, S_k_l2_norm, e_k_l2_norm = find_errors(estimated_q_function_k, estimated_q_function_kp1, policy_A_k, policy_B_k, gamma, estimated_transition_function_kp1, state_transition_dist, estimated_reward_function_k, num_iter_q)
            epsilon_k_l2_norms.append(epsilon_k_l2_norm)
            S_k_l2_norms.append(S_k_l2_norm)
            e_k_l2_norms.append(e_k_l2_norm)
            lhs = find_error_weighted_norm(recorded_q_functions[k + 1], optimal_q_function, initial_state_dist, initial_policy_A, initial_policy_B)
            rhs = find_error_upper_bound(gamma, cbar, epsilon_k_l2_norms, S_k_l2_norms, num_iter_k, num_iter_q, Rmax)
            lhs_l1_norm.append(lhs)
            rhs_upper_bound.append(rhs)
    summary_statistics = pd.DataFrame(data = {"t" : np.repeat(np.arange(terminate_time), num_iter_k - 1), "k" : np.tile(np.arange(terminate_time), num_iter_k - 1), "lhs_l1_norm" : lhs_l1_norm, "rhs_upper_bound" : rhs_upper_bound, "epsilon_k_l2_norms" : epsilon_k_l2_norms, "S_k_l2_norms" : S_k_l2_norms, "e_k_l2_norms" : e_k_l2_norms})
    summary_statistics.to_csv("recorded_statistics" + ".csv")
        

def run_experiment(budget, token_space, max_time, gamma, num_iter_k, num_sample_n, num_iter_q, initial_q=None, initial_policy_A=None, estimate_prob_transition="logistic", initial_w=None):
    np.random.seed(100) 

    states = []
    rewards = []
    actions = []

    # define agent and the environment
    env = Alesia(budget, token_space)
    env.reset()

    agent = Agent(env, gamma, num_iter_k, num_sample_n, num_iter_q, initial_q, initial_policy_A, estimate_prob_transition, initial_w)

    terminate = False
    while not terminate:
        action_A, action_B = agent.make_action()

        terminate, reward, state, _ = env.step(action_A, action_B)
        if env.t > max_time:
            terminate = True

        states.append(state)
        rewards.append(reward)
        actions.append((action_A, action_B))

    collect_statistics(min(env.t, max_time), num_iter_k, cbar = 1, Rmax = 1)
    return states, rewards


budget = 6
token_space = 5
gamma = 0.5
num_iter_k = 5
num_sample_n = 7 * 7 * 7 * 20
num_iter_q = 35
initial_q = None
initial_policy_A = None
max_time = 10

estimate_transition_distribution = "logistic"
initial_w = None
states, rewards = run_experiment(budget, token_space, max_time, gamma, num_iter_k, num_sample_n, num_iter_q, initial_q, initial_policy_A, estimate_transition_distribution, initial_w)


#collect_statistics(4, num_iter_k, cbar = 1, Rmax = 1)
