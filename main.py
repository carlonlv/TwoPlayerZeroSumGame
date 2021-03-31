import itertools
import math
import multiprocessing as mp
import time

import matplotlib.pyplot as plt
import nashpy as nash
import numpy as np
import pandas as pd
import scipy as sp
import tqdm
from scipy.stats import entropy
from sklearn import linear_model


class Alesia():
    def __init__(self, budget, token_space):
        self.budget = budget
        self.token_space = token_space
        self.terminate = False
        
        # A tuple (token_pos, budget_A, budget_B)
        # token_pos can go from 0 to token_space + 1. 
        # budget_A and budget_B can go from 0 to budget
        self.state = None

        self.t = 0

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
        if action_B > action_A:
            if token_pos == token_space:
                return 1

            lower_point = np.linspace(start = 0, stop = 0.5, num = budget_B, endpoint = False)
            if action_A is None:
                lower_point = lower_point[len(lower_point) - action_B]
            else:
                lower_point = lower_point[len(lower_point) - (action_B - action_A)]
            return np.random.uniform(low = lower_point, high = 0.5, size = 1)[0]
            
        elif action_A > action_B:
            if token_pos == 1:
                return -1

            higher_point = np.linspace(start = 0, stop = -0.5, num = budget_A, endpoint = False)
            if action_B is None:
                higher_point = higher_point[len(higher_point) - action_A]
            else:
                higher_point = higher_point[len(higher_point) - (action_A - action_B)]
            return np.random.uniform(low = -0.5, high = higher_point, size = 1)[0]
        
        else:
            higher_point = np.linspace(start = 0, stop = -0.5, num = budget_A, endpoint = False)
            higher_point = higher_point[len(higher_point) - action_A]
            lower_point = np.linspace(start = 0, stop = 0.5, num = budget_B, endpoint = False)
            lower_point = lower_point[len(lower_point) - action_B]
            return np.random.uniform(low = higher_point, high = lower_point, size = 1)[0]

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
            action_A = None
        if not action_B in action_space[1]:
            action_B = None

        done = Alesia.check_termination(self.state[0], self.token_space, self.state[1], self.state[2])
        self.terminate = done

        reward = Alesia.get_reward(self.state[0], self.state[1], self.state[2], action_A, action_B, self.token_space)

        if not done:
            token_budget_state_space = Alesia.get_state_transition(self.token_space, self.budget, self.state[0], self.state[1], self.state[2], action_A, action_B)
            self.state = Alesia.state_sampler(1, self.token_space, self.budget, token_budget_state_space)

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
            curr_sampled_reward = Alesia.get_reward(curr_sampled_token_pos, curr_sampled_budget_A, curr_sampled_budget_B, curr_sampled_action_A, curr_sampled_action_B, self.token_space)
            sample = pd.DataFrame([[curr_sampled_token_pos, curr_sampled_budget_A, curr_sampled_budget_B, curr_sampled_action_A, curr_sampled_action_B, curr_sampled_to_state[0], curr_sampled_to_state[1], curr_sampled_to_state[2], curr_sampled_reward]], columns = ["from_token_pos", "from_budget_A", "from_budget_B", "action_A", "action_B", "to_token_pos", "to_budget_A", "to_budget_B", "reward"])
            sampled_data = pd.concat([sampled_data, sample], axis = 0)
        return sampled_data        


class Agent():


    def __init__(self, game_env, gamma, num_iter_k, num_sample_n, num_iter_q, initial_q, initial_policy_A=None, estimate_prob_transition = "logistic", initial_w = None):
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
            self.policy_A = np.zeros((game_env.budget + 1, game_env.token_space + 2, game_env.budget + 1, game_env.budget + 1))
            from_budget_idx = np.arange((game_env.budget + 1) * (game_env.budget + 1)).reshape((game_env.budget + 1, game_env.budget + 1))
            with np.nditer(from_budget_idx, flags = ["multi_index"], op_flags = ["readwrite"]) as it:
                for _ in it:
                    curr_budget_A = it.multi_index[0]
                    curr_budget_B = it.multi_index[1]
                    action = Alesia.get_action_space(curr_budget_A, curr_budget_B)
                    self.policy_A[action[0][0], :, curr_budget_A, curr_budget_B] = 1
        else:
            self.policy_A = initial_policy_A

        ## 2 dimension arrau in order of (token_pos * budget_A * budget_B = dimension_of_to_states, #(token_pos, budget_A, budget_B, action_A, action_B) = 5)
        if estimate_prob_transition == "value" and initial_w is None:
            initial_w = np.full(((game_env.token_space + 2) * (game_env.budget + 1) * (game_env.budget + 1), 5), -np.Inf)
        else:
            self.w = initial_w

        ## 6 dimension array in order of (action_B, from_token_pos, from_budget_A, from_budget_B)
        self.policy_B = None
        ## 5 dimension array in order of (token_pos, budget_A, budget_B, action_A, action_B)
        self.q_function = initial_q
        ## 3 dimension array in order of (token_pos, budget_A, budget_B)
        self.value_function = None


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
                state_action_pair = from_state_action_pair.iloc[i]
                to_state = to_states.iloc[i]
                all_state_linear_predictor = np.apply_over_axes(lambda x: np.dot(x, state_action_pair), w, list(w.shape)[1:]).flatten()
                all_state_prob = np.exp(all_state_linear_predictor - sp.special.logsumexp(all_state_prob))
                expected_value_function = np.dot(all_state_prob, estimated_value_function.flatten())
                actual_value_function = estimated_value_function[to_state]
                total_costs += (expected_value_function - actual_value_function) ** 2
            total_costs = total_costs / len(to_states.index)
            return total_costs

        def cost_func_jac(w):
            ## We need to flatten everything to speed things up
            ## w is (token_pos * budget_A * budget_B = dimension_of_to_states, #(token_pos, budget_A, budget_B, action_A, action_B) = 5)
            w = w.reshape(((game_env.token_space + 2) * (game_env.budget + 1) * (game_env.budget + 1), 5))
            
            change = np.zeros(((game_env.token_space + 2) * (game_env.budget + 1) * (game_env.budget + 1), 5))
            for i in range(0, len(to_states.index)):
                state_action_pair = from_state_action_pair.iloc[i]
                to_state = to_states.iloc[i]
                to_state_idx = np.arange((game_env.token_space + 2) * (game_env.budget + 1) * (game_env.budget + 1)).reshape((game_env.token_space + 2, game_env.budget + 1, game_env.budget + 1))[to_state]

                all_state_linear_predictor = np.apply_over_axes(lambda x: np.dot(x, state_action_pair), w, list(w.shape)[1:]).flatten()
                
                ## Vector of shape (token_pos * budget_A * budget_B,)
                all_state_prob = np.exp(all_state_linear_predictor - sp.special.logsumexp(all_state_prob))
                
                ## Number
                expected_value_function = np.dot(all_state_prob, estimated_value_function.flatten())
                ## Number
                actual_value_function = estimated_value_function[to_state]

                cov_estimated_actual = np.sum(np.dot(np.multiply(all_state_prob, estimated_value_function.flatten()).reshape(-1, 1), state_action_pair.reshape(1, -1)), axis = 0) - expected_value_function * np.sum(np.dot(all_state_prob, state_action_pair.reshape(1, -1)), axis = 0)
                change[to_state_idx, ...] += (expected_value_function - actual_value_function) * cov_estimated_actual
            return change

        new_w = sp.optimize.minimize(cost_func, prev_w, method = "BFGS", jac = cost_func_jac).x
        

        ## 8 dimension array in order of (from_token_pos, from_budget_A, from_budget_B, to_token_pos, to_budget_A, to_budget_B, action_A, action_B)
        estimated_transition_dist = np.zeros((game_env.token_space + 2, game_env.budget + 1, game_env.budget + 1, game_env.token_space + 2, game_env.budget + 1, game_env.budget + 1, game_env.budget + 1, game_env.budget + 1))

        from_state_action_idx = estimated_transition_dist[:, :, :, 0, 0, 0, :, :]
        it = np.nditer(from_state_action_idx, flags = ["multi_index"], op_flags = ["readwrite"])
        for _ in tqdm.tqdm(it, total = from_state_action_idx.size):
            state_action_pair = np.array(list(it.multi_index))
            all_state_linear_predictor = np.apply_over_axes(lambda x: np.dot(x, state_action_pair), new_w, list(new_w.shape)[1:]).flatten()
            all_state_prob = np.exp(all_state_linear_predictor - sp.special.logsumexp(all_state_prob))
            estimated_transition_dist[it.multi_index[0], it.multi_index[1], it.multi_index[2], :, :, :, it.multi_index[3], it.multi_index[4]] = all_state_prob
        
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
        l2_change = np.Inf

        tolerance = 0.001
        curr_iter = 1

        pbar = tqdm.tqdm(total = num_iter_q)
        while (l2_change > tolerance) or (curr_iter < num_iter_q):
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

            l2_change = np.linalg.norm(new_q - curr_q)
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
    def find_optimal_policies(estimated_q_function, game_env):
        print("Finding min-max equilibrium policy...")
        ## Q function :(token_pos, budget_A, budget_B, action_A, action_B)
        ## Policy A is 4 dimension (action_A, from_token_pos, from_budget_A, from_budget_B), policy A minimizes target
        ## Policy B is 4 dimension (action_B, from_token_pos, from_budget_A, from_budget_B), policy B maximizes target

        policy_A = np.zeros((game_env.budget + 1) * (game_env.token_space + 2) * (game_env.budget + 1) * (game_env.budget + 1)).reshape((game_env.budget + 1, game_env.token_space + 2, game_env.budget + 1, game_env.budget + 1))
        policy_B = np.zeros((game_env.budget + 1) * (game_env.token_space + 2) * (game_env.budget + 1) * (game_env.budget + 1)).reshape((game_env.budget + 1, game_env.token_space + 2, game_env.budget + 1, game_env.budget + 1))
        
        from_state_idx = policy_A[0, ...]
        it = np.nditer(from_state_idx, flags = ["multi_index"], op_flags = ["readwrite"])
        for _ in tqdm.tqdm(it, total = from_state_idx.size):
            ## Row player: policy A, minimizer, Column player: policy B, maximizer
            matrixGame = nash.Game(estimated_q_function[it.multi_index[0], it.multi_index[1], it.multi_index[2], :, :])
            equilibriums = matrixGame.support_enumeration()
            max_entropy = -np.inf
            for eqs in equilibriums:
                curr_policy_A = eqs[0]
                curr_policy_B = eqs[1]
                curr_entropy = entropy(curr_policy_A) + entropy(curr_policy_B)
                if max_entropy < curr_entropy:
                    max_entropy = curr_entropy
                    policy_A[:, it.multi_index[0], it.multi_index[1], it.multi_index[2]] = curr_policy_A
                    policy_B[:, it.multi_index[0], it.multi_index[1], it.multi_index[2]] = curr_policy_B
        return policy_A, policy_B


    @staticmethod
    def sample_from_policy(policy_A, policy_B, game_env):
        curr_token_pos = game_env.state[0]
        curr_budget_A = game_env.state[1]
        curr_budget_B = game_env.state[2]
        curr_action_space = Alesia.get_action_space(curr_budget_A, curr_budget_B)
        
        if len(curr_action_space[0]) == 1 and curr_action_space[0] == 0:
            sample_action_A_success = True
        else:
            sample_action_A_success = False
        
        if len(curr_action_space[1]) == 1 and curr_action_space[1] == 0:
            sample_action_B_success = True
        else:
            sample_action_B_success = False

        curr_sampled_action_A = 0
        while not sample_action_A_success:
            curr_sampled_action_A = np.random.choice(np.arange(game_env.budget + 1), size = 1, p = policy_A[:, curr_token_pos, curr_budget_A, curr_budget_B])[0]
            if curr_sampled_action_A in curr_action_space[0]:
                sample_action_A_success = True
        
        curr_sampled_action_B = 0
        while not sample_action_B_success:
            curr_sampled_action_B = np.random.choice(np.arange(game_env.budget + 1), size = 1, p = policy_B[:, curr_token_pos, curr_budget_A, curr_budget_B])[0]
            if curr_sampled_action_B in curr_action_space[1]:
                sample_action_B_success = True
        return curr_sampled_action_A, curr_sampled_action_B
        
    
    def make_action(self):
        ## This is the main function
        for k in range(0, self.num_iter_k):
            print("Number of iteration: " + str(k))

            ## Draw samples by interacting with the environment
            training_set = self.game_env.sample_from_env(self.num_sample_n)

            ## Estimate reward function
            self.estimated_reward_function = Agent.estimate_reward_distribution(training_set, self.game_env)
            ## Estimate transition probability
            self.estimated_transition_function = Agent.estimate_transition_distribution(training_set, self.value_function, self.game_env)

            ## Estimate Value function and Q function
            self.q_function = Agent.estimate_q_function(self.q_function, self.estimated_reward_function, self.estimated_transition_function, self.policy_A, self.num_iter_q, self.gamma)

            self.policy_A, self.policy_B = Agent.find_optimal_policies(self.q_function, self.game_env)

            self.value_function = Agent.estimate_value_function_from_q_function(self.q_function, self.policy_A, self.policy_B)

        action_A, action_B = Agent.sample_from_policy(self.policy_A, self.policy_B, self.game_env)
        return action_A, action_B
        

def run_experiment(budget, token_space, gamma, num_iter_k, num_sample_n, num_iter_q, initial_q, initial_policy_A):
    np.random.seed(0) 

    states = []
    rewards = []
    actions = []

    # define agent and the environment
    env = Alesia(budget, token_space)
    env.reset()

    agent = Agent(env, gamma, num_iter_k, num_sample_n, num_iter_q, initial_q, initial_policy_A)


    terminate = False
    while not terminate:
        action_A, action_B = agent.make_action()

        terminate, reward, state, _ = env.step(action_A, action_B)

        states.append(state)
        rewards.append(reward)
        actions.append((action_A, action_B))
    return states, rewards


budget = 6
token_space = 5
gamma = 0.5
num_iter_k = 10
num_sample_n = 7 * 7 * 7 * 10
num_iter_q = 500
initial_q = np.zeros((token_space + 2, budget + 1, budget + 1, budget + 1, budget + 1))
initial_policy_A = None

states, rewards = run_experiment(budget, token_space, gamma, num_iter_k, num_sample_n, num_iter_q, initial_q, initial_policy_A)
