import math
import time
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn as sk

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
        self.state = (math.floor(self.token_space) / 2 , self.budget, self.budget)
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
            return np.random.uniform(low = lower_point, high = 0.5, size = 1)
            
        elif action_A > action_B:
            if token_pos == 1:
                return -1

            higher_point = np.linspace(start = 0, stop = -0.5, num = budget_A, endpoint = False)
            if action_B is None:
                higher_point = higher_point[len(higher_point) - action_A]
            else:
                higher_point = higher_point[len(higher_point) - (action_A - action_B)]
            return np.random.uniform(low = -0.5, high = higher_point, size = 1)
        
        else:
            higher_point = np.linspace(start = 0, stop = -0.5, num = budget_A, endpoint = False)
            higher_point = higher_point[len(higher_point) - action_A]
            lower_point = np.linspace(start = 0, stop = 0.5, num = budget_B, endpoint = False)
            lower_point = lower_point[len(lower_point) - action_B]
            return np.random.uniform(low = higher_point, high = lower_point, size = 1)

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
        # This sampling method is not based on trajectory
        sampled_data = pd.DataFrame(columns=["from_token_pos", "from_budget_A", "from_budget_B", "action_A", "action_B", "to_token_pos", "to_budget_A", "to_budget_B", "reward"])
        
        ## Independently sample states and bugets, also uniformly, can be changed
        sampled_token_pos = np.random.randint(low = 1, high = len(self.token_space) - 1, size = num_samples)
        sampled_budget_A = np.random.randint(low = 1, high = self.budget + 1, size = num_samples)
        sampled_budget_B = np.random.randint(low = 1, high = self.budget + 1, size = num_samples)
        
        for i in range(0, num_samples):
            curr_sampled_token_pos = sampled_token_pos[i]
            curr_sampled_budget_A = sampled_budget_A[i]
            curr_sampled_budget_B = sampled_budget_B[i]
            action_space = Alesia.get_action_space(curr_sampled_budget_A, curr_sampled_budget_B)
            curr_sampled_action_A = np.random.choice(action_space[0], size = 1)
            curr_sampled_action_B = np.random.choice(action_space[1], size = 1)
            token_budget_state_space = Alesia.get_state_transition(self.token_space, self.budget, curr_sampled_token_pos, curr_sampled_budget_A, curr_sampled_budget_B, curr_sampled_action_A, curr_sampled_action_B)
            curr_sampled_to_state = Alesia.state_sampler(1, self.token_space, self.budget, token_budget_state_space)
            curr_sampled_reward = Alesia.get_reward(curr_sampled_token_pos, curr_sampled_budget_A, curr_sampled_budget_B, curr_sampled_action_A, curr_sampled_action_B, self.token_space)
            sample = pd.DataFrame([[curr_sampled_token_pos, curr_sampled_budget_A, curr_sampled_budget_B, curr_sampled_action_A, curr_sampled_action_B, curr_sampled_to_state[0], curr_sampled_to_state[1], curr_sampled_to_state[2], curr_sampled_reward]], columns = ["from_token_pos", "from_budget_A", "from_budget_B", "action_A", "action_B", "to_token_pos", "to_budget_A", "to_budget_B", "reward"])
            sampled_data = pd.concat([sampled_data, sample], axis = 0)
        return sampled_data        


class Agent():

    def __init__(self, game_env, gamma, num_iter_k, num_sample_n, num_iter_q, initial_q, initial_policy_A):
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
        self.policy_A = initial_policy_A
        ## 6 dimension array in order of (action_B, from_token_pos, from_budget_A, from_budget_B)
        self.policy_B = None
        ## 5 dimension array in order of (token_pos, budget_A, budget_B, action_A, action_B)
        self.q_function = initial_q
        ## 3 dimension array in order of (token_pos, budget_A, budget_B)
        self.value_function = None

    @staticmethod
    def estimate_reward_distribution(training_set, game_env):
        state_action_pair = training_set[["from_token_pos", "from_budget_A", "from_budget_B", "action_A", "action_B"]]
        reward = training_set['reward']
        trained_model = sk.linear_model.LinearRegression().fit(X = state_action_pair, y = reward)
        
        estimated_reward_function = np.zeros(((game_env.token_space + 2, game_env.budget + 1, game_env.budget + 1, game_env.budget + 1, game_env.budget + 1)))
        
        with np.nditer(estimated_reward_function, flags = ["multi_index"], op_flags = ["readwrite"]) as it:
            for x in it:
                regressor = pd.DataFrame([list(it.multi_index)], columns = state_action_pair.values.tolist())
                x[...] = trained_model.predict(regressor)[0]

        return estimated_reward_function

    @staticmethod
    def estimate_transition_distribution(training_set, estimated_value_function, game_env):   
        from_state_action_pair = training_set[["from_token_pos", "from_budget_A", "from_budget_B", "action_A", "action_B"]]
        to_states = training_set[["to_token_pos", "to_budget_A", "to_budget_B"]]


        ## Here we use multionmial logistic to model the transition probability, this method does not consider difference in value function.
        
        state_index_dict = np.arange((game_env.token_space + 2) * (game_env.budget + 1) * (game_env.budget + 1)).reshape((game_env.token_space + 2, game_env.budget + 1, game_env.budget + 1))
        to_states_index = np.zeros(len(to_states.index))
        
        for i in range(0, len(to_states.index)):
            to_states_index[i] = state_index_dict[to_states.iloc[i, 0], to_states.iloc[i, 1], to_states.iloc[i, 2]]
        to_states_index = pd.Series(to_states_index)

        trained_model = sk.linear_model.LogisticRegression(multi_class= "multinomial", solver="lbfgs").fit(from_state_action_pair, to_states_index)

         ## 8 dimension array in order of (from_token_pos, from_budget_A, from_budget_B, to_token_pos, to_budget_A, to_budget_B, action_A, action_B)
        estimated_transition_dist = np.zeros((game_env.token_space + 2, game_env.budget + 1, game_env.budget + 1, game_env.token_space + 2, game_env.budget + 1, game_env.budget + 1, game_env.budget + 1, game_env.budget + 1))

        from_state_action_idx = np.arange((game_env.token_space + 2) * (game_env.budget + 1) * (game_env.budget + 1) * (game_env.budget + 1) * (game_env.budget + 1)).reshape((game_env.token_space + 2, game_env.budget + 1, game_env.budget + 1, game_env.budget + 1, game_env.budget + 1))
        it = np.nditer(from_state_action_idx, flags = ["multi_index"], op_flags = ["readwrite"])
        for _ in it:
            regressor = pd.DataFrame([list(it.multi_index)], columns = from_state_action_pair.values.tolist())
            predicted_probs = trained_model.predict_proba(regressor)
            to_probs = np.zeros((game_env.token_space + 2) * (game_env.budget + 1) * (game_env.budget + 1))
            to_probs[trained_model.classes_] = predicted_probs
            to_probs = to_probs.reshape((game_env.token_space + 2, game_env.budget + 1, game_env.budget + 1))
            estimated_transition_dist[it.multi_index[0], it.multi_index[1], it.multi_index[2], :, :, :, it.multi_index[6], it.multi_index[7]] = to_probs
        return estimated_transition_dist

    
    @staticmethod
    def estimate_q_function(q_0, estimated_reward_function, estimated_transition_function, policy_A, num_iter_q, gamma):
        curr_q = q_0
        for _ in range(num_iter_q):
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
            mod_value_func = value_func[:, :, :, np.newaxis, np.newaxis, np.newaxis, :, :]
            curr_q = estimated_reward_function + gamma * np.sum(np.multiply(estimated_transition_function, mod_value_func), axis = (3, 4, 5))

        return curr_q
    
    @staticmethod
    def find_optimal_policies(estimated_q_function, game_env):
        ## Q function :(token_pos, budget_A, budget_B, action_A, action_B)
        ## Policy A is 4 dimension (action_A, from_token_pos, from_budget_A, from_budget_B), policy A minimizes target
        policy_A = np.zeros((game_env.budget + 1) * (game_env.token_space + 2) * (game_env.budget + 1) * (game_env.budget + 1))
        
        from_state_action_idx = np.arange((game_env.token_space + 2) * (game_env.budget + 1) * (game_env.budget + 1) * (game_env.budget + 1) * (game_env.budget + 1)).reshape((game_env.token_space + 2, game_env.budget + 1, game_env.budget + 1, game_env.budget + 1, game_env.budget + 1))
        it = np.nditer(from_state_action_idx, flags = ["multi_index"], op_flags = ["readwrite"])
        for _ in it:
            policy_A[:, it.multi_index[0], it.multi_index[1], it.multi_index[2]] = np.argmin(estimated_q_function[it.multi_index[0], it.multi_index[1], it.multi_index[2], :, :], axis = 0)


        ## Policy B is 4 dimension (action_B, from_token_pos, from_budget_A, from_budget_B), policy B maximizes target
        
            

    def update_policy(self):
        return 0

    def make_action(self):
        ## This is the main function
        for k in range(0, self.num_iter_k):
            print(k)

            ## Draw samples by interacting with the environment
            training_set = self.game_env.sample_from_env(self.num_sample_n)

            ## Estimate reward function
            self.estimated_reward_function = Agent.estimate_reward_distribution(training_set, self.game_env)
            ## Estimate transition probability
            self.estimated_transition_function = Agent.estimate_transition_distribution(training_set, self.value_function, self.game_env)

            ## Estimate Value function and Q function
            Agent.estimate_q_function(self.q_function, self.estimated_reward_function, self.estimated_transition_function, policy_A, num_iter_q, gamma)


            ### Initialize Q_k_0
            for q in range(0, self.num_iter_q):
                print(q)
        return None

    

def run_experiment(num_runs, num_episodes,
                   P, r, budget, token_space, policy_features, value_features,
                   policy_stepsize, value_stepsize, nstep, lambdas, gamma,
                   FLAG_BASELINE, FLAG_LEARN_VPI, reward_noise=0, vpi_bias=0):
    bias_ = []
    variances = []

    for lambda_ in lambdas:
        np.random.seed(0) 

        # define agent and the environment
        env = Alesia(budget, token_space)

        agent = Agent(num_actions, policy_features, value_features,
                    policy_stepsize, value_stepsize, nstep, lambda_, gamma,
                    FLAG_BASELINE)

        return_across_episodes = []
        ep_len_across_episodes = []
        vpi_across_episodes = []

        sample_trajs = []
        grad_estimators = []
        for sample in range(num_samples):

            episode_trajs = []
            for episode in range(num_episodes):

                traj = {'state_list': [],
                        'action_list': [],
                        'action_prob_list': [],
                        'reward_list': [],
                        'next_state_list': []}

                # sample a trajectory from following the current policy
                done = False
                state = env.reset()
                while not done:
                    action, action_prob = agent.take_action(state)
                    done, reward, next_state , _ = env.step(action)

                    traj['state_list'].append(state)
                    traj['action_list'].append(action)
                    traj['action_prob_list'].append(action_prob)
                    traj['reward_list'].append(reward)
                    traj['next_state_list'].append(next_state)

                    state = next_state
                
                # Set v_pi to always be the ground truth. Since in the GAE setting, 
                # estimated v_pi can be calculated using the newly defined helper 
                # function `cal_TD`, so the ground truth v_pi can be used to 
                # calculate the ground truth unbiased advantage, for the later use
                # of bias-variance calculation.
                v_pi = env.calc_v_pi(agent.pi, gamma)
                q_pi = env.calc_q_pi(agent.pi, gamma)


                # Update the trajectory lists.
                episode_trajs.append(traj)
                sample_trajs.append(traj)

            grad_estimators.append(agent.calc_gae_pg(episode_trajs, v_pi, q_pi))
        
        true_gradient = agent.calc_gae_pg(sample_trajs, v_pi, q_pi, estimate=False)

        bias = (np.mean(np.asarray(grad_estimators), axis=0) - true_gradient) ** 2
        variance = np.var(np.asarray(grad_estimators), axis=0)
        bias_.append(bias)
        variances.append(variance)

        # print("Lambda {}".format(lambda_))
    bias_ = np.asarray(bias_)
    variances = np.asarray(variances)
    plt.plot(lambdas, bias_[:, 0, 0], label="bias")
    plt.plot(lambdas, variances[:, 0, 0], label="variance")
    plt.plot(lambdas, bias_[:, 0, 0] + variances[:, 0, 0], label="MSE")
    plt.xlabel("lambda")
    plt.legend()
    plt.title("bias-variance tradeoff for the (0, 0) entry of policy gradients")
    plt.show()

    plt.plot(lambdas, bias_[:, 0, 1], label="bias")
    plt.plot(lambdas, variances[:, 0, 1], label="variance")
    plt.plot(lambdas, bias_[:, 0, 1] + variances[:, 0, 1], label="MSE")
    plt.xlabel("lambda")
    plt.legend()
    plt.title("bias-variance tradeoff for the (0, 1) entry of policy gradients")
    plt.show()


    return



num_runs = 10
num_episodes = 20
num_samples = 50

start_state = 1
terminal_states = [0, 6]
reward_noise = 0.3
gamma = 0.5
lambdas = np.linspace(0.5, 0.99, 10)

num_actions = 2
FLAG_BASELINE = True

# nstep_list = [1, 2, 4, 16, 'inf']
# stepsize_list = [0.1, 0.3, 0.5, 0.7, 1]
nstep_list = [16]
stepsize_list = [0.1]


FLAG_LEARN_VPI = True
value_features = tabular_features
policy_features = tabular_features

tic = time.time()
exp_data1 = dict()
print('Starting the experiments. Estimated time to completion: 1000 seconds')
for nstep in nstep_list: 
    print("nstep: {}".format(nstep))
    exp_data1[nstep] = dict()
    for stepsize in stepsize_list:
        policy_stepsize = stepsize
        value_stepsize = stepsize

        dat = run_experiment(num_runs, num_episodes,
                             budget, token_space, policy_features, value_features,
                             policy_stepsize, value_stepsize, nstep, lambdas, gamma, 
                             FLAG_BASELINE, FLAG_LEARN_VPI, reward_noise)
        
        exp_data1[nstep][stepsize] = dat
    print('nstep: {}\ttime elapsed: {:.0f}s'.format(nstep, time.time() - tic)) 



