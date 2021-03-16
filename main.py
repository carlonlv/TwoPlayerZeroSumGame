import numpy as np
import pandas as pd
import math
import time
import matplotlib.pyplot as plt
from itertools import chain


P = np.array([[[1, 0, 0, 0, 0, 0, 0],    # s_0 terminal \
               [1, 0, 0, 0, 0, 0, 0],    # s_1           |
               [0, 1, 0, 0, 0, 0, 0],    # s_2           |
               [0, 0, 1, 0, 0, 0, 0],    # s_3           |-> for action a_0 i.e. left          
               [0, 0, 0, 1, 0, 0, 0],    # s_4           |
               [0, 0, 0, 0, 1, 0, 0],    # s_5           |
               [0, 0, 0, 0, 0, 0, 1]],   # s_6 terminal /
              
              [[1, 0, 0, 0, 0, 0, 0],    # s_0 terminal \
               [0, 0, 1, 0, 0, 0, 0],    # s_1           |
               [0, 0, 0, 1, 0, 0, 0],    # s_2           |
               [0, 0, 0, 0, 1, 0, 0],    # s_3           |-> for action a_1 i.e. right
               [0, 0, 0, 0, 0, 1, 0],    # s_4           |
               [0, 0, 0, 0, 0, 0, 1],    # s_5           |
               [0, 0, 0, 0, 0, 0, 1]]])  # s_6 terminal /
#  State:       0  1  2  3  4  5  6



r = np.array([[0, 0],  # s_0
              [0, 0],  # s_1
              [0, 0],  # s_2
              [0, 0],  # s_3
              [0, 0],  # s_4
              [0, 1],  # s_5
              [0, 0]]) # s_6
# Action:    a_0  a_1

pi = np.array([[0.5, 0.5],  # s_0
               [0.5, 0.5],  # s_1
               [0.5, 0.5],  # s_2
               [0.5, 0.5],  # s_3
               [0.5, 0.5],  # s_4
               [0.5, 0.5],  # s_5
               [0.5, 0.5]]) # s_6
# Action:       a_0  a_1

tabular_features = np.array([[0, 0, 0, 0, 0],  # s_0 terminal
                             [1, 0, 0, 0, 0],  # s_1
                             [0, 1, 0, 0, 0],  # s_2  
                             [0, 0, 1, 0, 0],  # s_3
                             [0, 0, 0, 1, 0],  # s_4
                             [0, 0, 0, 0, 1],  # s_5
                             [0, 0, 0, 0, 0]]) # s_6 terminal

## Transition State, action 1, action 2, State (n * n * 6)

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
    def get_reward(token_pos, token_space):
        if token_pos == 0:
            return -1
        if token_pos == token_space + 1:
            return 1
        return 0

    @staticmethod
    def get_state_transition(token_space, total_budget, token_pos, budget_A, budget_B, action_A, action_B):
        # Assume that action_A and action_B are valid, that is curr_state[1] - action_A >= 0 and curr_state[2] - action_B >= 0
        # Also, self.terminate is False
        # Returns a 3d transition matrix, (token_pos, budget_A, budget_B)
        result = np.zeros((token_space + 2, total_budget + 1, total_budget + 1))
        if action_A is None and action_B is None:
            raise Exception('Both action A and B is None, called before checking termination.')
        if action_A is None:
            result[token_pos + 1, budget_A, budget_B - action_B] = 1
            return result
        if action_B is None:
            result[token_pos - 1, budget_A - action_A, budget_B] = 1
            return result
        if action_A == action_B:
            result[token_pos, budget_A - action_A, budget_B - action_B] = 1
        elif action_A > action_B:
            result[token_pos - 1, budget_A - action_A, budget_B - action_B] = 1
        else:
            result[token_pos + 1, budget_A - action_A, budget_B - action_B] = 1
        return result

    @staticmethod
    def get_action_space(budget_A, budget_B):
        return [list(range(1, budget_A + 1)), list(range(1, budget_B + 1))]  

    @staticmethod
    def get_token_pos_space(token_space):
        return list(range(0, token_space + 2)) 

    @staticmethod
    def state_sampler(num_samples, token_space, total_budget, transition_distribution, replace = False):
        token_pos_space = Alesia.get_token_pos_space(token_space)
        budget_A_space = list(range(0, total_budget + 1))
        budget_B_space = list(range(0, total_budget + 1))

        idx = np.arange(len(token_pos_space) * len(budget_A_space) * len(budget_B_space))
        sampled_state_idx = np.random.choice(idx, size = num_samples, replace = replace, 
        p = np.reshape(transition_distribution, -1))
        
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
        assert action_A in action_space[0]
        assert action_B in action_space[1]

        done = Alesia.check_termination(self.state[0], self.token_space, self.state[1], self.state[2])
        self.terminate = done
        
        reward = Alesia.get_reward(self.state[0], self.token_space)

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
            curr_sampled_reward = Alesia.get_reward(curr_sampled_to_state[0], self.token_space)
            sample = pd.Series([curr_sampled_token_pos, curr_sampled_budget_A, curr_sampled_budget_B, curr_sampled_action_A, curr_sampled_action_B, curr_sampled_to_state[0], curr_sampled_to_state[1], curr_sampled_to_state[2], curr_sampled_reward],
            name = ["from_token_pos", "from_budget_A", "from_budget_B", "action_A", "action_B", "to_token_pos", "to_budget_A", "to_budget_B", "reward"])
            sampled_data = pd.concat([sampled_data, sample])
        return sampled_data        


class Agent():

    def __init__(self, num_actions, policy_features, value_features,
                 policy_stepsize, value_stepsize, nstep, lambda_, gamma,
                 FLAG_BASELINE, FLAG_POPULAR_PG=False):
        self.policy_features = policy_features
        self.value_features = value_features
        self.num_actions = num_actions

        self.policy_weight = np.zeros((policy_features.shape[1],
                                       num_actions))
        if value_features is None:
            self.value_weight = None
        else:
            self.value_weight = np.asarray([[-0.04230291],
                                            [ 0.139734  ],
                                            [ 0.14239203],
                                            [ 0.00787976],
                                            [ 0.91098493]])
            # self.value_weight = np.zeros((value_features.shape[1], 1))

        self.policy_stepsize = policy_stepsize
        self.value_stepsize = value_stepsize

        self.FLAG_BASELINE = FLAG_BASELINE
        self.FLAG_POPULAR_PG = FLAG_POPULAR_PG
        self.gamma = gamma
        # Parameter for calculating the generalized advantage.
        self.lambda_ = lambda_
        self.nstep = nstep

        self.pi = None
        self.FLAG_POLICY_UPDATED = True

    @staticmethod
    def softmax(x):
        e_x = np.exp(x - np.max(x, 1).reshape(-1, 1))
        out = e_x / e_x.sum(1).reshape(-1, 1)
        return out

    # At a given state, use the existing stochastic policy to decide which 
    # action to take.
    def take_action(self, state):
        if self.FLAG_POLICY_UPDATED:
            action_prefs = np.matmul(self.policy_features, self.policy_weight)
            self.pi = self.softmax(action_prefs)
            self.FLAG_POLICY_UPDATED = False
            
        action = np.random.choice(self.num_actions, p=self.pi[state])
        return action, self.pi[state, action]

    # Use the current value functions to make predictions.
    def calc_v_pi_pred(self):
        return np.matmul(self.value_features, self.value_weight)

    # =========================================================================
    # Calculate the advantage for a specific step.
    def calc_TD(self, curr_state, next_state, reward, v_pi):
        return reward + (self.gamma * self.value_features[next_state] @ self.value_weight - \
                         self.value_features[curr_state] @ self.value_weight)
        # return reward + (v_pi[next_state] - v_pi[curr_state])

    # Calculate GAE based on infinite horizon with absorbing states (s_0 and 
    # s_6). In the absorbing states though, the value function and the reward 
    # stay to be 0 and thus the TD at each absorbing step is also 0.
    def calc_generalized_advantage(self, t, traj, v_pi, q_pi):
        reward_list = traj['reward_list']
        next_state_list = traj['next_state_list']
        state_list = traj['state_list']
        action_list = traj['action_list']
        traj_length = len(reward_list)

        GAE = 0
        discount = 1
        for i in range(t, traj_length):
            GAE += discount * self.calc_TD(state_list[i], 
                                           next_state_list[i], 
                                           reward_list[i],
                                           v_pi)
            discount *= self.gamma * self.lambda_
            # discount *= self.gamma * 100

        # Calculate the ground truth unbiased advantage based on the ground
        # truth state action value function and state value function.
        true_advantage = q_pi[state_list[t], action_list[t]] - \
                         v_pi[state_list[t]]
            
        return GAE, true_advantage
    # =========================================================================
    

    # helper function for calculating the policy gradient.
    def calc_grad_log_pi(self, state, action):
        x = self.policy_features[state].reshape(-1, 1)
        action_prefs = np.matmul(x.T, self.policy_weight)
        pi = self.softmax(action_prefs).T

        I_action = np.zeros((self.num_actions, 1))
        I_action[action] = 1

        one_vec = np.ones((1, self.num_actions))

        return np.matmul(x, one_vec) * (I_action - pi).T

    # Calculate the GAE based policy gradient estimator or
    # calculate the true advantage based policy gradient.
    def calc_gae_pg(self, trajs, v_pi, q_pi, estimate=True):
        policy_grad = np.zeros(self.policy_weight.shape)
        for traj in trajs:
            state_list = traj['state_list']
            action_list = traj['action_list']
            traj_length = len(state_list)
            
            for t in range(traj_length):
                state = state_list[t]
                action = action_list[t]
                GAE, true_advantage = self.calc_generalized_advantage(t, traj, v_pi, q_pi)
                grad_log_pi = self.calc_grad_log_pi(state, action)
                
                if self.FLAG_BASELINE:
                    baseline = v_pi[state]
                else:
                    baseline = 0
                # policy_grad += self.gamma**t * GAE * grad_log_pi
                if estimate == True:
                    policy_grad += GAE * grad_log_pi
                else:
                    policy_grad += true_advantage * grad_log_pi

        return policy_grad / len(trajs)


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



