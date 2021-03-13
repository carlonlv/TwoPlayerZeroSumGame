import numpy as np
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
    def __init__(self, P, r, budget, token_space):
        self.P = P
        self.r = r

        self.budget = budget
        ## Fair game if token_space is odd, favour to player A (go towards index 0) if even
        self.token_space = list(range(token_space + 2))
        self.terminal_state = (0, token_space + 1)
    
        self.state = None

        self.t = 0

    def reset(self):
        ## Full budget for player A and B, token_space is set to be the middle
        self.state = (math.floor(self.token_space / 2) , self.budget, self.budget)

    def get_state_transition(self, curr_state, action_A, action_B):
        ## Assume that action_A and action_B are valid, that is curr_state[1] - action_A >= 0 and curr_state[2] - action_B >= 0

        result = np.zeros((self.token_space, self.budget, self.budget))
        if action_A == action_B:
            result[curr_state[0], curr_state[1] - action_A, curr_state[2] - action_B] = 1
        elif action_A > action_B:
            result[curr_state[0] - 1, curr_state]

        return result


    def get_reward(self, curr_state, action_A, action_B):
        return 0

    def step(self, action):
        if self.state is None:
            raise Exception('step() used before calling reset()')
        assert action in range(self.P.shape[0])

        reward = self.r[self.state, action] \
            + np.random.normal(loc=0, scale=0)
        self.state = np.random.choice(a=self.n, p=self.P[action, self.state])
        self.t = self.t + 1

        done = False
        if self.state in self.terminal_states:
            done = True

        return self.state, reward, done, {}

    def calc_v_pi(self, pi, gamma):
        # calculate P_pi from the transition matrix P and the policy pi
        P_pi = np.zeros(self.P[0].shape)
        for a in range(pi.shape[1]):
            P_pi += self.P[a] * pi[:, a].reshape(-1, 1)

        # calculate the vector r_pi
        r_pi = (self.r * pi).sum(1).reshape(-1, 1)

        # calculate v_pi using the equation given above
        v_pi = np.matmul(
            np.linalg.inv(np.eye(self.P.shape[-1]) - gamma * P_pi), 
            r_pi)

        return v_pi

    # Calculate the ground truth state-action value function based on the 
    # ground truth state value function.
    def calc_q_pi(self, pi, gamma):
        # First calcuate the ground truth value vector.
        v_pi = self.calc_v_pi(pi, gamma)
        q_pi = self.r + gamma * (self.P @ v_pi)[:, :, 0].T
        return q_pi



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
                   P, r, start_state, terminal_state,
                   num_actions, policy_features, value_features,
                   policy_stepsize, value_stepsize, nstep, lambdas, gamma,
                   FLAG_BASELINE, FLAG_LEARN_VPI, reward_noise=0, vpi_bias=0):
    bias_ = []
    variances = []

    for lambda_ in lambdas:
        np.random.seed(0) 

        # define agent and the environment
        env = Alesia(P, r, start_state, terminal_states, reward_noise)

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
                    next_state, reward, done, _ = env.step(action)

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
                             P, r, start_state, terminal_states,
                             num_actions, policy_features, value_features,
                             policy_stepsize, value_stepsize, nstep, lambdas, gamma, 
                             FLAG_BASELINE, FLAG_LEARN_VPI, reward_noise)
        
        exp_data1[nstep][stepsize] = dat
    print('nstep: {}\ttime elapsed: {:.0f}s'.format(nstep, time.time() - tic)) 



