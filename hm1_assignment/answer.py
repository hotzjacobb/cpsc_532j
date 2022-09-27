import argparse
from xml.sax.handler import all_properties
func_registry = {}


def handle(number):
    def register(func):
        func_registry[number] = func
        return func

    return register


def run(question):
    if question not in func_registry:
        raise ValueError(f"unknown question {question}")
    return func_registry[question]()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-q", "--question", required=True, choices=func_registry.keys())
    args = parser.parse_args()
    return run(args.question)


import world_gen, episode_sim, policy_print
import numpy as np
from copy import deepcopy

ps = world_gen.gen_transitions()
## ps[0][i, j] is the probability of moving to j when trying to move up from state i
## ps[1][i, j] the chance of moving from i to j, when trying to move right
## ps[2] the same for down
## ps[3] the same for left

states = world_gen.gen_states()
w, h = 4, 3
rewards = world_gen.gen_rewards(-0.04)
## rewards[i] is the reward that comes from being in state i


def value_iteration(ps, rewards, gamma=1):
    """takes the current transition probabilities, as a stack a four matrices
    along with a list of rewards, and a gamma factor and performs value iteration to calculate U
    
    return U
    
    note: iterate until the bellman update ceases to change the values of U"""
    state_indices = range(rewards.shape[0])

    U = np.zeros(rewards.shape[0])
    new_utilities = np.zeros(rewards.shape[0])
    old_utilities = np.zeros(rewards.shape[0])
    while True:
        for curr_state in range(rewards.shape[0]):
            reward = rewards[curr_state]

            # calculate the best action
            # (state utility * probability) for all states resulting from performing the action from the current state
            up_value = np.sum([ps[0][curr_state, new_state] * new_utilities[new_state] for new_state in state_indices])
            right_value = np.sum([ps[1][curr_state, new_state] * new_utilities[new_state] for new_state in state_indices])
            down_value = np.sum([ps[2][curr_state, new_state] * new_utilities[new_state] for new_state in state_indices])
            left_value = np.sum([ps[3][curr_state, new_state] * new_utilities[new_state] for new_state in state_indices])

            update_util = np.max([up_value, right_value, down_value, left_value])

            old_utilities = deepcopy(new_utilities)
            new_utilities[curr_state] = reward + (gamma * update_util)

            if (all((((abs(old_util - new_util) < 1e-31) and (update_util != 0))) for old_util, new_util in zip(old_utilities, new_utilities))):
                U = new_utilities
                return U

def best_policy(ps, U):
    """takes the current transition probabilities, and the estimates of utility for each state
    
    return a list of the best action to take, by index (index in ps), in each state
    e.g. [0, 3, 3, 3, 0, 0, 0, 0, 1, 1, 1, 0] is the policy for beta = -0.04"""
    state_indices = range(rewards.shape[0])
    policy = np.zeros(rewards.shape[0], dtype = np.int64)

    for curr_state in state_indices:
        up_utility = np.sum([ps[0][curr_state, new_state] * U[new_state] for new_state in state_indices])
        right_utility = np.sum([ps[1][curr_state, j] * U[j] for j in state_indices])
        down_utility = np.sum([ps[2][curr_state, j] * U[j] for j in state_indices])
        left_utility = np.sum([ps[3][curr_state, j] * U[j] for j in state_indices])
        action_utilities = [up_utility, right_utility, down_utility, left_utility]
        if up_utility == np.max(action_utilities):
            policy[curr_state] = 0
        elif right_utility == np.max(action_utilities):
            policy[curr_state] = 1
        elif down_utility == np.max(action_utilities):
            policy[curr_state] = 2
        elif left_utility == np.max(action_utilities):
            policy[curr_state] = 3

    return policy
    
def policy_eval(ps, rewards, policy, gamma=1):
    """takes the current transition probabilities, the rewards for each state, the stochastic policy and gamma
    
    return the estimated Us
    
    note: iterate until the bellman update ceases to change the values of U"""
    state_indices = range(rewards.shape[0])

    Us = np.zeros(rewards.shape[0])
    new_state = np.zeros(rewards.shape[0])
    old_utilities = np.zeros(rewards.shape[0])
    while True:
        for curr_state in range(rewards.shape[0]):
            reward = rewards[curr_state]

            # calculate the best action
            # (state utility * probability) for all states resulting from performing the action from the current state
            up_value = np.sum([ps[0][curr_state, j] * new_state[j] for j in state_indices])
            right_value = np.sum([ps[1][curr_state, j] * new_state[j] for j in state_indices])
            down_value = np.sum([ps[2][curr_state, j] * new_state[j] for j in state_indices])
            left_value = np.sum([ps[3][curr_state, j] * new_state[j] for j in state_indices])

            # give values weights according to the policy
            up_value_weighted = up_value * policy[0][curr_state]
            right_value_weighted  = right_value * policy[1][curr_state]
            down_value_weighted  = down_value * policy[2][curr_state]
            left_value_weighted  = left_value * policy[3][curr_state]
            
            update_util = np.sum([up_value_weighted, right_value_weighted, down_value_weighted, left_value_weighted])

            old_utilities = deepcopy(new_state)
            new_state[curr_state] = reward + (gamma * update_util)

            if (all((((abs(old_util - new_util) < 1e-31) and (update_util != 0))) for old_util, new_util in zip(old_utilities, new_state))):
                Us = new_state
                return Us


def policy_iteration(ps, rewards, start_pol, gamma=1):
    """Perform policy iteration, takes the transition probability, the rewards and gamma and the initial policy
    
    return a sequence of policies calculated by policy iteration"""
    #pols = [start_pol,]
    pols = []
    state_indices = range(rewards.shape[0])

    U = np.zeros(rewards.shape[0])

    policy = start_pol
    while True:
        policy_changed = False
        U = policy_eval(ps, rewards, policy, gamma)
        for curr_state in range(rewards.shape[0]):

            # calculate the best action
            # (state utility * probability) for all states resulting from performing the action from the current state
            up_value = np.sum([ps[0][curr_state, new_state] * U[new_state] for new_state in state_indices])
            right_value = np.sum([ps[1][curr_state, new_state] * U[new_state] for new_state in state_indices])
            down_value = np.sum([ps[2][curr_state, new_state] * U[new_state] for new_state in state_indices])
            left_value = np.sum([ps[3][curr_state, new_state] * U[new_state] for new_state in state_indices])

            update_util = np.max([up_value, right_value, down_value, left_value])
            update_move = np.argmax([up_value, right_value, down_value, left_value])

            # assumes that policy is non-stochastic
            current_mov_index = np.argmax(policy, axis=0)[curr_state]
            if current_mov_index == 0:
                curr_mov_value = up_value
            elif current_mov_index == 1:
                curr_mov_value = right_value
            elif current_mov_index == 2:
                curr_mov_value = down_value 
            else: # current_mov_index == 3
                curr_mov_value = left_value  


            if update_util > curr_mov_value:
                old_pol = deepcopy(policy)
                pols.append(old_pol)
                policy[update_move][curr_state] = 1
                move_indices = [move for move in range(4) if move != update_move]
                for mov_index in move_indices:
                    policy[mov_index][curr_state] = 0
                policy_changed = True

            if not policy_changed:
                pols.append(policy)
                return pols
    
    return pols

@handle('1')
def Q1():
    ## hint try running value iteration on different values of beta
    ## can you tell by the optimal policies if you probably skipped a beta with a different optimal policy?  
    for beta in (-.025, -.029, -.050, -.100, -.500, -.900, -1.60, -1.70):
    ## add a full list of represenative betas each with a different optimal policy, and be sure to get all the different policies, (for beta < 0.)
    ## ensure your betas are in descending order, from least negative to most negative
        rewards = world_gen.gen_rewards(beta)
        U = value_iteration(ps, rewards, gamma=1)
        print(f"$\\pi_{{\\beta={beta}}}^*=$")
        policy = best_policy(ps, U)
        # TODO: delete line below
        #policy_print.print_policy(policy)
        policy_print.latex_policy(policy)
        
@handle('1.1')
def Q1_1():
    beta = 0.1
    rewards = world_gen.gen_rewards(beta)
    U = value_iteration(ps, rewards, gamma=0.99)
    policy = best_policy(ps, U)
    policy_print.latex_policy(policy)
        
@handle('2')     
def Q2():
    ## note this is a stochastic policy, unlike the ones we printed in Q1 which merely give the best action to take in each state.
    rewards = world_gen.gen_rewards(-0.04)
    policy = np.zeros((4, len(rewards)))
    policy += 1/4
    U = policy_eval(ps, rewards, policy, gamma=1)
    policy_print.latex_grid(U.reshape(h, w))

@handle('3')
def Q3():
    rewards = world_gen.gen_rewards(-0.04)
    up_policy = np.zeros((4, len(rewards)))
    up_policy[0,:] = 1
    pols = policy_iteration(ps, rewards, up_policy, gamma=1)
    for n, pol in enumerate(pols):
        ## if you have kept the stochastic representation, you need to convert it to a deterministic policy for printing,
        ## e.g. pol.argmax(axis=0)
        print(f"$\\pi_{{{n}}}=$")
        policy_print.latex_policy(pol.argmax(axis=0))
    #raise NotImplementedError


import json
with open('episodes.json', 'r') as file:
    episodes = json.load(file)


def MC_update(U, episode, gamma=1, alpha=0.01):
    """Performs the MC updates to U as associated with the provided episode"""
    r_s, s_s, a_s = episode
    ep_length = len(r_s)
    state_returns = np.zeros(len(s_s))
    for timestep in reversed(range(ep_length)):
        terminal = timestep == len(r_s) - 1
        state_returns[timestep] = r_s[timestep] + (gamma * state_returns[timestep+1]) if not terminal else r_s[timestep]
    for i, state_return in enumerate(state_returns):
        state = s_s[i]
        U[state] = U[state] + alpha * (state_return - U[state])
    return U

def TD_update(U, episode, gamma=1, alpha=0.01):
    """Performs the TD updates to U as associated with the provided episode"""
    r_s, s_s, a_s = episode
    ep_length = len(r_s)

    for timestep in range(ep_length):
        terminal = timestep == len(r_s) - 1
        state = s_s[timestep]
        next_state = s_s[timestep + 1] if not terminal else None
        reward = r_s[timestep]
        if not terminal:
            U[state] = U[state] + alpha * (reward + (gamma * U[next_state]) - U[state])
        else:
            U[state] = U[state] + alpha * (reward - U[state])

    return U
        
def Q_learning_update(Q, episode, gamma=1, alpha=0.01):
    """Performs the Q learning updates to Q as associated with the provided episode"""
    r_s, s_s, a_s = episode
    ep_length = len(r_s)

    for timestep in range(ep_length):
        terminal = timestep == len(r_s) - 1
        state = s_s[timestep]
        next_state = s_s[timestep + 1] if not terminal else None
        reward = r_s[timestep]
        action = np.argmax(Q[state]) # greedily chose the action from existing Q values
        if not terminal:
            up_Q = Q[next_state][0]
            right_Q = Q[next_state][1]
            down_Q = Q[next_state][2]
            left_Q = Q[next_state][3]
            next_state_action = np.argmax([up_Q, right_Q, down_Q, left_Q])
            Q[state][action] = Q[state][action] + alpha * (reward + (gamma * Q[next_state][next_state_action]) - Q[state][action])
        else:
            Q[state][action] = Q[state][action] + alpha * (reward - Q[state][action])
    return Q

@handle('4')  
def Q4():
    U = np.zeros(h*w)
    for ep in episodes:
        U = MC_update(U, [ep['e_rewards'], ep['e_states'], ep['e_actions']])
    policy_print.latex_grid(U.reshape(h, w))

@handle('5') 
def Q5():
    U = np.zeros(h*w)
    for ep in episodes:
        U = TD_update(U, [ep['e_rewards'], ep['e_states'], ep['e_actions']])
    policy_print.latex_grid(U.reshape(h, w))

@handle('6')   
def Q6():
    ## check the orientation of your Q, I used this, but could do (4, 12)
    Q = np.zeros((h*w, 4)) 
    for ep in episodes:
        Q = Q_learning_update(Q, [ep['e_rewards'], ep['e_states'], ep['e_actions']])
    U = Q.max(axis=1)
    policy_print.latex_grid(U.reshape(h, w))
    policy_print.latex_policy(best_policy(ps, U))
          
    ## and then print the best policy
    
if __name__ == "__main__":
    main()
    
 
    
