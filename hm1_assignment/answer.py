import argparse
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
    raise NotImplementedError
    return U

def best_policy(ps, U):
    """takes the current transition probabilities, and the estimates of utility for each state
    
    return a list of the best action to take, by index (index in ps), in each state
    e.g. [0, 3, 3, 3, 0, 0, 0, 0, 1, 1, 1, 0] is the policy for beta = -0.04"""
    raise NotImplementedError
    return policy
    
def policy_eval(ps, rewards, policy, gamma=1):
    """takes the current transition probabilities, the rewards for each state, the stochastic policy and gamma
    
    return the estimated Us
    
    note: iterate until the bellman update ceases to change the values of U"""
    raise NotImplimentedError
    return Us


def policy_iteration(ps, rewards, start_pol, gamma=1):
    """Perform policy iteration, takes the transition probability, the rewards and gamma and the initial policy
    
    return a sequence of policies calculated by policy iteration"""
    pols = [start_pol,]
    raise NotImplementedError
    ## policy iteration
    return pols

@handle('1')
def Q1():
    ## hint try running value iteration on different values of beta
    ## can you tell by the optimal policies if you probably skipped a beta with a different optimal policy?
    raise NotImplementedError    
    for beta in (-0.04,):
    ## add a full list of represenative betas each with a different optimal policy, and be sure to get all the different policies, (for beta < 0.)
    ## ensure your betas are in descending order, from least negative to most negative
        rewards = world_gen.gen_rewards(beta)
        U = value_iteration(ps, rewards, gamma=1)
        print(f"$\\pi_{{\\beta={beta}}}^*=$")
        policy = best_policy(ps, U)
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
    raise NotImplementedError

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
    raise NotImplementedError
    return U

def TD_update(U, episode, gamma=1, alpha=0.01):
    """Performs the TD updates to U as associated with the provided episode"""
    r_s, s_s, a_s = episode
    raise NotImplementedError
    return U
        
def Q_learning_update(Q, episode, gamma=1, alpha=0.01):
    """Performs the Q learning updates to Q as associated with the provided episode"""
    r_s, s_s, a_s = episode
    raise NotImplementedError
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
    
 
    
