import numpy as np
from numpy.random import default_rng
rng = default_rng()

def sim(ps, s, a, rewards):
    """samples the new state when performing action a in state s, according to ps"""
    new_s = rng.choice(np.arange(len(rewards)), p=ps[a][s])
    return (new_s, rewards[new_s])

def episode(ps, s, policy, rewards, absorbing_inds):
    """calculates an episode following the policy starting in state s, according to ps
    the episode ends when an absorbing state (here index) is encountered
    
    return the history of rewards, states and actions"""
    cumu_r = [rewards[s],]
    cumu_s = [s,]
    cumu_a = []
    while s not in absorbing_inds:
        a = rng.choice(np.arange(4), p=policy[:, s])
        new_s, reward = sim(ps, s, a, rewards)
        s = new_s
        cumu_r.append(reward)
        cumu_s.append(s)
        cumu_a.append(a)
    return cumu_r, cumu_s, cumu_a