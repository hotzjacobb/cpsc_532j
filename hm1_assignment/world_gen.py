import numpy as np
from collections import defaultdict


def gen_states(burden=-0.04):
    w, h = 4, 3
    states = np.zeros((h, w)) + burden
    states[(2,3)] = 1
    states[(1,3)] = -1
    states[(1,1)] = np.nan
    return states

def attempt_shift(n, m, s_n, s_m, states):
    """return state after movement, might encounter side or wall"""
    if (0 <= n+s_n < states.shape[0]
        and 0 <= m+s_m < states.shape[1]
        and ~np.isnan(states[n+s_n, m+s_m])):
        return (n+s_n, m+s_m)
    else:
        return (n, m)

def move(n, m, states, p, v):
    """return the possible states after acting,
    along with their probabilities in the form of a dictionary
    """
    next_states = defaultdict(lambda : 0)
    if v:
        next_states[attempt_shift(n, m, 1 if p else -1, 0,
                                  states)] += 0.8
        next_states[attempt_shift(n, m, 0, -1, states)] += 0.1
        next_states[attempt_shift(n, m, 0, 1, states)] += 0.1
    else:
        next_states[attempt_shift(n, m, 0, 1 if p else -1,
                                  states)] += 0.8
        next_states[attempt_shift(n, m, -1, 0, states)] += 0.1
        next_states[attempt_shift(n, m, 1, 0, states)] += 0.1
    return next_states


def gen_Ps(states, indicies, p, v):
    P = np.zeros((len(states.flatten()),
                  len(states.flatten())))
    for n in range(states.shape[0]):
        for m in range(states.shape[1]):
            state_ps = move(n, m, states, p, v)
            for nn, nm in state_ps.keys():
                P[indicies[(n, m)],
                  indicies[(nn, nm)]] = state_ps[(nn, nm)]
    return P

def gen_transitions():
    w, h = 4, 3
    states = gen_states(0)
    absorbing_states = [(2,3), (1,3), (1,1)]
    indicies = np.arange(h*w).reshape((h,w))
    action_Ps = [gen_Ps(states, indicies, p, v)
               for p in (True, False)
               for v in (True, False)]
    for action_P in action_Ps:
        for a, b in absorbing_states:
            action_P[indicies[(a,b)]] = np.zeros(h*w)
            #bit hacky
    return np.stack(action_Ps)

def gen_absorbing_inds():
    w, h = 4, 3
    indicies = np.arange(h*w).reshape((h,w))
    absorbing_states = [(2,3), (1,3), (1,1)]
    absorbing_inds = [indicies[state] for state in absorbing_states]
    return absorbing_inds

def gen_rewards(burden):
    return np.nan_to_num(gen_states(burden)).flatten()