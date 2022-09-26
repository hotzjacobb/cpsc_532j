import numpy as np

w, h = 4, 3
indicies = np.arange(h*w).reshape((h,w))
absorbing_states = [(2,3), (1,3), (1,1)]
absorbing_inds = [indicies[state] for state in absorbing_states]

def print_policy(policy):
    policy = np.array(policy)
    for state in absorbing_states:
        policy[indicies[state]] = 4
    for line in np.array([['↑', '→', '↓', '←', 'x'][i]
                          for i in policy]).reshape(h, w)[::-1]:
        print(" ".join(line))
        
def latex_policy(policy):
    print(f"\\begin{{tabular}}{{{' '.join('c')*w}}}")
    policy = np.array(policy)
    for state in absorbing_states:
        policy[indicies[state]] = 4
    for line in np.array([['↑', '→', '↓', '←', 'x'][i]
                          for i in policy]).reshape(h, w)[::-1]:
        print(' & '.join(line), "\\\\")
    print("\\end{tabular}")
    
def latex_grid(grid):
    print(f"\\begin{{tabular}}{{{' '.join('c')*w}}}")
    policy = np.array(grid)

    for line in grid[::-1]:
        print(np.array2string(line, separator=" & ", precision=4)[1:-1], "\\\\")
    print("\\end{tabular}")