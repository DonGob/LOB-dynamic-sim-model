import numpy as np

def thinning_algo(events, states, intensity_funcs, trans_mtrx, n_events = 4):
    lambda_star = 10
    selected_t_list = np.zeros(4)
    for k in range(n_events):
        t_k = events[-1][0]
        for i in range(100000): #arbitrary large number 
            delta_t = np.random.exponential(lambda_star)
            u = np.random.uniform(0,1)
            t_k = t_k + delta_t
            lambda_k = intensity_funcs[k](t_k)
            if lambda_k/lambda_star > u:
                selected_t_list[k] = t_k
                break
    tj = selected_t_list.min()
    yj = selected_t_list.argmin()
    ej = (tj, yj)
    xjmin1 = states[-1]
    xj = sample_transition_matrix(trans_mtrx, yj, xjmin1)
    return ej, xj

def sample_transition_matrix(mtrx, event, prev_state):
    probabilities = mtrx[event][prev_state]
    choice = np.random.choice(3, 10000000, p=probabilities)
    return choice
        