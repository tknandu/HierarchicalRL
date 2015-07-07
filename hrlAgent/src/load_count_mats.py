import pickle
import numpy as np
import sys

n_eps = 1000

f = open('phi_mat' + str(n_eps) + '.dat','r')
unpickler = pickle.Unpickler(f)
phi_mat = unpickler.load()

f = open('u_mat' + str(n_eps) + '.dat','r')
unpickler = pickle.Unpickler(f)
u_mat = unpickler.load()

f = open('peeyush_u_mat' + str(n_eps) + '.dat','r')
unpickler = pickle.Unpickler(f)
pu_mat = unpickler.load()

n_valid_states = len(phi_mat.keys())
valid_states = phi_mat.keys()
valid_states.sort()
n_actions = 12
print valid_states
print n_valid_states

f = open('valid_states' + str(n_eps) + '.dat','w')
pickle.dump(valid_states,f)

#Using the matrices from page 39 of thesis
d_mat = np.zeros((n_valid_states,n_valid_states),dtype=float)
for (i_1,state_1) in enumerate(valid_states):
    for (i_2,state_2) in enumerate(valid_states):
        total = 0
        for action in range(n_actions):
            if action in u_mat[state_1]:
                if state_2 in u_mat[state_1][action]:
                    total += u_mat[state_1][action][state_2]
        d_mat[i_1][i_2] = float(total)

f = open('sparse_d_mat' + str(n_eps) + '.dat','w')
pickle.dump(d_mat,f)

p_mat = np.zeros((n_valid_states,n_actions,n_valid_states),dtype=float)
for (i_1,state_1) in enumerate(valid_states):
    for action in range(n_actions):
        if action in u_mat[state_1]:
            total = 0
            for (i_2, state_2) in enumerate(valid_states):
                if state_2 in u_mat[state_1][action]:
                    total += u_mat[state_1][action][state_2]
            if total == 0:
                p_mat[i_1][action][i_1] = 1 #If we have no transitions for (s_1,a), then set p(s_1,a,s_1) to 1 and let the others be 0
            else:
                for (i_2,state_2) in enumerate(valid_states):
                    if state_2 in u_mat[state_1][action]:
                        p_mat[i_1][action][i_2] = u_mat[state_1][action][state_2]/float(total)
        else:
            p_mat[i_1][action][i_1] = 1

f = open('sparse_p_mat' + str(n_eps) + '.dat','w')
pickle.dump(p_mat,f)

t_mat = np.zeros((n_valid_states,n_valid_states),dtype=float)
for (i_1,state_1) in enumerate(valid_states):
    total = 0
    for (i_2, state_2) in enumerate(valid_states):
        total += d_mat[i_1][i_2]
    if total == 0:
        t_mat[i_1][i_1] = 1 #If we have no transitions out of s_1, then we set p(s_1,s_1) to 1 and let the others be 0
    else:
        for (i_2,state_2) in enumerate(valid_states):
            t_mat[i_1][i_2] = d_mat[i_1][i_2]/float(total)

f = open('sparse_t_mat' + str(n_eps) + '.dat','w')
pickle.dump(t_mat,f)
