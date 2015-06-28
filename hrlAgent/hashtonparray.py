import pickle
import numpy as np

f = open('transmatrix.dat','r')
unpick = pickle.Unpickler(f)
tm = unpick.load() #The transition matrix
n_states_seen = len(tm.keys())
n_actions = 12

print n_states_seen
np_tm = np.zeros((n_states_seen,n_actions,n_states_seen),dtype=np.int) #The transition matrix as a numpy array 
#print np_tm
print np_tm.shape

state_i = 0
state_mapping = {}
for state in tm.keys():
    state_mapping[state] = state_i
    state_i += 1

for state in tm.keys():
    mp_state = state_mapping[state] #State mapped to an int
    for action in tm[state].keys():
        for nextstate in tm[state][action].keys():
            mp_nextstate = state_mapping[nextstate]
            np_tm[mp_state][action][mp_nextstate] = tm[state][action][nextstate]

npf = open('nptransmatrix.dat','w')
pickle.dump(np_tm,npf)
