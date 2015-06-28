import pickle
import numpy as np

print 1
f = open('transmatrix.dat','r')
print 2
unpick = pickle.Unpickler(f)
tm = unpick.load() #The transition matrix
n_states_seen = len(tm.keys())

print n_states_seen
np_tm = np.zeros((n_states_seen,n_states_seen),dtype=np.float) #The transition matrix as a numpy array 
#print np_tm
print np_tm.shape
assert False

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
            np_tm[mp_state][mp_nextstate] += tm[state][action][nextstate]

npf = open('np_tmatrix.dat','w')
pickle.dump(np_tm,npf)
