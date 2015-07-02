import numpy as np

traj = [np.array([1,2,3]),np.array([1,2])]

t = np.array(traj)

t.append(np.array([1]))

print t