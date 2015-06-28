import pickle
import matplotlib.pyplot as plt

f = open('normalumat1000.dat','r')
(steps,returns) = pickle.load(f)

results2 = []
for i in range(100,len(returns)):
    results2.append(sum(returns[i-100:i])/100.0)


plt.plot(results2)
plt.xlabel('Episode Number')
plt.ylabel('Returns')
plt.legend()
plt.show()
