import pickle

f = open('chi_mat.dat','r')
a = pickle.Unpickler(f)
chi_mat = a.load()

maxes = []
maxases = []
argmaxes = []
for i in range(8):
    maxases.append([])
for (row_i,row) in enumerate(chi_mat):
    maxes.append(max(row))
    argmaxes.append(row.argmax())
    maxases[row.argmax()].append(row_i)

print chi_mat
print chi_mat.shape

print maxes
print maxases
print [len(l) for l in maxases]
print argmaxes
