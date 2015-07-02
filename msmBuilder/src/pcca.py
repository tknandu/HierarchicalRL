import numpy as np
import sys

optimizeChi = False

#===============================================================================
def symmetrize(matrix, weights, correct_weights=False, error=1E-02):
	weights_new = weights
 
	if correct_weights:
		diff = 1		
		while diff >= error:
  			# left eigenvector of matrix yields weights_new
			weights_new = np.dot(np.transpose(matrix), weights)
			# iterate until we are below the error
			diff = np.linalg.norm(weights - weights_new, 2)
			weights = weights_new

	diff = 1
	while diff >= error:
		# scaling row sum of matrix to the respective weight yields matrix_new
		matrix_new = np.dot( np.diag( (1/np.sum(matrix, axis=1))*weights ), matrix )
		# make matrix_new symmetric
		matrix_new = 0.5*( matrix_new + np.transpose(matrix_new) )
		# iterate until we are below the error
		diff = np.linalg.norm(matrix - matrix_new, 2)
		matrix = matrix_new
		
	# make matrix_new stochastic
	matrix_new = np.dot( np.diag( (1/np.sum(matrix_new, axis=1)) ), matrix_new )
	
	return(matrix_new, weights_new)


#===============================================================================
def orthogonalize(eigenvalues, eigenvectors, weights):
	perron = 0
	# count degeneracies
	for eigval in eigenvalues:
		if eigval > 0.9999:
			perron += 1
		else:
			break

	if perron > 1:
		# look for most constant eigenvector
		max_scal = 0.0
	
		for i in range(perron):
			scal = np.dot( np.transpose(eigenvectors[:,i]), weights )
			if np.abs(scal) > max_scal:
				max_scal = np.abs(scal)
				max_i = i

		# swap non-constant eigenvector
		eigenvectors[:,max_i] = eigenvectors[:,0]
		eigenvectors[:,0] = np.ones(eigenvectors.shape[1])

		# weight-orthogonalize all other eigenvectors
		for i in range(1, perron):
			for j in range(i):
				scal = np.dot( np.dot( np.transpose(eigenvectors[:,j]), np.diag(weights) ), eigenvectors[:,i] ) 
				eigenvectors[:,i] -= scal * eigenvectors[:,j]

	# normalize
	for eigvec in np.transpose(eigenvectors):
		weighted_norm = np.dot( np.dot( np.transpose(eigvec), np.diag(weights) ), eigvec )
		eigvec /= np.sqrt(weighted_norm)

	eigenvectors[:,0] = np.ones(eigenvectors.shape[1])
	return eigenvectors

#===============================================================================
def cluster_by_isa(eigenvectors, n_clusters):
	#TODO: check this somehow, probably more args nessecary	
	# eigenvectors have to be sorted in descending order in regard to their eigenvalues
	if n_clusters > len(eigenvectors):
		n_clusters = len(eigenvectors)

    # the actual ISA algorithm
	c = eigenvectors[:, range(n_clusters)]
	ortho_sys = np.copy(c)
	max_dist = 0.0
	ind = np.zeros(n_clusters, dtype=np.int32)

	# first two representatives with maximum distance
	for (i, row) in enumerate(c):        
		if np.linalg.norm(row, 2) > max_dist:
			max_dist = np.linalg.norm(row, 2)
			ind[0] = i

	ortho_sys -= c[ind[0], None]
	
	# further representatives via Gram-Schmidt orthogonalization
	for k in range(1, n_clusters):
		max_dist = 0.0
		temp = np.copy(ortho_sys[ind[k-1]])
		

		for (i, row) in enumerate(ortho_sys):
			row -= np.dot( np.dot(temp, np.transpose(row)), temp )
			distt = np.linalg.norm(row, 2)
			if distt > max_dist:
				max_dist = distt
				ind[k] = i

		ortho_sys /= np.linalg.norm( ortho_sys[ind[k]], 2 )

	# linear transformation of eigenvectors
	rot_mat = np.linalg.inv(c[ind])
	
	chi = np.dot(c, rot_mat)

	# determining the indicator
	indic = np.min(chi)
	# Defuzzifizierung der Zugehoerigkeitsfunktionen
	#[minVal cF]=max(transpose(Chi)); #TODO minval? Marcus-check
	#minVal = np.max(np.transpose(chi))
	c_f = np.amax(np.transpose(chi))

	return (c_f, indic, chi, rot_mat)

#===============================================================================
def opt_soft(eigvectors, rot_matrix, n_clusters):

	# only consider first n_clusters eigenvectors
	eigvectors = eigvectors[:,:n_clusters]
	
	# crop first row and first column from rot_matrix
	rot_crop_matrix = rot_matrix[1:,1:]
	
	(x, y) = rot_crop_matrix.shape
	
	# reshape rot_crop_matrix into linear vector
	rot_crop_vec = np.reshape(rot_crop_matrix, x*y)

	# target function for optimization
	def susanna_func(rot_crop_vec, eigvectors):
		# reshape into matrix
		rot_crop_matrix = np.reshape(rot_crop_vec, (x, y))
		# fill matrix
		rot_matrix = fill_matrix(rot_crop_matrix, eigvectors)

		result = 0
		for i in range(0, n_clusters):
			for j in range(1, n_clusters):
				result += np.power(rot_matrix[j,i], 2) / rot_matrix[0,i]
		return(-result)


	from scipy.optimize import fmin
	rot_crop_vec_opt = fmin( susanna_func, rot_crop_vec, args=(eigvectors,) )
	
	rot_crop_matrix = np.reshape(rot_crop_vec_opt, (x, y))
	rot_matrix = fill_matrix(rot_crop_matrix, eigvectors)

	return(rot_matrix)

#===============================================================================
def calc_matrix(nodes, shift=0, cache_denom=False):
	mat = np.zeros( (len(nodes), len(nodes)) )
	for (i, ni) in enumerate(nodes):
		print("Working on: %s"%ni)
		if(cache_denom):
			phi_denom = get_phi_denom(ni.trajectory, nodes)
		frame_weights = ni.frameweights
		if shift > 0:
			frame_weights = frame_weights[:-shift]
		for (j, nj) in enumerate(nodes):
			if(cache_denom):
				mat[i, j] = np.average(get_phi_num(ni.trajectory, nj)[shift:] / phi_denom[shift:], weights=frame_weights)
			else:
				mat[i, j] = np.average(get_phi(ni.trajectory, nj)[shift:], weights=frame_weights)
	return(mat)

################################## Main Code ###################################
#===============================================================================

# Get s_matrix
s_matrix = np.ones((100,100))

# calculate and sort eigenvalues in descending order
(eigvalues, eigvectors) = np.linalg.eig(s_matrix)
argsorted_eigvalues = np.argsort(-eigvalues)
eigvalues = eigvalues[argsorted_eigvalues]
eigvectors = eigvectors[:, argsorted_eigvalues]
	
gaps = np.abs(eigvalues[1:]-eigvalues[:-1])
gaps = np.append(gaps, 0.0)
wgaps = gaps*eigvalues

print "\n### Sorted eigenvalues of symmetrized S matrix:"
for (idx, ev, gap, wgap) in zip(range(1, len(eigvalues)+1), eigvalues, gaps, wgaps):
	print "EV%04d: %f, gap to next: %f, EV-weighted gap to next: %f" % (idx, ev, gap, wgap)
n_clusters = np.argmax(wgaps)+1
print "\n### Maximum gap %f after top %d eigenvalues." % (np.max(gaps), n_clusters)
print "### Maximum EV-weighted gap %f after top %d eigenvalues." % (np.max(wgaps), np.argmax(wgaps)+1)
sys.stdout.flush()

print "### Using %d clusters for PCCA+ ..."%n_clusters

# orthogonalize and normalize eigenvectors 
eigvectors = orthogonalize(eigvalues, eigvectors, corr_node_weights)

# perform PCCA+
# First two return-values "c_f" and "indicator" are not needed
(chi_matrix, rot_matrix) = cluster_by_isa(eigvectors, n_clusters)[2:]

if optimizeChi:
	print "\n### Optimizing chi matrix ..."
		
	outliers = 5
	mean_weight = np.mean(corr_node_weights)
	threshold = mean_weight/100*outliers
	print "Light-weight node threshold (%d%% of mean corrected node weight): %.4f."%(outliers, threshold)

	# accumulate nodes for optimization
	edges = np.where(np.max(chi_matrix, axis=1) > 0.9999)[0] # edges of simplex
	heavies = np.where( corr_node_weights > threshold)[0] # heavy-weight nodes
	filtered_eigvectors = eigvectors[ np.union1d(edges, heavies) ]

	# perform the actual optimization
	rot_matrix = opt_soft(filtered_eigvectors, rot_matrix, n_clusters)

	chi_matrix = np.dot(eigvectors[:,:n_clusters], rot_matrix)
		
	# deal with light-weight nodes: shift and scale
	for i in np.where(corr_node_weights <= threshold)[0]:
		if(i in edges):
			print "Column %d belongs to (potentially dangerous) light-weight node, but its node is a simplex edge."%(i+1)
			continue
		print "Column %d is shifted and scaled."%(i+1)
		col_min = np.min( chi_matrix[i,:] )
		chi_matrix[i,:] -= col_min
		chi_matrix[i,:] /= 1-(n_clusters*col_min)
			
qc_matrix = np.dot( np.dot( np.linalg.inv(rot_matrix), np.diag(eigvalues[range(n_clusters)]) ), rot_matrix ) - np.eye(n_clusters)
cluster_weights = rot_matrix[0]
	
print "\n### Matrix numerics check"
print "-- Q_c matrix row sums --"
print np.sum(qc_matrix, axis=1)
print "-- cluster weights: first column of rot_matrix --"
print cluster_weights
print "-- cluster weights: numpy.dot(node_weights, chi_matrix) --"
print np.dot(corr_node_weights, chi_matrix)
print "-- chi matrix column max values --"
print np.max(chi_matrix, axis=0)
print "-- chi matrix row sums --"
print np.sum(chi_matrix, axis=1)

# store final results
#np.savez(pool.chi_mat_fn, matrix=chi_matrix, n_clusters=n_clusters, node_names=[n.name for n in active_nodes])
#np.savez(pool.qc_mat_fn,  matrix=qc_matrix,  n_clusters=n_clusters, node_names=[n.name for n in active_nodes], weights=cluster_weights)

