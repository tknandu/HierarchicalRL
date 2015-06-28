import numpy as np
import sys

class PCCA():

    """
    Perron Cluster Cluster Analysis
    """

    #===============================================================================
    def __init__(self, optimizeChi=False):
        self.optimizeChi = optimizeChi

    #===============================================================================
    def orthogonalize(self, eigenvalues, eigenvectors):
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
                scal = np.sum(eigenvectors[:,i])
                if np.abs(scal) > max_scal:
                    max_scal = np.abs(scal)
                    max_i = i

            # swap non-constant eigenvector
            eigenvectors[:,max_i] = eigenvectors[:,0]
            eigenvectors[:,0] = np.ones(eigenvectors.shape[1])

            # weight-orthogonalize all other eigenvectors
            for i in range(1, perron):
                for j in range(i):
                    scal = np.dot( eigenvectors[:,j], eigenvectors[:,i] ) 
                    eigenvectors[:,i] -= scal * eigenvectors[:,j]

        # normalize
        for eigvec in np.transpose(eigenvectors):
            norm = np.dot( eigvec, eigvec )
            eigvec /= np.sqrt(norm)

        eigenvectors[:,0] = np.ones(eigenvectors.shape[1])
        return eigenvectors

    #===============================================================================
    def cluster_by_isa(self, eigenvectors, n_clusters):
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


    def fill_matrix(self, rot_crop_matrix, eigvectors):

        (x, y) = rot_crop_matrix.shape

        row_sums = np.sum(rot_crop_matrix, axis=1)  
        row_sums = np.reshape(row_sums, (x,1))

        # add -row_sums as leftmost column to rot_crop_matrix 
        rot_crop_matrix = np.concatenate((-row_sums, rot_crop_matrix), axis=1 )

        tmp = -np.dot(eigvectors[:,1:], rot_crop_matrix)

        tmp_col_max = np.max(tmp, axis=0)
        tmp_col_max = np.reshape(tmp_col_max, (1,y+1))

        tmp_col_max_sum = np.sum(tmp_col_max)

        # add col_max as top row to rot_crop_matrix and normalize
        rot_matrix = np.concatenate((tmp_col_max, rot_crop_matrix), axis=0 )
        rot_matrix /= tmp_col_max_sum

        return rot_matrix

    def opt_soft(self, eigvectors, rot_matrix, n_clusters):

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
            rot_matrix = self.fill_matrix(rot_crop_matrix, eigvectors)

            result = 0
            for i in range(0, n_clusters):
                for j in range(1, n_clusters):
                    result += np.power(rot_matrix[j,i], 2) / rot_matrix[0,i]
            return(-result)


        from scipy.optimize import fmin
        rot_crop_vec_opt = fmin( susanna_func, rot_crop_vec, args=(eigvectors,) )
        
        rot_crop_matrix = np.reshape(rot_crop_vec_opt, (x, y))
        rot_matrix = self.fill_matrix(rot_crop_matrix, eigvectors)

        return(rot_matrix)



    def pcca(self, s_matrix):
    # input s_matrix corresponds to the transition matrix
    
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
        #TODO wgaps to gaps was done
        n_clusters = 7 
#        n_clusters = np.argmax(wgaps)+1
        print "\n### Maximum gap %f after top %d eigenvalues." % (np.max(gaps), n_clusters)
        print "### Maximum EV-weighted gap %f after top %d eigenvalues." % (np.max(wgaps), np.argmax(wgaps)+1)
        sys.stdout.flush()

        print "### Using %d clusters for PCCA+ ..."%n_clusters

        # orthogonalize and normalize eigenvectors 
        eigvectors = self.orthogonalize(eigvalues, eigvectors)

        # perform PCCA+
        # First two return-values "c_f" and "indicator" are not needed
        n_clusters = 8
        while n_clusters > 0:
            try:
                print n_clusters
                (chi_matrix, rot_matrix) = self.cluster_by_isa(eigvectors, n_clusters)[2:]
                break
            except np.linalg.linalg.LinAlgError:
                n_clusters -= 1


        # TODO: Is this optimization needed? What to do about the weights then?
        if self.optimizeChi:
            print "\n### Optimizing chi matrix ..."

            # perform the actual optimization
            rot_matrix = self.opt_soft(eigvectors, rot_matrix, n_clusters)

            chi_matrix = np.dot(eigvectors[:,:n_clusters], rot_matrix)

        qc_matrix = np.dot( np.dot( np.linalg.inv(rot_matrix), np.diag(eigvalues[range(n_clusters)]) ), rot_matrix ) - np.eye(n_clusters)
        cluster_weights = rot_matrix[0]
            
        print "\n### Matrix numerics check"
        print "-- Q_c matrix row sums --"
        print np.sum(qc_matrix, axis=1)
        print "-- cluster weights: first column of rot_matrix --"
        print cluster_weights
#        print "-- cluster weights: numpy.dot(node_weights, chi_matrix) --"
#        print np.dot(corr_node_weights, chi_matrix)
        print "-- chi matrix column max values --"
        print np.max(chi_matrix, axis=0)
        print "-- chi matrix row sums --"
        print np.sum(chi_matrix, axis=1)

        # store final results
        #np.savez(pool.chi_mat_fn, matrix=chi_matrix, n_clusters=n_clusters, node_names=[n.name for n in active_nodes])
        #np.savez(pool.qc_mat_fn,  matrix=qc_matrix,  n_clusters=n_clusters, node_names=[n.name for n in active_nodes], weights=cluster_weights)

        return chi_matrix
