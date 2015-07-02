import numpy as np

class NeuralNet:

    def __init__(self, layers, epsilon=0.12, learningRate=0.1):
        '''
        Constructor
        Arguments:
        	layers - a numpy array of L integers (L is # layers in the network)
        	epsilon - one half the interval around zero for setting the initial weights
        	learningRate - the learning rate for backpropagation
        '''
        self.layers = layers
        self.learningRate = learningRate
        self.thetas = []
        for l in xrange(len(self.layers)-1):
            tht = np.random.uniform(low=-1.0, high=1.0, size=(self.layers[l+1],self.layers[l]+1))*epsilon
            self.thetas = np.append(self.thetas, np.ravel(tht))

    def propagateAndUpdate(self, x, y):
        '''
        Used to forward propagate a prediction based on input x, and update against truth y
        '''
        a = self.propagate(x)
        self.update(x, y, a)

    def update(self, x, y, a):
        '''
        Used to backpropagate the and correct based on an input x, prediction a, and truth y
        '''
        #Loop over layers backwards
        tht_tot = 0
        tot = self.layers[len(self.layers)-1]
        d = a[len(a)-tot:] - y
        d[y==0.0] == 0.0 #IMPORTANT: This step makes us only update the action that we observed
        deltas = np.zeros(len(self.thetas))
        for l in reversed(xrange(len(self.layers)-1)):
            size1 = self.layers[l]                    
            size2 = self.layers[l+1]
            if (l==0):
                a_t = np.append(1.0, x)
            else:
                a_t = np.append(1.0, a[len(a)-tot-size1:len(a)-tot])
            tot += size1                    
            deltas[len(deltas)-tht_tot-(size1+1)*size2:len(deltas)-tht_tot] += np.ravel(np.outer(d, a_t))
            if (l==0):
                break
            tht = self.thetas[len(self.thetas)-tht_tot-(size1+1)*size2:len(self.thetas)-tht_tot]
            tht_tot += (size1+1)*size2
            g = np.multiply(a_t, 1-a_t)
            d = np.multiply(np.dot(np.reshape(tht,(size2,size1+1)).T,d), g)[1:]
        self.thetas -= self.learningRate*deltas

    def propagate(self, x):
        '''
        Used the model to predict weighted output values for instance x
        Arguments:
            x is a d-dimenstional numpy array
        Returns:
            a c-dimensional numpy array of the strength of each output
        '''
        temp2 = x
        total = []
        total_len = 0
        for i in xrange(len(self.layers)-1):
            temp = temp2
            size1 = self.layers[i]
            size2 = self.layers[i+1]
            tht = np.reshape(self.thetas[total_len:total_len+((size1+1)*size2)], (size2, size1+1))
            temp2 = np.dot(tht, np.append(1.0, temp))
            total_len += (size1+1) * size2
            total = np.append(total, temp2)
        return 1.0/(1.0+np.exp(-total))        
    
