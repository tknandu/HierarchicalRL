from li_nn import NeuralNet
# from nn import NeuralNet
import random
import numpy as np

class T_Experience():
    def __init__(self, s1, s2, t):
        '''
        Need to check necessity of Experience Replay for Transition Network too - anyway going ahead with it for now
        '''
        self.s1 = s1
        self.s2 = s2
        self.t = t

class TNN():
    """
    Transition Neural Network. Includes experience replay (interesting to see how it will work for Transition network da)!
    
    ouput layer has 1 node - transition probability
    input_size: the number of inputs (2* size of state representation)
    max_experiences: the total number of experiences to save for replay
    gamma: future rewards discount rate
    alpha: learning rate for underlying NN
    use_sarsa: flag whether to use the SARSA update rule
    """
    
    def __call__(self,s1,s2):
        """ implement here the returned T(s1,s2)
        """
        return self.GetValue(s1,s2)

    def __init__(self, input_size, max_experiences=500, alpha=0.1):
        # lay = [input_size, int((nactions+input_size)/2.0), nactions]
        lay = [input_size, int((1+input_size)/2.0), 1]
        self.NN = NeuralNet(layers=lay, epsilon=0.154, learningRate=alpha)
        self.experiences = []
        self.max_experiences = max_experiences
        self.prob_remember = 0.1
        self.num_replay_samples = 10

    def GetValue(self, s1, s2):
        """ Return the T(s_1,s_2)
        """
        out = self.NN.propagate(np.concatenate((s1,s2)))
        return out[a]

    def Update(self, s1, s2, t):
        """ update transition prob t
        """
        a = np.zeros(1)
        a[0] = t
        self.NN.propagateAndUpdate(np.concatenate((s1,s2)),a)

    def RememberExperience(self, s1, s2, t):
        if (random.random() < self.prob_remember):
            if (len(self.experiences) >= self.max_experiences):
                #TODO: Something more intelligent about how we determine what is worth forgetting
                self.experiences.pop(random.randint(0, self.max_experiences-1))
            self.experiences.append(T_Experience(s1, s2, t))

    def ExperienceReplay(self):
        #Skip until we have enough experience
        if (len(self.experiences) < self.num_replay_samples):
            return
        for i in xrange(self.num_replay_samples):
            index = random.randint(0, len(self.experiences)-1)
            exp = self.experiences[index]
            self.Update(exp.s1, exp.s2, exp.t)