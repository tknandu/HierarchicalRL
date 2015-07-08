import random
import time
import string
import pickle
from rlglue.agent.Agent import Agent
from rlglue.types import Action
from rlglue.types import Observation
from rlglue.agent import AgentLoader as AgentLoader
import numpy as np
from qnn import QNN
from pcca import PCCA
from deep_qnn import Deep_QNN
from tnn import TNN
import math
import copy
from operator import itemgetter
import scipy.sparse
import bisect

class Monster:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.sx = 0
        self.sy = 0
        self.type = -1
        self.typeName = ""
        self.winged = False

monsterNames = ["Mario", "Red Koopa", "Green Koopa", "Goomba", "Spikey", "Piranha Plant", "Mushroom", "Fire Flower", "Fireball", "Shell", "Big Mario", "Fiery Mario"]

'''
* Valid tiles:
* M - the tile mario is currently on. there is no tile for a monster.
* $ - a coin
* b - a smashable brick
* ? - a question block
* | - a pipe. gets its own tile because often there are pirahna plants
*     in them
* ! - the finish line
* And an integer in [1,7] is a 3 bit binary flag
*  first bit is "cannot go through this tile from above"
*  second bit is "cannot go through this tile from below"
*  third bit is "cannot go through this tile from either side"
'''
#This is the encoder for the single layer state
tileEncoder = {'\0': -1, '1': -1, '2': -1, '3': -1, '4': -1, '5': -1, '6': -1, '7': -1, 'M': 0, '$': 2, 'b': 2, '?': 2, '|': 2, '!': 2, ' ': 1, '\n': 1}

#These are the encoders for the multiple substrates state
backgroundLayer = {'\0': 2, '1': 1, '2': 1, '3': 1, '4': 1, '5': 1, '6': 1, '7': 2, 'M': 0, '$': 0, 'b': 0, '?': 0, '|': 0, '!': 0, ' ': 0, '\n': 0}
rewardLayer = {'\0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, 'M': 0, '$': 1, 'b': 1, '?': 1, '|': 1, '!': 2, ' ': 0, '\n': 0}

class MarioAgent(Agent):

    # related to options
    q_stepsize = 0.1
    q_epsilon = 0.1
    q_gamma = 0.9
    value_function = None
    optionCurrentlyOn = False
    normalizationC = 0.5
    currentOptionTime = 0
    currentOptionReward = 0.0
    numActions = 12
    randGenerator=random.Random()


    def agent_init(self,taskSpecString):
        self.policy_frozen = False
        self.total_steps = 0
        self.trial_start = 0.0
        self.total_steps = 0
        self.step_number = 0
        self.exp = 0.75 # Exploration factor, reset from param
        self.substrate_mode = False
        if (self.substrate_mode):
            self.state_dim_x = 33
            self.state_dim_y = 7
        else:
            self.state_dim_x = 20
            self.state_dim_y = 12

        self.last_state = None
        self.last_action = None
        random.seed(0)
        
        self.Q = Deep_QNN(nactions=12, input_size=(self.state_dim_x*self.state_dim_y), max_experiences=500, gamma=0.6, alpha=0.2)

        #####
        self.regularization_constant = 0.4 # For rewards incorporated into transition structure
        self.episodesRun = 100000 # This is just used to generate the file names of the stored data to be read in
        ######

        self.n_bins = 1000 # filled in from experiment.py
        self.n_hidden_layer_outputs = 1 # filled in from experiment.py
        self.option_learning_frozen = True # filled in from experiment.py

        self.discretization_done = False
        self.n_disc_states = math.pow((self.n_bins + 3),self.n_hidden_layer_outputs)
        self.phi_mat = {}
        self.U_mat = {}
        self.Peey_U_mat = {} # The (probably) incorrect U_mat that Peeyush writes in his algo
        self.seen_expanded_states = {} # A dictionary of all the non-approximated (original Mario input) states that were seen.
        self.last_enc_state = None
        self.last_enc_action = None

        #####################################################################

        # When learning to play with options
        if self.option_learning_frozen==False:
            # Obtain T & P matrices
            tmatfile = open('t_mat' + str(self.episodesRun) + '.dat','r')
            unpickler = pickle.Unpickler(tmatfile)
            self.t_mat = unpickler.load()

            pmatfile = open('p_mat' + str(self.episodesRun) + '.dat','r')
            unpickler = pickle.Unpickler(pmatfile)
            self.p_mat = unpickler.load()

            validstatesfile = open('valid_states'+str(self.episodesRun)+'.dat','r')
            unpickler = pickle.Unpickler(validstatesfile)
            self.valid_states = unpickler.load()

            # Run PCCA to obtain Chi matrix
            self.clusterer = PCCA(True)
            self.chi_mat = np.mat(self.clusterer.pcca(self.t_mat))

            #Converting to np.mat format for consistency with 3 room code. Doing it only here because pcca needs it as ndarray
            self.t_mat = np.mat(self.t_mat)

            # 0,1,2,..11 - primitive actions, 12... - options
            self.value_function=[(self.chi_mat.shape[1]+self.numActions)*[0.0] for i in range(self.chi_mat.shape[0])]

            self.absStateMembership = []
            self.statesInAbsState = [[] for i in xrange(self.chi_mat.shape[1])]
            for (row_i,row) in enumerate(self.chi_mat):
                self.absStateMembership.append(row.argmax())
                self.statesInAbsState[row.argmax()].append(row_i)

            self.connect_mat = self.chi_mat.T*self.t_mat*self.chi_mat

    def egreedy(self, state):
        maxIndex=0
        a=1
        if self.randGenerator.random()<self.q_epsilon:
            randact = self.randGenerator.randint(0,self.chi_mat.shape[1]+self.numActions-1)
            if randact < self.numActions:
                return self.actionEncoder(self.randomAction(True))
            else:
                return randact

        temp = [(i,v) for i,v in enumerate(self.value_function[state])]
        random.shuffle(temp)
        a = max(temp,key=itemgetter(1))[0]
        print self.value_function[state]
        return a

    def posGradientAction(self,s):
        startAction = 4
        thisIntAction=startAction
        maxVal = -np.inf
        for a in xrange(self.numActions-startAction): 
            print 'Action: ',a+startAction,' ',max(self.normalizationC*(np.sum(np.dot(np.array(self.p_mat[s][a+startAction]),np.array(self.chi_mat.T[self.option_S_j].T))) - self.chi_mat[s,self.option_S_j]),0)
            action_pref = max(self.normalizationC*(np.sum(np.dot(np.array(self.p_mat[s][a+startAction]),np.array(self.chi_mat.T[self.option_S_j].T))) - self.chi_mat[s,self.option_S_j]),0)
            if action_pref > maxVal:
                thisIntAction = a+startAction
                maxVal = action_pref
        print 'Option chosen'
        return thisIntAction        

    def nextAction(self,state):
        # Choose either a primitive action or option
        a = self.egreedy(state)

        if a<self.numActions:
            # Primitive action
            thisIntAction = a
            self.optionCurrentlyOn = False
            print 'Primitive action chosen'

        else:    
            # Composing an option from S_i to S_j
            self.optionCurrentlyOn = True
            self.currentOptionTime = 0
            self.curentOptionStartState = state
            self.currentOptionReward = 0.0

                # 1. Find the abstract state you belong to & going to
            self.option_S_i = self.absStateMembership[state] # initiation step
            self.option_S_j = a-self.numActions # actually, we will have to choose S_j based on SMDP

            # 2. Choose action based on membership ascent
            thisIntAction = self.posGradientAction(state)
            self.currentOptionTime += 1
        return thisIntAction

    def agent_start(self,observation):
        self.step_number = 0
        self.trial_start = time.clock()

        # When learning to play with options
        if self.option_learning_frozen == False:
            self.optionCurrentlyOn = False
            theState=self.getDiscretizedState(tuple(self.Q.getHiddenLayerRepresentation(self.stateEncoder(observation))))
            s = theState #valid_states is not needed in Mario
            
            thisIntAction = self.nextAction(s)

            print 'Action chosen: ',thisIntAction
            act=self.actionDecoder(thisIntAction)

        # When learning transition probabilities
        if self.transition_learning_frozen == False:
            act = self.getAction(observation)
            if self.discretization_done:
                enc_state = tuple(self.Q.getHiddenLayerRepresentation(self.stateEncoder(observation)))
                self.last_disc_state = self.getDiscretizedState(enc_state)
                self.last_enc_action = self.actionEncoder(act)
                if self.last_disc_state not in self.phi_mat:
                    self.phi_mat[self.last_disc_state] = {}
                if self.last_disc_state not in self.U_mat:
                    self.U_mat[self.last_disc_state] = {}
                if self.last_disc_state not in self.Peey_U_mat:
                    self.Peey_U_mat[self.last_disc_state] = {}

        # When using QNN
        if self.policy_frozen == False:
            self.seen_expanded_states[tuple(self.stateEncoder(observation))] = True
            act = self.getAction(observation)

        # When extracting trajectories to fit MSM
        if self.trajectory_learning_frozen == False:
            act = self.getAction(observation)
            if self.discretization_done:
                enc_state = tuple(self.Q.getHiddenLayerRepresentation(self.stateEncoder(observation)))
                self.last_disc_state = self.getDiscretizedState(enc_state)
                self.last_enc_action = self.actionEncoder(act)
            print 'Trajectory: '+str(self.last_disc_state)+',',

        self.last_action = copy.deepcopy(act)
        self.last_state  = copy.deepcopy(observation)

        return act
 
    def agent_step(self,reward, observation):
        self.step_number += 1
        self.total_steps += 1

        # When learning to play with options
        if self.option_learning_frozen == False:        
            newState = self.getDiscretizedState(tuple(self.Q.getHiddenLayerRepresentation(self.stateEncoder(observation))))
            lastState = self.getDiscretizedState(tuple(self.Q.getHiddenLayerRepresentation(self.stateEncoder(self.last_state))))
            lastAction = self.actionEncoder(self.last_action)

            s = newState

            # Check if an option is going on
            if self.optionCurrentlyOn:

                # add reward to ongoing option
                self.currentOptionReward += reward

                # Decide whether to terminate option
                try:
                    beta = min(math.log(abs(self.chi_mat[s,self.option_S_i]))/math.log(abs(self.chi_mat[s,self.option_S_j])),1)
                except ValueError:
                    print self.chi_mat[s,self.option_S_i]
                    print self.chi_mat[s,self.option_S_j]
                    assert False
                if self.randGenerator.random() < beta:
                    self.optionCurrentlyOn = False
                    print 'Terminated option'

                    # Update Q value of terminated option
                    Q_sa=self.value_function[self.curentOptionStartState][self.numActions+self.option_S_j] # 4... - options
                    max_Q_sprime_a = max(self.value_function[newState])     
                    new_Q_sa=Q_sa + self.q_stepsize  * (self.currentOptionReward + math.pow(self.q_gamma,self.currentOptionTime) * max_Q_sprime_a - Q_sa)
                    self.value_function[self.curentOptionStartState][self.numActions+self.option_S_j]=new_Q_sa                
                        
                    # Pick a new primitive action/option
                    newIntAction = self.nextAction(s)

                else:
                    # If not terminated, choose action based on membership ascent
                    print 'Not terminated option, still searching.'
                    newIntAction = self.posGradientAction(s)
                    self.currentOptionTime += 1

            # No option currently running
            else:

                # update Q-value of last primitive action
                Q_sa=self.value_function[lastState][lastAction]
                max_Q_sprime_a = max(self.value_function[newState])     
                new_Q_sa=Q_sa + self.q_stepsize  * (reward + self.q_gamma * max_Q_sprime_a - Q_sa)
                self.value_function[lastState][lastAction]=new_Q_sa

                # pick either a new primitive action/option
                newIntAction = self.nextAction(s)
            
            print 'Action chosen: ',newIntAction

            act = self.actionDecoder(newIntAction)

        # When learning transition probabilities
        if self.transition_learning_frozen == False:
            act = self.getAction(observation)
            enc_state = tuple(self.Q.getHiddenLayerRepresentation(self.stateEncoder(observation)))
            disc_state = self.getDiscretizedState(enc_state)
            enc_action = self.actionEncoder(act)        
            if disc_state not in self.phi_mat:
                self.phi_mat[disc_state] = {}
            if disc_state not in self.U_mat:
                self.U_mat[disc_state] = {}
            if disc_state not in self.Peey_U_mat:
                self.Peey_U_mat[disc_state] = {}
 
            if self.last_enc_action not in self.phi_mat[self.last_disc_state]:
                self.phi_mat[self.last_disc_state][self.last_enc_action] = {}
                self.phi_mat[self.last_disc_state][self.last_enc_action][disc_state] = 1
            else:
                if disc_state not in self.phi_mat[self.last_disc_state][self.last_enc_action]:
                    self.phi_mat[self.last_disc_state][self.last_enc_action][disc_state] = 1
                else:
                    self.phi_mat[self.last_disc_state][self.last_enc_action][disc_state] += 1
            if self.last_enc_action not in self.U_mat[self.last_disc_state]:
                self.U_mat[self.last_disc_state][self.last_enc_action] = {}
                self.U_mat[self.last_disc_state][self.last_enc_action][disc_state] = math.exp(-self.regularization_constant*abs(reward))
            else:
                if disc_state not in self.U_mat[self.last_disc_state][self.last_enc_action]:
                    self.U_mat[self.last_disc_state][self.last_enc_action][disc_state] = math.exp(-self.regularization_constant*abs(reward))
                else:
                    self.U_mat[self.last_disc_state][self.last_enc_action][disc_state] += math.exp(-self.regularization_constant*abs(reward))
            if self.last_enc_action not in self.Peey_U_mat[self.last_disc_state]:
                self.Peey_U_mat[self.last_disc_state][self.last_enc_action] = {}
                self.Peey_U_mat[self.last_disc_state][self.last_enc_action][disc_state] = self.phi_mat[self.last_disc_state][self.last_enc_action][disc_state]*math.exp(-self.regularization_constant*abs(reward))
            else:
                if disc_state not in self.Peey_U_mat[self.last_disc_state][self.last_enc_action]:
                    self.Peey_U_mat[self.last_disc_state][self.last_enc_action][disc_state] = self.phi_mat[self.last_disc_state][self.last_enc_action][disc_state]*math.exp(-self.regularization_constant*abs(reward))
                else:
                    self.Peey_U_mat[self.last_disc_state][self.last_enc_action][disc_state] += self.phi_mat[self.last_disc_state][self.last_enc_action][disc_state]*math.exp(-self.regularization_constant*abs(reward))

            self.last_enc_state = enc_state
            self.last_enc_action = enc_action
            self.last_disc_state = disc_state

        # When using QNN
        if self.policy_frozen == False:
            act = self.getAction(observation)    
            self.seen_expanded_states[tuple(self.stateEncoder(observation))] = True
            self.update(observation, act, reward)

        # When extracting trajectories to fit MSM
        if self.trajectory_learning_frozen == False:
            act = self.getAction(observation)
            enc_state = tuple(self.Q.getHiddenLayerRepresentation(self.stateEncoder(observation)))
            self.last_disc_state = self.getDiscretizedState(enc_state)
            self.last_enc_action = self.actionEncoder(act)
            print str(self.last_disc_state)+',',

        self.last_action = copy.deepcopy(act)
        self.last_state  = copy.deepcopy(observation)    

        return act
    
    def agent_end(self,reward):
        time_passed = time.clock() - self.trial_start
        print "ended after " + str(self.total_steps) + " total steps"
        print "average " + str(self.step_number/time_passed) + " steps per second"
    
        # When learning to play using options
        if self.option_learning_frozen == False:
            lastState = self.getDiscretizedState(tuple(self.Q.getHiddenLayerRepresentation(self.stateEncoder(self.last_state))))
            lastAction = self.actionEncoder(self.last_action)

            if self.optionCurrentlyOn:
                self.currentOptionReward += reward
                # Update Q value of terminated option
                Q_sa=self.value_function[self.curentOptionStartState][self.numActions+self.option_S_j] # 4... - options 
                new_Q_sa=Q_sa + self.q_stepsize  * (self.currentOptionReward - Q_sa)
                self.value_function[self.curentOptionStartState][self.numActions+self.option_S_j]=new_Q_sa                
            else:
                # update Q-value of last primitive action
                Q_sa=self.value_function[lastState][lastAction]    
                new_Q_sa=Q_sa + self.q_stepsize  * (reward - Q_sa)
                self.value_function[lastState][lastAction]=new_Q_sa

        # When extracting trajectories to fit MSM
        if self.trajectory_learning_frozen == False:
            print ''

    def agent_cleanup(self):
        pass
    
    def agent_freeze(self):
        pass
    
    def agent_message(self,inMessage):
        if inMessage.startswith("freeze_learning"):
            self.policy_frozen=True
            return "message understood, policy frozen"
        if inMessage.startswith("unfreeze_learning"):
            self.policy_frozen=False
            return "message understood, policy unfrozen"
        if inMessage.startswith("set_exploring"):
            splitString=inMessage.split(" ")
            self.exp = float(splitString[1])
            return "message understood, setting exploration factor"
        if inMessage.startswith("save_policy"):
            splitString=inMessage.split(" ")
            self.saveQFun(splitString[1])
            print "Q function saved."
            return "message understood, saving policy"
        if inMessage.startswith("load_policy"):
            splitString=inMessage.split(" ")
            self.loadQFun(splitString[1])
            print "Q function loaded."
            return "message understood, loading policy"
        if inMessage.startswith("use_impactful_experiences"):
            self.Q.use_impactful = True
            return "message understood, using impactful experiences"
        if inMessage.startswith("use_all_experiences"):
            self.Q.use_impactful = False
            return "message understood, using all experiences"
        if inMessage.startswith("reset_q"):
            self.Q = QNN(nactions=12, input_size=(self.state_dim_x*self.state_dim_y), max_experiences=500, gamma=0.6, alpha=0.2)
            return "message understood, reseting q-function"

        # Added now
        if inMessage.startswith("freeze_transition_learning"):
            self.transition_learning_frozen=True
            return "message understood, transition learning frozen"    
        if inMessage.startswith("unfreeze_transition_learning"):
            self.transition_learning_frozen=False
            return "message understood, transition learning unfrozen"        
        if inMessage.startswith("freeze_option_learning"):
            self.option_learning_frozen=True
            return "message understood, option learning frozen"    
        if inMessage.startswith("unfreeze_option_learning"):
            self.option_learning_frozen=False

        # Save the tuples corresponding to each state.
        if inMessage.startswith("save_state_reps"):
            enc_states = []
            print len(self.seen_expanded_states.keys())
            for state in self.seen_expanded_states.keys():
                enc_states.append(tuple(self.Q.getHiddenLayerRepresentation(list(state))))
            splitstring = inMessage.split()
            outfile = open(splitstring[1],'w')
            pickle.dump(enc_states,outfile)

        # Once we have frozen our state reps, this function discretizes them and populates the self.colBins
        if inMessage.startswith("save_bins_from_state_reps"):
            enc_states = []
            for state in self.seen_expanded_states.keys():
                curr_rep = tuple(self.Q.getHiddenLayerRepresentation(list(state)))
                curr_enc_state = []
                for curr_rep_i in range(1,len(curr_rep)):
                    curr_enc_state.append(curr_rep[curr_rep_i][0])
                enc_states.append(curr_enc_state)
            enc_states = np.array(enc_states)
            self.colBins = []
            for col_i in range(enc_states.shape[1]):
                curr_col = enc_states[:,col_i]
                self.colBins.append(self.getBins(curr_col))

            splitstring = inMessage.split()
            outfile = open(splitstring[1],'w')
            pickle.dump(self.colBins,outfile)
            self.discretization_done = True

        if inMessage.startswith("get_bins_from_state_reps"):
            splitstring = inMessage.split()
            colfile = open(splitstring[1],'r')
            unpickler = pickle.Unpickler(colfile)
            self.colBins = unpickler.load()
            self.discretization_done = True
            return "message understood, loading discretization"

        if inMessage.startswith("save_phi"):
            splitstring = inMessage.split()
            phioutfile = open(splitstring[1],'w')
            pickle.dump(self.phi_mat,phioutfile)

            uoutfile = open(splitstring[2],'w')
            pickle.dump(self.U_mat,uoutfile)

            puoutfile = open(splitstring[3],'w')
            pickle.dump(self.Peey_U_mat,puoutfile)

        if inMessage.startswith("load_vf"):
            splitstring = inMessage.split()
            theFile = open('valuefunction'+str(splitstring[1])+'.dat', "r")
            self.value_function = pickle.load(theFile)
            theFile.close()
        if inMessage.startswith("save_vf"):
            vffile = open('valuefunction'+str(splitstring[1])+'.dat', "w")
            pickle.dump(self.value_function,vffile)
            vffile.close()

        if inMessage.startswith("freeze_trajectory"):
            self.trajectory_learning_frozen=True
            return "message understood, trajectory learning frozen"
        if inMessage.startswith("unfreeze_trajectory"):
            self.trajectory_learning_frozen=False
            return "message understood, trajectory learning unfrozen"

        if inMessage.startswith("set_n_bins"):
            splitstring = inMessage.split()
            self.n_bins = int(splitstring[1])
            return "message understood, no of bins set"

        if inMessage.startswith("set_n_hidden_layer_nodes"):
            splitstring = inMessage.split()
            self.n_hidden_layer_outputs = int(splitstring[1])
            return "message understood, no of hidden layer nodes set"

        return None


    # Get the discretized states. Assumes that the bins are available.
    def getDiscretizedState(self,enc_state):
        
        disc_tuple = []
        for i in range(0,len(enc_state)-1):
            curr_elem = enc_state[i+1][0]
            disc_tuple.append(np.digitize([curr_elem],self.colBins[i])[0])
        
        flat_disc_state = 0
        mult = 1
        if len(set(disc_tuple)) > 1:
            print 'HEREE'
        for elem in disc_tuple:
            flat_disc_state += elem*mult
            mult *= (self.n_bins+3)
        if self.option_learning_frozen==False:
            try:
                return self.valid_states.index(flat_disc_state) 
            except ValueError:
                i = bisect.bisect_left(self.valid_states,flat_disc_state)-1
                if i < 0:
                    i = 0
                return i
        else:
            return flat_disc_state

    def getBins(self,i_array):
        bins = [0.0]
        i_array.sort()

        i = 0.0
        while i < len(i_array):
            bins.append(i_array[math.floor(i)])
            i += float(len(i_array))/float(self.n_bins)
        bins.append(1.0)
        return np.array(bins)
                
    def getMonsters(self, observation):
        monsters = []
        i = 0
        while (1+2*i < len(observation.intArray)):
            m = Monster()
            m.type = observation.intArray[1+2*i]
            m.winged = (observation.intArray[2+2*i]) != 0
            m.x = observation.doubleArray[4*i];
            m.y = observation.doubleArray[4*i+1];
            m.sx = observation.doubleArray[4*i+2];
            m.sy = observation.doubleArray[4*i+3];
            m.typeName = monsterNames[m.type]
            monsters.append(m)
            i += 1
        return monsters

    def getMario(self, observation):
        monsters = self.getMonsters(observation)
        for i in xrange(len(monsters)):
            if (monsters[i].type == 0 or monsters[i].type == 10 or monsters[i].type == 11):
                return (monsters[i].x-observation.intArray[0], monsters[i].y)
        return None

    def getMarioFromTiles(self, observation):
        for xi in xrange(22):
            for yi in xrange(16):
                if (self.getTileAt(xi, yi, observation) == 'M'):
                    return (xi, yi)
        return None

    def getTileAt(self, x, y, observation):
        if (x < 0):
            return '7'
        y = 16 - y
        x -= observation.intArray[0]
        if (x<0 or x>21 or y<0 or y>15):
            return '\0';
        index = y*22+x;
        return observation.charArray[index];        

    #Handy little function for debugging by printing out the char array
    def printFullState(self, observation):
        print "-----------------"
        for yi in xrange(16):
            out_string = ""
            for xi in xrange(22):
                out_string += observation.charArray[yi*22+xi]
            print out_string

    #Handy little functio for debugging by printing out Mario's (x,y) position
    def printMarioState(self, observation):
        mar = self.getMario(observation)
        print "Mario X: " + str(mar[0]) + " Mario Y: " + str(mar[1])

    #Handy little function for debugging by printing out the encoded state that is being send to the NN
    def printEncodedState(self, s):
        print "-----------------"
        for yi in xrange(self.state_dim_y):
            out_string = ""
            for xi in xrange(self.state_dim_x):
                out_string += (str(s[yi*self.state_dim_x+xi]) + " ")
            print out_string

    def getAction(self, observation):
        act = self.qnnAction(observation)
        return act

    def update(self, observation, action, reward):
        self.qnnUpdate(observation, action, reward)
        self.Q.ExperienceReplay()

    def stateEncoder(self, observation):
        if (not self.substrate_mode):
            return self.stateEncoderSingle(observation)
        else:
            return self.stateEncoderMultiple(observation)
        pass

    def stateEncoderSingle(self, observation):
        s = []
        #Determine Mario's current position. Everything is relative to Mario
        mar = self.getMario(observation)
        mx = int(mar[0])
        my = 15 - int(mar[1])
        #Update based on the environment
        for yi in xrange(self.state_dim_y):
            for xi in xrange(self.state_dim_x):
                x = mx + xi - int(self.state_dim_x/2.0)
                y = my + yi - int(self.state_dim_y/2.0)
                if (x < 0 or x > 21 or y < 0 or y > 15):
                    s.append(-1)
                    continue
                s.append(tileEncoder[observation.charArray[y*22+x]]) # shouldn't this be 20
        #Add monsters
        monsters = self.getMonsters(observation)
        for mi in xrange(len(monsters)):
            if (monsters[mi].type == 0 or monsters[mi].type == 10 or monsters[mi].type == 11): #skip mario
                continue
            monx = int(monsters[mi].x)
            mony = 15 - int(monsters[mi].y)
            x = monx - mx + int(self.state_dim_x/2.0) - observation.intArray[0]
            y = mony - my + int(self.state_dim_y/2.0)
            if (x < 0 or x >= self.state_dim_x or y < 0 or y >= self.state_dim_y): #skip monsters farther away
                continue
            s[y*self.state_dim_x + x] = -2
        return s

    def stateEncoderMultiple(self, observation):
        s1 = []
        s2 = []
        s3 = []
        #Determine Mario's current position. Everything is relative to Mario
        mar = self.getMario(observation)
        mx = int(mar[0])
        my = 15 - int(mar[1])
        #Add background and reward layers with a placeholder monster layer
        for yi in xrange(self.state_dim_y):
            for xi in xrange(self.state_dim_x/3):
                x = mx + xi - int(self.state_dim_x/6.0)
                y = my + yi - int(self.state_dim_y/2.0)
                if (x < 0 or x > 21 or y < 0 or y > 15):
                    s1.append(0)
                    s2.append(0)
                    s3.append(0)
                    continue
                s1.append(backgroundLayer[observation.charArray[y*22+x]])
                s2.append(rewardLayer[observation.charArray[y*22+x]])
                s3.append(0)
        #Add monsters to the monster layer
        monsters = self.getMonsters(observation)
        for mi in xrange(len(monsters)):
            if (monsters[mi].type == 0 or monsters[mi].type == 10 or monsters[mi].type == 11):#skip mario
                continue
            monx = int(monsters[mi].x)
            mony = 15 - int(monsters[mi].y)
            x = monx - mx + int(self.state_dim_x/6.0) - observation.intArray[0]
            y = mony - my + int(self.state_dim_y/2.0)
            if (x < 0 or x >= self.state_dim_x/3 or y < 0 or y >= self.state_dim_y): #skip monsters farther away
                continue
            s3[y*self.state_dim_x/3 + x] = 1
        return s1 + s2 + s3

    def actionEncoder(self, act):
        a = 1*act.intArray[2] + 2*act.intArray[1] + 4*(act.intArray[0]+1)
        return a

    def actionDecoder(self, a):
        act = Action()
        act.intArray.append(int(a/4.0) - 1)
        act.intArray.append(int(a/2.0) % 2)
        act.intArray.append(a % 2)
        return act

    def randomAction(self, forwardBias):
        act = Action()
        #The first control input is -1 for left, 0 for nothing, 1 for right
        if (not forwardBias or random.random() < 0.1):
            act.intArray.append(random.randint(-1,1))
        else:
            act.intArray.append(random.randint(0,1))
        #The second control input is 0 for nothing, 1 for jump
        act.intArray.append(random.randint(0,1))
        #The third control input is 0 for nothing, 1 for speed increase
        act.intArray.append(random.randint(0,1))        
        return act

    def qnnAction(self, observation):
        s = self.stateEncoder(observation)
        #self.printEncodedState(s)
        if (random.random()>self.exp):
            a = np.argmax(self.Q(s))
            act = self.actionDecoder(a)
        else:
            act = self.randomAction(True)
        return act

    def qnnUpdate(self, observation, action, reward):
        if (self.last_state == None or self.last_action == None):
            return
        s = self.stateEncoder(observation)
        ls = self.stateEncoder(self.last_state)
        a = self.actionEncoder(action)
        la = self.actionEncoder(self.last_action)
        self.Q.RememberExperience(ls, la, reward, s, a)

    def saveQFun(self, fileName):
        theFile = open(fileName, "w")
        pickle.dump(self.Q, theFile)
        theFile.close()

    def loadQFun(self, fileName):
        theFile = open(fileName, "r")
        self.Q = pickle.load(theFile)
        theFile.close()

    '''
    Redundant below
    '''
    def train_TNN(self):
        for s1 in self.transition_probs:
            for a in self.transition_probs[s1]:
                for s2 in self.transition_probs[s1][a]:
                    self.TNN.Update(np.asarray(s1),np.asarray(s2),self.transition_probs[s1][a][s2])

    # TODO - take the current transition_matrix and compute transition_probs (&save it)
    def saveTprobs(self, fileName):
        transition_probs = {}
        for s1 in self.transition_matrix.keys():
            transition_probs[s1] = {}
            for a in self.transition_matrix[s1].keys():
                transition_probs[s1][a] = {}
                Z = 0 #Normalisation constant
                for s2 in self.transition_matrix[s1][a].keys():
                    Z += self.transition_matrix[s1][a][s2]
                for s2 in self.transition_matrix[s1][a].keys():
                    transition_probs[s1][a][s2] = float(self.transition_matrix[s1][a][s2])/float(Z)
        theFile = open(fileName,'w')
        pickle.dump(transition_probs,theFile)

    def loadTprobs(self, fileName):
        theFile = open(fileName, "r")
        self.transition_probs = pickle.load(theFile)
        theFile.close()

if __name__=="__main__":        
    AgentLoader.loadAgent(MarioAgent())
