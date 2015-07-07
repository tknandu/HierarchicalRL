#!/usr/bin/python

# Mario Environment Loader

import random
import rlglue.RLGlue as RLGlue
import matplotlib.pyplot as plt
from consoleTrainerHelper import *
import pickle
import numpy as np

#Utility function
def loadCountMats(episodesTrainedOn):

    n_eps = episodesTrainedOn

    f = open('phi_mat' + str(n_eps) + '.dat','r')
    unpickler = pickle.Unpickler(f)
    phi_mat = unpickler.load()

    f = open('u_mat' + str(n_eps) + '.dat','r')
    unpickler = pickle.Unpickler(f)
    u_mat = unpickler.load()

    f = open('peeyush_u_mat' + str(n_eps) + '.dat','r')
    unpickler = pickle.Unpickler(f)
    pu_mat = unpickler.load()

    n_valid_states = len(phi_mat.keys())
    valid_states = phi_mat.keys()
    valid_states.sort()
    n_actions = 12
#    print valid_states
#    print n_valid_states

    f = open('valid_states' + str(n_eps) + '.dat','w')
    pickle.dump(valid_states,f)

#Using the matrices from page 39 of thesis
    d_mat = np.zeros((n_valid_states,n_valid_states),dtype=float)
    for (i_1,state_1) in enumerate(valid_states):
        for (i_2,state_2) in enumerate(valid_states):
            total = 0
            for action in range(n_actions):
                if action in u_mat[state_1]:
                    if state_2 in u_mat[state_1][action]:
                        total += u_mat[state_1][action][state_2]
            d_mat[i_1][i_2] = float(total)

    f = open('d_mat' + str(n_eps) + '.dat','w')
    pickle.dump(d_mat,f)

    p_mat = np.zeros((n_valid_states,n_actions,n_valid_states),dtype=float)
    for (i_1,state_1) in enumerate(valid_states):
        for action in range(n_actions):
            if action in u_mat[state_1]:
                total = 0
                for (i_2, state_2) in enumerate(valid_states):
                    if state_2 in u_mat[state_1][action]:
                        total += u_mat[state_1][action][state_2]
                if total == 0:
                    p_mat[i_1][action][i_1] = 1 #If we have no transitions for (s_1,a), then set p(s_1,a,s_1) to 1 and let the others be 0
                else:
                    for (i_2,state_2) in enumerate(valid_states):
                        if state_2 in u_mat[state_1][action]:
                            p_mat[i_1][action][i_2] = u_mat[state_1][action][state_2]/float(total)
            else:
                p_mat[i_1][action][i_1] = 1

    f = open('p_mat' + str(n_eps) + '.dat','w')
    pickle.dump(p_mat,f)

    t_mat = np.zeros((n_valid_states,n_valid_states),dtype=float)
    for (i_1,state_1) in enumerate(valid_states):
        total = 0
        for (i_2, state_2) in enumerate(valid_states):
            total += d_mat[i_1][i_2]
        if total == 0:
            t_mat[i_1][i_1] = 1 #If we have no transitions out of s_1, then we set p(s_1,s_1) to 1 and let the others be 0
        else:
            for (i_2,state_2) in enumerate(valid_states):
                t_mat[i_1][i_2] = d_mat[i_1][i_2]/float(total)

    f = open('t_mat' + str(n_eps) + '.dat','w')
    pickle.dump(t_mat,f)

def trainQNetwork(episodesToRun):

    # Training Q-Network
    totalSteps = 0
    exp = 1.0 # reset exp

    RLGlue.RL_agent_message("freeze_trajectory")
    RLGlue.RL_agent_message("freeze_option_learning")
    RLGlue.RL_agent_message("freeze_transition_learning")
    RLGlue.RL_agent_message("unfreeze_learning")
    raw_results = []
    for i in range(episodesToRun):
        if (i % 100 == 0):
            if (exp > 0.1):
                exp -= 0.05
            RLGlue.RL_agent_message("set_exploring " + str(exp))
        RLGlue.RL_episode(2000)
        thisSteps = RLGlue.RL_num_steps()
        print "Total steps in episode %d is %d" %(i, thisSteps)
        thisReturn = RLGlue.RL_return()
        if (thisReturn > 50.0):
            thisReturn = 10.0
        print "Total return in episode %d is %f" %(i, thisReturn)
        raw_results.append(thisReturn)
        totalSteps += thisSteps
    print "Total steps : %d\n" % (totalSteps)

    RLGlue.RL_agent_message("save_state_reps state_reps"+str(episodesToRun)+".dat")
    RLGlue.RL_agent_message("save_bins_from_state_reps colbins"+str(episodesToRun)+".dat")
    RLGlue.RL_agent_message("save_policy qfun"+str(episodesToRun)+".dat")

def learnTransitionProb(episodesToRun):

    # Transition Probs learning
    totalSteps = 0
    exp = 1.0 # reset exp

    RLGlue.RL_agent_message("get_bins_from_state_reps colbins"+str(episodesToRun)+".dat")

    RLGlue.RL_agent_message("freeze_trajectory")
    RLGlue.RL_agent_message("freeze_option_learning")
    RLGlue.RL_agent_message("freeze_learning")
    RLGlue.RL_agent_message("unfreeze_transition_learning")
    for i in range(episodesToRun):
        if (i % 100 == 0):
            if (exp > 0.1):
                exp -= 0.05
            RLGlue.RL_agent_message("set_exploring " + str(exp))
        RLGlue.RL_episode(2000)
        thisSteps = RLGlue.RL_num_steps()
        print "Total steps in episode %d is %d" %(i, thisSteps)
        thisReturn = RLGlue.RL_return()
        if (thisReturn > 50.0):
            thisReturn = 10.0
        print "Total return in episode %d is %f" %(i, thisReturn)
        totalSteps += thisSteps
    print "Total steps : %d\n" % (totalSteps)

    RLGlue.RL_agent_message("save_phi phi_mat"+str(episodesToRun)+".dat u_mat"+str(episodesToRun)+".dat peeyush_u_mat"+str(episodesToRun)+".dat")

#####################################################################################################################


def getTrajectories(episodesTrainedOn, episodesToRun):

    RLGlue.RL_agent_message("freeze_learning")
    RLGlue.RL_agent_message("freeze_transition_learning")
    RLGlue.RL_agent_message("freeze_option_learning")
    RLGlue.RL_agent_message("unfreeze_trajectory")
    RLGlue.RL_agent_message("load_policy qfun"+str(episodesTrainedOn)+".dat")
    RLGlue.RL_agent_message("get_bins_from_state_reps colbins"+str(episodesTrainedOn)+".dat")

    loadCountMats(episodesTrainedOn)

    exp = 1.0 # reset exp
    for i in range(episodesToRun):
        if (i % 100 == 0):
            if (exp > 0.1):
                exp -= 0.05
            RLGlue.RL_agent_message("set_exploring " + str(exp))
        RLGlue.RL_episode(2000)
        thisSteps = RLGlue.RL_num_steps()
        print "Total steps in episode %d is %fd" %(i, thisSteps)
        thisReturn = RLGlue.RL_return()
        print "Total return in episode %d is %f" %(i, thisReturn)

#####################################################################################################################

def optionPlay(episodesTrainedOn, episodesToRun):

    value_function_already_present = False
    exp = 1.0
    totalSteps = 0

    RLGlue.RL_agent_message("freeze_trajectory")
    RLGlue.RL_agent_message("freeze_learning")
    RLGlue.RL_agent_message("freeze_transition_learning")
    RLGlue.RL_agent_message("unfreeze_option_learning")
    RLGlue.RL_agent_message("load_policy qfun"+str(episodesTrainedOn)+".dat")
    RLGlue.RL_agent_message("get_bins_from_state_reps colbins"+str(episodesTrainedOn)+".dat")

    # load SMDP value function if present already
    if value_function_already_present:
        RLGlue.RL_agent_message("load_vf valuefunction"+str(episodesTrainedOn)+".dat")

    stepsarray = []
    returnarray = []
    for i in range(episodesToRun):
        if (i % 100 == 0):
            if (exp > 0.1):
                exp -= 0.05
            RLGlue.RL_agent_message("set_exploring " + str(exp))
        RLGlue.RL_episode(2000)
        thisSteps = RLGlue.RL_num_steps()
        stepsarray.append(thisSteps)
        print "Total steps in episode %d is %fd" %(i, thisSteps)
        thisReturn = RLGlue.RL_return()
        returnarray.append(thisReturn)
        print "Total return in episode %d is %f" %(i, thisReturn)
        totalSteps += thisSteps
    print "Total steps : %d\n" % (totalSteps)

    thisoutput = (stepsarray,returnarray)
    f = open('normalumat'+str(episodesToRun)+'.dat','w')
    pickle.dump(thisoutput,f)

    # save updated SMDP value function
    RLGlue.RL_agent_message("save_vf valuefunction"+str(episodesTrainedOn)+".dat")

#####################################################################################################################

def testAgent():
    episodesToRun = 500
    totalSteps = 0

    RLGlue.RL_agent_message("freeze learning")
    for i in range(episodesToRun):
        RLGlue.RL_episode(2000)
        thisSteps = RLGlue.RL_num_steps()
        print "Total steps in episode %d is %fd" %(i, thisSteps)
        thisReturn = RLGlue.RL_return()
        print "Total return in episode %d is %f" %(i, thisReturn)
        totalSteps += thisSteps
    print "Total steps : %d\n" % (totalSteps)
    RLGlue.RL_agent_message("unfreeze learning");

#####################################################################################################################

def main():

    episodesToTrain = 100000
    episodesToRun = 100
    n_bins = 1000
    n_hidden_layer_nodes = 1

    '''
    Parameter definition:
    fast - determines if Mario runs very fast or at playable-speed. Set it to true to train your agent, false if you want to actually see what is going on.
    dark - make Mario visible or not. Set it to true to make it invisible when training.
    levelSeed - determines Marios behavior 
    levelType - 0..2: outdoors/subterranean/other
    levelDifficulty - 0..10, how hard it is. 
    instance - 0..9, determines which Mario you run.    
    '''
    whichTrainingMDP = 0
    loadMario(True, True, 3, 0, 1, whichTrainingMDP)
    RLGlue.RL_init()
    RLGlue.RL_agent_message("set_n_bins "+str(n_bins))
    RLGlue.RL_agent_message("set_n_hidden_layer_nodes "+str(n_hidden_layer_nodes))

    '''
    First, we unfreeze learning to train QNN. Once trained, we also have state representations (hidden layer output of trained network).
    Next, we freeze layer and only learn the transition probabilities.
    '''
#    trainQNetwork(episodesToTrain)
#    learnTransitionProb(episodesToTrain)

#    getTrajectories(episodesToTrain,episodesToRun)

    #optionPlay(episodesToTrain, episodesToRun)

    RLGlue.RL_cleanup()

if __name__ == "__main__":
    main()
