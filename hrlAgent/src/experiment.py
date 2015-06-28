#!/usr/bin/python

# Mario Environment Loader

import random
import rlglue.RLGlue as RLGlue
import matplotlib.pyplot as plt
from consoleTrainerHelper import *
import pickle

def trainAgent():
    
    episodesToRun = 100000 # param
    exp = 1.0 # epsilon?

    totalSteps = 0

    '''
    #Random player
    raw_results = []
    RLGlue.RL_agent_message("freeze_learning")
    RLGlue.RL_agent_message("freeze_transition_learning")
    for i in range(episodesToRun):
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
    results1 = []
    for i in range(100,episodesToRun):
        results1.append(sum(raw_results[i-100:i])/100.0)
    '''

    #Q-NN agent
    # First, we unfreeze learning to train QNN. Once trained, we also have state representations (hidden layer output of trained network).
    # Next, we freeze layer and only learn the transition probabilities.
    
    RLGlue.RL_agent_message("freeze_option_learning")
    RLGlue.RL_agent_message("unfreeze_learning")
    RLGlue.RL_agent_message("freeze_transition_learning")
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
    results2 = []
    for i in range(100,episodesToRun):
        results2.append(sum(raw_results[i-100:i])/100.0)

    #plt.plot(results1, color='red', label='Random')
#    plt.plot(results2, color='blue', label='Neural Q-Network')
#    plt.xlabel('Episode Number')
#    plt.ylabel('Mean Total Reward over 100 Episodes')
#    plt.legend()
#    plt.show()

    RLGlue.RL_agent_message("save_state_reps state_reps"+str(episodesToRun)+".dat")
    RLGlue.RL_agent_message("get_bins_from_state_reps secondcolbins"+str(episodesToRun)+".dat thirdcolbins"+str(episodesToRun)+".dat")
    RLGlue.RL_agent_message("save_policy qfun"+str(episodesToRun)+".dat")

    # Transition Probs learning
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
        raw_results.append(thisReturn)
        totalSteps += thisSteps
    print "Total steps : %d\n" % (totalSteps)

#    RLGlue.RL_agent_message("savetransmatrix transitionProbs.dat")
    RLGlue.RL_agent_message("save_phi phi_mat"+str(episodesToRun)+".dat u_mat"+str(episodesToRun)+".dat peeyush_u_mat"+str(episodesToRun)+".dat")

    # Train TNN
#    RLGlue.RL_agent_message("train_TNN")


#####################################################################################################################

def optionPlay():
    episodesToRun = 4000
    totalSteps = 0    

    RLGlue.RL_agent_message("freeze_learning")
    RLGlue.RL_agent_message("freeze_transition_learning")
    RLGlue.RL_agent_message("unfreeze_option_learning")
    RLGlue.RL_agent_message("load_policy qfun1000.dat")

    # load SMDP value function if present already
    RLGlue.RL_agent_message("load_vf")

    stepsarray = []
    returnarray = []
    for i in range(episodesToRun):
        RLGlue.RL_episode(episodesToRun)
        thisSteps = RLGlue.RL_num_steps()
        stepsarray.append(thisSteps)
        print "Total steps in episode %d is %fd" %(i, thisSteps)
        thisReturn = RLGlue.RL_return()
        returnarray.append(thisReturn)
        print "Total return in episode %d is %f" %(i, thisReturn)
        totalSteps += thisSteps
    
    thisoutput = (stepsarray,returnarray)
    f = open('normalumat'+str(episodesToRun)+'.dat','w')
    pickle.dump(thisoutput,f)

    #RLGlue.RL_agent_message("save_vf")

    print "Total steps : %d\n" % (totalSteps)


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


def main():
    whichTrainingMDP = 0

    '''
    Parameter definition:
    fast - determines if Mario runs very fast or at playable-speed. Set it to true to train your agent, false if you want to actually see what is going on.
    dark - make Mario visible or not. Set it to true to make it invisible when training.
    levelSeed - determines Marios behavior 
    levelType - 0..2: outdoors/subterranean/other
    levelDifficulty - 0..10, how hard it is. 
    instance - 0..9, determines which Mario you run.    
    '''

    loadMario(True, True, 3, 0, 1, whichTrainingMDP)

    RLGlue.RL_init()

    #RLGlue.RL_agent_message("loadtransmatrix")

    #RLGlue.RL_agent_message("load_policy agents/exampleAgent.dat")

    trainAgent()

    #optionPlay()

    #RLGlue.RL_agent_message("save_policy agents/exampleAgent.dat")

    #RLGlue.RL_agent_message("savetransmatrix")

    #testAgent()

    #RLGlue.RL_agent_message("save_policy qfun.dat")

    RLGlue.RL_cleanup()

if __name__ == "__main__":
    main()
