# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        # Use counter here for future use of helper dict methods
        # Great reference for this agent found here
        # https://towardsdatascience.com/a-beginners-guide-to-q-learning-c3e2a30a653c
        # For reference to the above article the below counter is our "Q-Table" implementation
        self.Q_VALUE_TABLE = util.Counter()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        # Simply get the q-value of this state and action
        # from our "Q-table"
        return self.Q_VALUE_TABLE[(state,action)]


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        legalActions = self.getLegalActions(state)
        # See Above
        if not legalActions:
            return 0.0
        # Using chad's implementation of least possible value.
        maxAction = float('-Inf')
        for action in legalActions:
            # Use above implemented q-value getter and compare
            maxAction = max(maxAction, self.getQValue(state,action))
        return maxAction


    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        legalActions = self.getLegalActions(state)
        bestAction = None
        if not legalActions:
            return bestAction

        # First: Initialize maxQValue as first legal Action then
        # see if there are other  actions that have a greater or equal
        # q-value. If equal then  we break the tie using random.choice() mentioned
        # in project 3 notes. Or else set the return action to the best q-value action
        returnAction = legalActions[0]
        maxQValue = self.getQValue(state, returnAction)

        #Find true max qValue
        for action in legalActions:
            actionQValue = self.getQValue(state, action)
            if actionQValue > maxQValue:
                maxQValue = actionQValue
                returnAction = action
            elif actionQValue == maxQValue:
                returnAction = random.choice([returnAction, action])

        return returnAction





    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None

        probability = util.flipCoin(self.epsilon)
        if probability:
            action = random.choice(legalActions)
        else:
            action = self.computeActionFromQValues(state)


        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        currentQValue = self.Q_VALUE_TABLE[(state,action)]
        # The new value is calculated by discounting the next states value
        # and then adding the new reward

        derivedQValue = self.getValue(nextState) * self.discount + reward

        # We then update the q-value for this state, action pair
        # using the learning rate formula below found in RL2 lecture @ 10:05
        # ( 1 - alpha) QValue + alpha * Q**
        self.Q_VALUE_TABLE[(state, action)] = (1.0 - self.alpha) * currentQValue + self.alpha * derivedQValue


    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        # Initialize
        qValue = 0
        featureVector = self.featExtractor.getFeatures(state,action).items()
        weights = self.getWeights()

        # Dot product the above formula using provided properties and methods
        for feature in featureVector:
            qValue += weights[feature] * feature[1]

        return qValue

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        # Formulas below come from lecture notes & project 3 hints:
        "*** YOUR CODE HERE ***"
        #   difference = ( reward + discount * max Q(s',a') ) - Q(s,a)
        difference = reward + self.discount * self.getValue(nextState) - self.getQValue(state, action)

        featureVector = self.featExtractor.getFeatures(state,action).items()
        # print featureVector
        for feature in featureVector:
        #   weight <- weight + alpha * difference * fi( s, a )
            self.weights[feature] = self.weights[feature] + self.alpha * difference * feature[1]

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            for weight in self.weights:
                print "Weight: %s", str(weight)
