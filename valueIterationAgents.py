# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*
        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.
          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        #For every iteration...
        for x in range(self.iterations):
            #Copy needed to not completely overwrite the Counter
            tempValues = self.values.copy()

            #For every single state calculate the best possible action via the highest Q-Value
            for state in self.mdp.getStates():
                #If it's a terminal state, no action can be computed, so skip
                if self.mdp.isTerminal(state):
                    continue

                #List of possible actions at the state
                tempActions = self.mdp.getPossibleActions(state)
                #set the current state's value to the best action via highest Q-Value as indicated in lecture slides
                maxQValue = float('-Inf')
                for action in tempActions:
                    qValue = self.getQValue(state, action)
                    if qValue >= maxQValue:
                        maxQValue = qValue
                #Set the Counter dictionary's value at <state> to the found maxQValue
                tempValues[state] = maxQValue

            #Finally after iterating through every state set the current Counter to the new counter where all states are set to maximum Q-Value actions
            self.values = tempValues


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        qValue = 0
        for nextState, transition in self.mdp.getTransitionStatesAndProbs(state, action):
            #qValue += transition * ((discount * value) + reward)
            #qValue += T(s,a,s') * ((reward * V(s')) + R(s,a,s'))
            qValue += transition * ((self.values[nextState] * self.discount) + self.mdp.getReward(state, action, nextState))
        return qValue

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.
          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """

        #If current state is a terminal state, best to return is do nothing
        if self.mdp.isTerminal(state):
            return None

        #Otherwise find the best possible action via Q-Value
        values = util.Counter()
        actions = self.mdp.getPossibleActions(state)
        for action in actions:
            values[action] = self.getQValue(state, action)

        #Return the action with the largest Q-Value
        return values.argMax()


    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*
        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.
          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        #Go through each iteration k, iterations is an arguement so we need it to be iterable hence the range
        for i in range(self.iterations):
            # Method to update "only one state in each iteration", this could aslo be implemented
            # using a for loop through states but this method is much more straight forward.
            state = states[i % len(states)]
            # Find action for chosen state
            action = self.computeActionFromValues(state)
            #Check if action exists
            if action:
                # Update values by computing qValue
                self.values[state] = self.computeQValueFromValues(state, action)


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        #Initialize an empty priority queue
        priority = util.PriorityQueue()
        #Priority is a set, not a list, to avoid duplicates
        predecessors = {}

        #Compute predecessors of all states
        #Iterate through every state
        for state in self.mdp.getStates():
          #If the state is a terminal state ignore
          if self.mdp.isTerminal(state):
            continue
          #Otherwise iterate through every possible subsequent state
          for action in self.mdp.getPossibleActions(state):
            #Make a dictionary of all possible nextStates
            for nextStateValues in self.mdp.getTransitionStatesAndProbs(state, action):
                #nextStateValues[0] = the next State
                if nextStateValues[0] in predecessors:
                    predecessors[nextStateValues[0]].add(state)
                else:
                    predecessors[nextStateValues[0]] = {state}

        #For each non-terminal state...
        for state in self.mdp.getStates():
          if self.mdp.isTerminal(state):
              continue
          qValues = []
          #Calculate all of the qValues
          for action in self.mdp.getPossibleActions(state):
            qValues.append(self.computeQValueFromValues(state, action))
          #find the absolute value of the difference...
          diff = abs(max(qValues) - self.values[state])
          #Push s into the priority queue with priority -diff...
          priority.push(state, -diff)

        #For iteration in 0, 1, 2, ..., self.iterations - 1, do
        for i in range(self.iterations):
          #If priority queue is empty, then terminate
          if priority.isEmpty():
            break
          #pop a state s off the priority queue
          s = priority.pop()
            
          #If it is not a terminal state, update s's value in self.values
          if not self.mdp.isTerminal(s):
            qValues = []
            #Updating s's value in self.values with maximum qValue
            for action in self.mdp.getPossibleActions(s):
              q_value = self.computeQValueFromValues(s, action)
              qValues.append(q_value)
            self.values[s] = max(qValues)
          #For each predecessor p of s
          for p in predecessors[s]:
            #Calculate diff again
            qValues = []
            for action in self.mdp.getPossibleActions(p):
                qValues.append(self.computeQValueFromValues(p, action))
            #Calculated diff
            diff = abs(max(qValues) - self.values[p])
            #If diff > theta push p into priority queue...
            if diff > self.theta:
                priority.update(p, -diff)

