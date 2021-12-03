# SARSA is an ON-Policy technique in Reinforcement learning
from agents.agent import Agent
from agents.utils import randArgMax
import random
import numpy as np

class SARSAAgent(Agent):
    '''Class to perform SARSA learning'''

    def __init__(self, mask, env, a=0.01, g=0.75, e=0.001, name='SARSA'):
        '''Initialize the agent
        input:
            mask: Mask used to understand the game
            a: Learning rate
            g: Discount factor
            e: Exploration rate
            name: Name of agent. Used in tag.'''

        super().__init__(mask, env, name)
        self.alpha = a
        self.gamma = g
        self.epsilon = e
        # Initialize table
        self.tuples = np.zeros((self.mask.getMaxTupleNum(), 4), dtype=float)

    def learn(self, prevState, action, state, reward): 
        '''SARSA Learning Algorithm
        input:
            prevState: State before action is taken
            action: Action taken
            state: State after action is taken
            reward: Reward recieved from action'''
        # Get tupleNums of previous state
        tupleNums = self.mask.getTupleNums(prevState)
        # Choose next action on policy
        next_action = self.chooseAction(state, self.env.available_actions())
        # Calculate sarsaError
        sarsaError = self.alpha*(reward+self.gamma*self.lookUp(state,next_action)-self.lookUp(prevState,action))
        # Update table entry for each tupleNum
        for num in tupleNums:
            self.tuples[num, action] += sarsaError
            if self.tuples[num, action] < 0:
                self.tuples[num, action] = 0
        
    def chooseAction(self, state, actions):
        '''Choose next action to take with sarsa algorithm
        input:
            state: Current state of game
            actions: Possible actions to take
        output: Next action to take'''
        # Epsilon percent of the time take a random action
        if (random.random() < self.epsilon):
            return actions[random.randint(0, np.size(actions) - 1)]
        # Else Choose action that has highest value in lookup table
        values = self.lookUp(state)
        for action in [0, 1, 2, 3]:
            if not np.isin(action, actions):
                values[action] = -1
        return randArgMax(values)
        
    def lookUp(self, state, action=None):
        ''' Look up value of state(action pair) in look up table
        input:
            state: State to look up
            action: Next action to take. If action is none look up the value
                    for each action.
        output: Value of state(action pair) in look up table'''
        # Get tuple nums of state
        tupleNums = self.mask.getTupleNums(state)
        # If action is none get value for each action
        if action is None:
            return np.sum([self.tuples[num] for num in tupleNums], axis=0)
        else:
            # Add up the value for each tupleNum
            return sum([self.tuples[num, action] for num in tupleNums])

    def getTag(self):
        '''Return tag of agent'''
        tag = super().getTag()
        tag += '_a'+str(self.alpha).split('.')[1]
        tag += 'e'+str(self.epsilon).split('.')[1]
        tag += 'g'+str(self.gamma).split('.')[1]
        return tag
