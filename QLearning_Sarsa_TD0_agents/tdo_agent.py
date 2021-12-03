from QLearning_Sarsa_TD0_agents.agent import Agent
from QLearning_Sarsa_TD0_agents.utils import randArgMax
import random
import numpy as np
import gym
import gym_2048

class TD0Agent(Agent):
    '''Class to perform TD0 learning'''

    def __init__(self, mask, env, a=0.02, g=0.9999, e=0.0001, name='td0'):
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
        #Initialize table
        self.tuples = np.zeros(self.mask.getMaxTupleNum(), dtype=float)

    def learn(self, prevState, action, state, reward): 
        '''TD0 Learning Algorithm
        input:
            prevState: State before action is taken
            action: Action taken
            state: State after action is taken
            reward: Reward recieved from action'''
        # Get tupleNums of previous state
        tupleNums = self.mask.getTupleNums(prevState)
        # Calculate tdError
        tdError = self.alpha*(reward+self.gamma*self.lookUp(state)-self.lookUp(prevState))
        # Update table entry for each tupleNum
        for num in tupleNums:
            self.tuples[num] += tdError
            if self.tuples[num] < 0:
                self.tuples[num] = 0

    def chooseAction(self, state, actions):
        '''Choose next action to take with td0 algorithm
        input:
            state: Current state of game
            actions: Possible actions to take
        output: Next action to take'''
        # Epsilon percent of the time take a random action
        if (random.random() < self.epsilon):
            return actions[random.randint(0, np.size(actions) - 1)]
        # Else take action that puts you in state with highest value in look up
        # table
        values = np.full(4, -1, dtype=float)
        for action in actions:
            tempGame = gym.make('2048-v0', state=np.copy(state))
            tempGame.reset()
            board, reward,done,info = tempGame.step(action)
            values[action] = reward + self.lookUp(board)
        return randArgMax(values)

    def lookUp(self, state):
        ''' Look up value of state(action pair) in look up table
        input:
            state: State to look up
            action: Next action to take. If action is none look up the value
                    for each action.
        output: Value of state(action pair) in look up table'''
        # Get tuple nums of state
        tupleNums = self.mask.getTupleNums(state)
        # Add up the value for each tupleNum
        return np.sum([self.tuples[num] for num in tupleNums])

    def getTag(self):
        '''Return tag of agent'''
        tag = super().getTag()
        tag += '_a'+str(self.alpha).split('.')[1]
        tag += 'e'+str(self.epsilon).split('.')[1]
        tag += 'g'+str(self.gamma).split('.')[1]
        return tag   
