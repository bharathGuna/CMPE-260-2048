from abc import ABC, abstractmethod
import np as np
class Agent(ABC):
    '''Abstract class defining required functions for an agent'''

    def __init__(self, env, name):
        '''Initialize the agent
        input:
            mask: Mask used to understand the game
            name: Name of agent. Used in tag.'''
        self.env = env
        self.name = name

    @abstractmethod
    def learn(self, prevState, action, state, reward):
        '''Learning Algorithm
        input:
            prevState: State before action is taken
            action: Action taken
            state: State after action is taken
            reward: Reward recieved from action'''
        pass

    @abstractmethod
    def chooseAction(self, state, actions):
        '''Choose next action to take
        input:
            state: Current state of game
            actions: Possible actions to take
        output: Next action to take'''
        pass

    def play(self, verbose=False):
        """Agent plays a single game
           Based on the code from georgwiese:https://github.com/georgwiese/2048-rl
        input:
            verbose: If verbose is true also return game states and scores
        output:
            final score and log if verbose is set to true"""
        env = self.env
        state = env.reset()
        # record previous state to update learning algorithm
        # whether or not game has reached a gameover state
        game_over = env.is_done()
        # If verbose record a log of game states and scores
        if verbose:
            log = []
            log.append([env.score(), state.copy()])
        while not game_over:
            # Choose next action
            next_action = self.chooseAction(state.copy(),
                                            env.available_actions())
            # Perform action and recieve a reward
            reward = game.do_action(next_action)
            # Update learning algorithm
            self.learn(prevState, next_action, game.state().copy(), reward)
            # Update prevState
            prevState = game.state().copy()
            # Add random tile to state
            game.add_random_tile()
            # If verbose add new state and score to log
            if verbose:
                log.append([game.score(), game.state().copy()])
            # Check if game is over
            game_over = game.game_over()
        # If verbose return final score and log
        if verbose:
            return game.score(), log
        # Else return just final score of game
        else:
            return game.score()

    def train(self, numIterations=1000, logFile=None, _mode='w'):
        """Train agent over many games 
        input:
            numIterations: Number of games to play
            logFile: logFile to record final game scores. If false, doesn't
                     record to a file
            _mode: Mode to write to the logFile
        output:
            final score of games"""
        # Initialize score array
        scores = np.zeros(numIterations, dtype=np.int32)
        # For every game
        for i in range(numIterations):
            print(i)
            # Play game and record score
            scores[i] = self.play(verbose=False)
        # If logfile is not none write to the logFile
        if logFile is not None:
            with open(logFile, mode=_mode) as log_File:
                writer = csv.writer(log_File, delimiter='\n',
                                    lineterminator='\n', quoting=csv.QUOTE_NONE)
                writer.writerow(scores)
        return scores

    def makeGif(self, gif_file, num_trials=10, board_size=4, graphic_size=750,
                top_margin=40, seperator_width=12, end_pause=50):
        '''Construct gif of agent playing a game.
        input:
            gif_file: File to save gif
            num_trials: Number of games to look at and choose the best to make
                        the gif
            board_size: Number of tiles in one side of board
            graphic_size: Size of graphic
            top_margin: Size of top margin
            seperator_width: Seperation between tiles in graphic
            end_pause: How many frame to pause at end of gif'''
        # Play num_trials games and choose best for gif
        bestFinalScore = 0
        for i in range(num_trials):
            finalScore, log = self.play(verbose=True) 
            if finalScore > bestFinalScore:
                bestFinalScore = finalScore
                bestLog = log
        # Write to gif_file
        with imageio.get_writer(gif_file, mode='I') as writer:
            # For every game state
            for i in range(np.shape(bestLog)[0]):
                # Create image
                img=makeImage(bestLog[i][0], bestLog[i][1], board_size,
                              graphic_size,top_margin, seperator_width)
                # Append to gif
                writer.append_data(img)
                # Pause on last frame
                if i == np.shape(bestLog)[0]-1:
                    for i in range(end_pause):
                        writer.append_data(img)

    def makeGraph(self, scores=[], logFile=None, graphFile=None, label=None, rollingWindow=30):
        '''Construct graph showing performance over training.
        input:
            scores: Scores to plot
            logFile: File to read scores in from. Will be append to provided
                     scores.
            graphFile: File to write graph to. Does not save graph if is None.
            label: Label for graph
            rollingWindow: Window for rolling average to smooth graph'''
        # Append scores in logFile to scores
        if logFile is not None:
            with open(logFile, mode='r') as log_File:
                reader = csv.reader(log_File, delimiter='\n')
                for row in reader:
                    scores.append(int(row[0]))
        # Calculate rolling averages
        rollingAverages = np.convolve(scores, np.ones((rollingWindow,))/rollingWindow, mode='valid')
        # Calculate values for x axis
        x = np.arange(len(rollingAverages))+rollingWindow/2
        # If label is none set label to tag
        if label is None:
            label=self.getTag()
        # Plot rollingAverages versus x
        plt.plot(x, rollingAverages, label=label)
        # Label axes
        plt.xlabel('Trial')
        plt.ylabel('Score')
        # Save graph
        if graphFile is not None:
            plt.savefig(graphFile)
            plt.clf()
    
    def getTag(self):
        '''Return tag of agent'''
        return self.name + '_' + self.mask.getTag()

    def save(self, fileName):
        '''Save agent with pickle
        input:
            fileName: Save file '''
        pickleFile = open(fileName, 'wb')
        pickle.dump(self.tuples, pickleFile)
        pickleFile.close()

    def load(self, fileName):
        '''Load agent using pickle
        input:
            fileName: Save file'''
        pickleFile = open(fileName, 'rb')
        self.tuples = pickle.load(pickleFile)
        pickleFile.close()

class QAgent(Agent):
    '''Class to perform q learning'''

    def __init__(self, mask, a=0.025, g=0.9999, e=0.0001, name='q'):
        '''Initialize the agent
        input:
            mask: Mask used to understand the game
            a: Learning rate
            g: Discount factor
            e: Exploration rate
            name: Name of agent. Used in tag.'''
        super().__init__(mask, name)
        self.alpha = a
        self.gamma = g
        self.epsilon = e
        # Initialize q table
        self.tuples = numpy.zeros((self.mask.getMaxTupleNum(), 4), dtype=float)

    def learn(self, prevState, action, state, reward): 
        '''Q Learning Algorithm
        input:
            prevState: State before action is taken
            action: Action taken
            state: State after action is taken
            reward: Reward recieved from action'''
        # Get tupleNums of previous state
        tupleNums = self.mask.getTupleNums(prevState)
        # Choose next action off policy 
        next_action = randArgMax(numpy.sum([self.tuples[num] for num in tupleNums], axis=0))
        # Calculate qError
        qError = self.alpha*(reward+self.gamma*self.lookUp(state,next_action)-self.lookUp(prevState,action))
        # Update table entry for each tupleNum

        for num in tupleNums:
            self.tuples[num, action] += qError
            if self.tuples[num, action] < 0:
                self.tuples[num, action] = 0
        
    def chooseAction(self, state, actions):
        '''Choose next action to take with q algorithm
        input:
            state: Current state of game
            actions: Possible actions to take
        output: Next action to take'''
        # Epsilon percent of the time take a random action
        if (random.random() < self.epsilon):
            return actions[random.randint(0, numpy.size(actions) - 1)]
        # Else Choose action that has highest value in lookup table
        values = self.lookUp(state)
        for action in [0, 1, 2, 3]:
            if not numpy.isin(action, actions):
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
            return numpy.sum([self.tuples[num] for num in tupleNums], axis=0)
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
