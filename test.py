import gym
import gym_2048
from agents.q_agent import QAgent
from agents.sarsa_agent import SARSAAgent
from agents.tdo_agent import TD0Agent
from agents.mask import Mask_rxcx4
env = gym.make('2048-v0')
env.seed(42)
print(env.observation_space)
env.reset()
env.render()
env.render(mode="rgb_array")

mask = Mask_rxcx4()
# Initialize Agents
print("Q Learning")
qagent = QAgent(mask,env)
qagent.train(10)
qagent.makeGif('qlearning.gif', graphic_size=200, top_margin=20,
                seperator_width=6, num_trials=50)

print("Sarsa")
sarsa = SARSAAgent(mask,env)
sarsa.train(10)
sarsa.makeGif('sarsa.gif', graphic_size=200, top_margin=20,
                seperator_width=6, num_trials=50)

print("Tdo")
tdo = TD0Agent(mask,env)
tdo.train(10)
tdo.makeGif('tdo.gif', graphic_size=200, top_margin=20,
                seperator_width=6, num_trials=50)
# done = False
# moves = 0
# score = 0
# while not done:
#     action = env.np_random.choice(range(4), 1).item()
#     next_state, reward, done, info = env.step(action)
#     print(next_state)
#     moves += 1
#     score +=reward
#     print('Next Action: "{}"\n\nReward: {}'.format(
#       gym_2048.Base2048Env.ACTION_STRING[action], reward))
#     env.render()

# print(score)
# print(env.score())
# env.render(mode="rgb_array")

# print('\nTotal Moves: {}'.format(moves))
