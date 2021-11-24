import gym
import gym_2048
from agents.agent import QAgent
from agents.mask import Mask_rxcx4

env = gym.make('2048-v0')
env.seed(42)
print(env.observation_space)
env.reset()
env.render()
env.render(mode="rgb_array")

mask = Mask_rxcx4()
# Initialize Agents
qagent = QAgent(mask,env)
print(qagent.train(100))
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
