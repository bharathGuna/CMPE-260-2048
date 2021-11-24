import numpy as np
import gym
import gym.spaces as spaces
from gym.utils import seeding
from PIL import Image, ImageDraw, ImageFont


class Base2048Env(gym.Env):
  metadata = {
      'render.modes': ['human', 'rgb_array'],
  }

  ##
  # NOTE: Don't modify these numbers as
  # they define the number of
  # anti-clockwise rotations before
  # applying the left action on a grid
  #
  LEFT = 0
  UP = 1
  RIGHT = 2
  DOWN = 3

  ACTION_STRING = {
      LEFT: 'left',
      UP: 'up',
      RIGHT: 'right',
      DOWN: 'down',
  }

  def __init__(self, score=0, width=4, height=4):
    self.width = width
    self.height = height
    self._score = score

    self.observation_space = spaces.Box(low=1,
                                        high=17,
                                        shape=(self.width, self.height),
                                        dtype=np.int64)

    self.action_space = spaces.Discrete(4)

    # Internal Variables
    self._board = None
    self.np_random = None
    self.grid_size = 70

    self.seed()
    self.reset()

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def step(self, action: int):
    """Rotate board aligned with left action"""

    # Align board action with left action
    prevBoard = self._board.copy()
    rotated_obs = np.rot90(self._board, k=action)
    reward, updated_obs = self._slide_left_and_merge(rotated_obs)
    self._board = np.rot90(updated_obs, k=4 - action)

    # Place one random tile on empty location
    self._place_random_tiles(count=1)

    done = self.is_done()
    self._score += reward

    return self._board, reward, done, {}

  def is_done(self):
    copy_board = self._board.copy()

    if not copy_board.all():
      return False

    for action in [0, 1, 2, 3]:
      rotated_obs = np.rot90(copy_board, k=action)
      _, updated_obs = self._slide_left_and_merge(rotated_obs)
      if not updated_obs.all():
        return False

    return True


  def available_actions(self):
      """Computes the set of actions that are available."""
      return [action for action in range(4) if self.is_action_available(action)]

  def is_action_available(self, action):
      """Determines whether action is available.
      That is, executing it would change the state."""
      temp_board = np.rot90(self._board, action)
      return self._is_action_available_left(temp_board)

  def _is_action_available_left(self, state):
      '''Determines whether action 'Left' is available.
          True if any field is 0 (empty) on the left of 
          a tile or two tiles can be merged.'''
      for row in range(self.width):
          has_empty = False
          for col in range(self.height):
              has_empty |= state[row, col] == 0
              if state[row, col] != 0 and has_empty:
                  return True
              if (state[row, col] != 0 and col > 0 and
                  state[row, col] == state[row, col - 1]):
                  return True
      return False

  def reset(self):
    """Place 2 tiles on empty board."""

    self._board = np.zeros((self.width, self.height), dtype=np.int64)
    self._place_random_tiles(count=2)
    self._score = 0
    return self._board

  def score(self):
    return self._score

  def highest(self):
      return np.max(self._board)

  def get(self, x, y):
      return self._board[x][y]
  
  def getBoard(self):
    return self._board.copy()

  def _place_random_tiles(self, count=1):
      """Adds a random tile to the grid. Assumes that it has empty fields."""
      x_pos, y_pos = np.where(self._board == 0)
      if len(x_pos) != 0:
      
        empty_index = np.random.choice(len(x_pos))
        # Adding 2 or 4 tile
        value = np.random.choice([1, 2], p=[0.9, 0.1])
        self._board[x_pos[empty_index], y_pos[empty_index]] = value
 
  def _slide_left_and_merge(self, board):
    """Slide tiles on a grid to the left and merge."""

    result = []

    score = 0
    for row in board:
      row = np.extract(row > 0, row)
      score_, result_row = self._try_merge(row)
      score += score_
      row = np.pad(np.array(result_row), (0, self.width - len(result_row)),
                   'constant', constant_values=(0,))
      result.append(row)

    return score, np.array(result, dtype=np.int64)

  @staticmethod
  def _try_merge(row):
    score = 0
    result_row = []

    i = 1
    while i < len(row):
      if row[i] == row[i - 1]:
        score += 2**row[i] + 2**row[i - 1]
        result_row.append(row[i] + 1)
        i += 2
      else:
        result_row.append(row[i - 1])
        i += 1

    if i == len(row):
      result_row.append(row[i - 1])

    return score, result_row

  def pow2(self,x):
    return pow(2,x)

  def render(self, mode='human'):
    if mode == 'human':
      for row in self._board.tolist():
        print(' \t'.join(map(lambda x: str(pow(2,x)), row)))

    if mode == 'rgb_array':
            black = (0, 0, 0)
            grey = (128, 128, 128)
            white = (255, 255, 255)
            tile_colour_map = {
                0: white,
                1: (255, 0, 0),
                2: (224, 32, 0),
                3: (192, 64, 0),
                4: (160, 96, 0),
                5: (128, 128, 0),
                6: (96, 160, 0),
                7: (64, 192, 0),
                8: (32, 224, 0),
                9: (0, 255, 0),
                10: (0, 224, 32),
                11: (0, 192, 64),
                12: (0, 160, 96),
            }
            grid_size = self.grid_size

            # Render with Pillow
            pil_board = Image.new("RGB", (grid_size * 4, grid_size * 4))
            draw = ImageDraw.Draw(pil_board)
            draw.rectangle([0, 0, 4 * grid_size, 4 * grid_size], white)
            fnt = ImageFont.load_default()

            for y in range(4):
              for x in range(4):
                o = self.get(y, x)
                val = 2 ** o
                #  if o:
                draw.rectangle([x * grid_size, y * grid_size, (x + 1) * grid_size, (y + 1) * grid_size], tile_colour_map[o], outline=black)
                (text_x_size, text_y_size) = draw.textsize(str(val), font=fnt)
                if val != 0:
                  draw.text((x * grid_size + (grid_size - text_x_size) // 2, y * grid_size + (grid_size - text_y_size) // 2), str(val), font=fnt, fill=white)
                  assert text_x_size < grid_size
                  assert text_y_size < grid_size
            pil_board.save('test.jpg')
            return np.asarray(pil_board).swapaxes(0, 1)
