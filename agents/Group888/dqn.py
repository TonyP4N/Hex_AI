import numpy as np
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from random import choice

from BestAgent import NaiveAgent

from enum import Enum


class Colour(Enum):
    """This enum describes the sides in a game of Hex."""

    # RED is vertical, BLUE is horizontal
    RED = (1, 0)
    BLUE = (0, 1)

    def get_text(colour):
        """Returns the name of the colour as a string."""

        if colour == Colour.RED:
            return "Red"
        elif colour == Colour.BLUE:
            return "Blue"
        else:
            return "None"

    def get_char(colour):
        """Returns the name of the colour as an uppercase character."""

        if colour == Colour.RED:
            return "R"
        elif colour == Colour.BLUE:
            return "B"
        else:
            return "0"

    def from_char(c):
        """Returns a colour from its char representations."""

        if c == "R":
            return Colour.RED
        elif c == "B":
            return Colour.BLUE
        else:
            return None

    def opposite(colour):
        """Returns the opposite colour."""

        if colour == Colour.RED:
            return Colour.BLUE
        elif colour == Colour.BLUE:
            return Colour.RED
        else:
            return None

class Tile:
    """The class representation of a tile on a board of Hex."""

    # number of neighbours a tile has
    NEIGHBOUR_COUNT = 6

    # relative positions of neighbours, clockwise from top left
    I_DISPLACEMENTS = [-1, -1, 0, 1, 1, 0]
    J_DISPLACEMENTS = [0, 1, 1, 0, -1, -1]

    def __init__(self, x, y, colour=None):
        super().__init__()

        self.x = x
        self.y = y
        self.colour = colour

        self.visited = False

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def set_colour(self, colour):
        self.colour = colour

    def get_colour(self):
        return self.colour

    def visit(self):
        self.visited = True

    def is_visited(self):
        return self.visited

    def clear_visit(self):
        self.visited = False

class Board:
    """Class that describes the Hex board."""

    def __init__(self, board_size=11):
        super().__init__()

        self._board_size = board_size

        self._tiles = []
        for i in range(board_size):
            new_line = []
            for j in range(board_size):
                new_line.append(Tile(i, j))
            self._tiles.append(new_line)

        self._winner = None

    def from_string(string_input, board_size=11, bnf=True):
        """Loads a board from a string representation. If bnf=True, it will
        load a protocol-formatted string. Otherwise, it will load from a
        human-readable-formatted board.
        """

        b = Board(board_size=board_size)

        if (bnf):
            lines = string_input.split(",")
            for i, line in enumerate(lines):
                for j, char in enumerate(line):
                    b.set_tile_colour(i, j, Colour.from_char(char))
        else:
            lines = [line.strip() for line in string_input.split("\n")]
            for i, line in enumerate(lines):
                chars = line.split(" ")
                for j, char in enumerate(chars):
                    b.set_tile_colour(i, j, Colour.from_char(char))

        return b

    def has_ended(self):
        """Checks if the game has ended. It will attempt to find a red chain
        from top to bottom or a blue chain from left to right of the board.
        """

        # Red
        # for all top tiles, check if they connect to bottom
        for idx in range(self._board_size):
            tile = self._tiles[0][idx]
            if (not tile.is_visited() and
                tile.get_colour() == Colour.RED and
                    self._winner is None):
                self.DFS_colour(0, idx, Colour.RED)
        # Blue
        # for all left tiles, check if they connect to right
        for idx in range(self._board_size):
            tile = self._tiles[idx][0]
            if (not tile.is_visited() and
                tile.get_colour() == Colour.BLUE and
                    self._winner is None):
                self.DFS_colour(idx, 0, Colour.BLUE)

        # un-visit tiles
        self.clear_tiles()

        return self._winner is not None

    def clear_tiles(self):
        """Clears the visited status from all tiles."""

        for line in self._tiles:
            for tile in line:
                tile.clear_visit()

    def DFS_colour(self, x, y, colour):
        """A recursive DFS method that iterates through connected same-colour
        tiles until it finds a bottom tile (Red) or a right tile (Blue).
        """

        self._tiles[x][y].visit()

        # win conditions
        if (colour == Colour.RED):
            if (x == self._board_size-1):
                self._winner = colour
        elif (colour == Colour.BLUE):
            if (y == self._board_size-1):
                self._winner = colour
        else:
            return

        # end condition
        if (self._winner is not None):
            return

        # visit neighbours
        for idx in range(Tile.NEIGHBOUR_COUNT):
            x_n = x + Tile.I_DISPLACEMENTS[idx]
            y_n = y + Tile.J_DISPLACEMENTS[idx]
            if (x_n >= 0 and x_n < self._board_size and
                    y_n >= 0 and y_n < self._board_size):
                neighbour = self._tiles[x_n][y_n]
                if (not neighbour.is_visited() and
                        neighbour.get_colour() == colour):
                    self.DFS_colour(x_n, y_n, colour)

    def print_board(self, bnf=True):
        """Returns the string representation of a board. If bnf=True, the
        string will be formatted according to the communication protocol.
        """

        output = ""
        if (bnf):
            for line in self._tiles:
                for tile in line:
                    output += Colour.get_char(tile.get_colour())
                output += ","
            output = output[:-1]
        else:
            leading_spaces = ""
            for line in self._tiles:
                output += leading_spaces
                leading_spaces += " "
                for tile in line:
                    output += Colour.get_char(tile.get_colour()) + " "
                output += "\n"

        return output

    def get_winner(self):
        return self._winner

    def get_size(self):
        return self._board_size

    def get_tiles(self):
        return self._tiles

    def set_tile_colour(self, x, y, colour):
        self._tiles[x][y].set_colour(colour)


class HexDQN(NaiveAgent):
    HOST = "127.0.0.1"
    PORT = 1234

    def __init__(self):

        # super().__init__()
        self.board_size = 11
        self.board = self.board = [
                    [0]*self.board_size for i in range(self.board_size)]
        self.colour = "R"
        self.turn_count = 0

        self.state_size = 121
        self.action_size = 121
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 0.8  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.previousState = []
        self.randnum = 0
        self.prednum = 0

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(128, input_dim=self.state_size, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def board_string_to_vector(self,board_string):
        symbol_to_vector = {
            'R': 1,
            'B': 2,
            '0': 0
        }

        # Removing commas and converting each cell to a vector
        vectorized_board = [symbol_to_vector[str(cell)] for row in board_string for cell in row]

        return vectorized_board

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            choices = []
            for i in range(self.board_size):
                for j in range(self.board_size):
                    if self.board[i][j] == 0:
                        choices.append((i, j))
            act_values = choice(choices)
            self.randnum += 1
        else:
            numerical_board = self.board_string_to_vector(state)
            predict = self.model.predict(np.array(numerical_board).reshape((1, -1)))
            predict= np.argmax(predict[0])
            x = predict//11
            y = predict%11
            act_values = (x,y)
            self.prednum += 1
        return act_values # returns action

    def make_move(self):
        """Makes a move from the available pool of choices. If it can
  swap, chooses to do so 50% of the time.
  """   
        self.load('naivemodel')

        if np.random.rand() <= self.epsilon:
            choices = []
            for i in range(self.board_size):
                for j in range(self.board_size):
                    if self.board[i][j] == 0:
                        choices.append((i, j))
            act_values = choice(choices)
            self.randnum += 1
            # print('did a random')
        else:
            numerical_board = self.board_string_to_vector(self.board)
            # print('shape',numerical_board.shape)
            predict = self.model.predict(np.array(numerical_board).reshape((1, -1)))
            predict= np.argmax(predict[0])
            x = predict//11
            y = predict%11
            act_values = (x,y)
            self.prednum += 1
        print('random:',self.randnum)
        print('predict:',self.randnum)
        print('actval:',act_values)

        # self.s.sendall(bytes(f"{act_values[0]},{act_values[1]}\n", "utf-8"))
        self.board[act_values[0]][act_values[1]] = self.colour
        self.turn_count += 1

        self.remember(self.previousState, np.argmax(act_values[0]), 1, self.board, False)
        self.previousState = self.board

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

# if (__name__ == "__main__"):
#     agent = HexDQN()
#     agent.run()
agent1 = HexDQN()
board = Board()
# agent2 = HexDQN()
state_size = 121# Define state size based on Hex board representation
action_size = 121# Define action size based on possible moves in Hex
batch_size = 32

def calculate_reward(old_board, new_board, action):
    reward = 0.0

    print(str(new_board))
    Board.from_string(str(new_board))
    outcome = Board.has_ended()
    if outcome == Colour.RED:
        reward += 1.0
    elif outcome == Colour.BLUE:
        reward -= 1.0
    elif outcome == None:
        reward += 0.0  # or some small value to denote a tie

    # Evaluate strategic moves - this is simplified
    if is_valid_position(old_board, action):
        reward += 0.1  # small reward for making a progressive move
    else:
        reward += -100000

    return reward
def is_valid_position(old_board, action):
    if old_board[action[0]][action[1]] != 0:
        return False
    return True



for e in range(30):
    # reset state at the start of each game
    state = Board()
    # state = np.reshape(state, [1, state_size])

    for time in range(500):
        # agent takes action
        print(state.print_board())
        action = agent1.act()
        # apply action, get rewards and new state
        print(state)
        next_state = state
        next_state[action[0]][action[1]] = agent1.colour
        print(next_state)
        reward = calculate_reward(state,next_state,action)
        Board.from_string(str(next_state))
        outcome = Board.has_ended()
        if outcome == Colour.RED:
            done = True
        elif outcome == Colour.BLUE:
            done = True
        elif outcome == None:
            done = False  # or some small value to denote a tie
        
        # remember the previous state, action, reward, and done
        agent1.remember(state, action, reward, next_state, done)

        # make next_state the new current state
        state = next_state

        if done:
            # print the score and break out of the loop
            break

        if len(agent1.memory) > batch_size:
            agent1.replay(batch_size)
