import numpy as np
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
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
        self.colour = Colour.RED
        self.turn_count = 0

        self.state_size = 121
        self.action_size = 121
        self.memory = deque(maxlen=50000)
        self.current_game_memory = []

        # self.gamma = 0.95  # discount rate
        self.gamma = 1  # discount rate
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
        # model = Sequential()
        # model.add(Dense(128, input_dim=self.state_size, activation='relu'))
        # model.add(Dense(128, activation='relu'))
        # model.add(Dense(self.action_size, activation='linear'))
        # model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        # return model

        model = Sequential()   
        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(11, 11, 1)))
        for _ in range(9):  
           model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(Flatten()) 
        model.add(Dense(self.action_size, activation='sigmoid'))  
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))  
        return model

    def remember(self, state, action, reward, next_state, done):
        self.current_game_memory.append((state, action, reward, next_state, done))

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
        state = state.split(",")
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
            predict = self.model.predict(np.array(numerical_board).reshape((1, 11, 11, 1)))
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
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = self.board_string_to_vector(state.print_board().split(","))
            state = np.array(state).reshape((1, 11, 11, 1))
            next_state = self.board_string_to_vector(next_state.print_board().split(","))
            next_state = np.array(next_state).reshape((1, 11, 11, 1))
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            # print(target_f[0])
            action = action[0]*11+action[1]
            # print(action)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    def end_game(self, win):
        # 游戏结束时调用此方法来更新记忆并将其添加到主记忆库
        feedback = 1 if win else -1
        for state, action, reward, next_state, done in self.current_game_memory:
            # 更新奖励并添加到主记忆库
            self.memory.append((state, action, reward+feedback, next_state, done))
        self.current_game_memory.clear()  # 清空当前游戏的记忆

# if (__name__ == "__main__"):
#     agent = HexDQN()
#     agent.run()
agent1 = HexDQN()
agent1.colour = Colour.RED
agent2 = HexDQN()
agent2.colour = Colour.BLUE
board = Board()
state_size = 121# Define state size based on Hex board representation
action_size = 121# Define action size based on possible moves in Hex
batch_size = 32

def calculate_reward(old_board, new_board, action, player):
    reward = 0.0
    valid_move_flag = True
    # outcome = new_board.has_ended()
    # if outcome == Colour.RED:
    #     if player == 1:
    #         reward += 1.0
    #     else:
    #         reward -= 1.0
    # elif outcome == Colour.BLUE:
    #     if player == 2:
    #         reward += 1.0
    #     else:
    #         reward -= 1.0
    # elif outcome == None:
    #     reward += 0.0  # or some small value to denote a tie

    # Evaluate strategic moves - this is simplified
    if is_valid_position(old_board, action):
        # reward += 0.1  # small reward for making a progressive move
        reward += 0
    else:
        reward += -100  # big penalty for making an invalid move
        valid_move_flag = False

    return reward, valid_move_flag


def is_valid_position(old_board, action):
    old_board = old_board.get_tiles()
    print('Colour',old_board[action[0]][action[1]].colour)
    if old_board[action[0]][action[1]].colour != None:
        return False
    return True

redwin = 0
bluewin = 0

for e in range(200):
    # reset state at the start of each game
    state = Board()
    done = None
    print('blue:',bluewin)
    print('red:',redwin)
    while not done:
        if (len(agent1.memory) + len(agent1.current_game_memory)) % batch_size == 0:
            agent1.replay(batch_size)
        if (len(agent2.memory) + len(agent2.current_game_memory)) % batch_size == 0:
            agent2.replay(batch_size)

        # agent takes action
        state_string = state.print_board()
        action1 = agent1.act(state.print_board())
        # apply action, get rewards and new state
        next_state = Board.from_string(state_string,11,bnf=True)
        next_state.set_tile_colour(action1[0],action1[1],agent1.colour)
        reward1, valid_move_flag1 = calculate_reward(state,next_state,action1,agent1)
        print(state.print_board(bnf=False))

        # Board.from_string(str(next_state))
        outcome = next_state.has_ended()
        if outcome == Colour.RED:
            redwin +=1
            done = True
        elif outcome == Colour.BLUE or valid_move_flag1 == False:
            bluewin +=1
            done = True
        elif outcome == None:
            done = False  # or some small value to denote a tie
        print('player:', agent1.colour)
        print('action1:',action1)
        print('reward1:',reward1)


        if done:
            agent1.end_game(win = valid_move_flag1)
            agent2.end_game(win = not valid_move_flag1)
            break


        # remember the previous state, action, reward, and done
        
        agent1.remember(state, action1, reward1, next_state, done)

        # make next_state the new current state
        state = Board.from_string(next_state.print_board(),11,bnf=True)

        print(state.print_board(bnf=False))

        state_string = state.print_board()
        action2 = agent2.act(state_string)
        # apply action, get rewards and new state
        next_state = Board.from_string(state_string,11,bnf=True)
        next_state.set_tile_colour(action2[0],action2[1],agent2.colour)
        reward2, valid_move_flag2 = calculate_reward(state,next_state,action2,2)
        # Board.from_string(str(next_state))
        outcome = next_state.has_ended()
        if outcome == Colour.RED or valid_move_flag2 == False:
            redwin +=1
            done = True
        elif outcome == Colour.BLUE:
            bluewin +=1
            done = True
        elif outcome == None:
            done = False  # or some small value to denote a tie
        
        # remember the previous state, action, reward, and done
        print(state.print_board(bnf=False))

        print('player:', agent2.colour)
        print('action2:',action2)
        print('reward2:',reward2)
        agent2.remember(state, action2, reward2, next_state, done)

        # make next_state the new current state
        state = Board.from_string(next_state.print_board(),11,bnf=True)


        if done:
            agent1.end_game(win = not valid_move_flag2)
            agent2.end_game(win = valid_move_flag2)
            break
        


agent2.save("agent2_weights.h5")
agent1.save("agent1_weights.h5")

