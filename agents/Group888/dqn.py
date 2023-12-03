import numpy as np
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from random import choice

from BestAgent import NaiveAgent

class HexDQN(NaiveAgent):
    HOST = "127.0.0.1"
    PORT = 1234

    # print('np')

    def __init__(self):
        # print('np')
        super().__init__()
        # self.s = socket.socket(
        #     socket.AF_INET, socket.SOCK_STREAM
        # )

        # self.s.connect((self.HOST, self.PORT))

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

    # def run(self):
    #     """Reads data until it receives an END message or the socket closes."""

    #     while True:
    #         data = self.s.recv(1024)
    #         if not data:
    #             break
    #         # print(f"{self.colour} {data.decode('utf-8')}", end="")
    #         if (self.interpret_data(data)):
    #             break

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

    def make_move(self):
        """Makes a move from the available pool of choices. If it can
  swap, chooses to do so 50% of the time.
  """   
        self.load('naivemodel')
        # print('working')
        # # print(f"{self.colour} making move")
        # if self.colour == "B" and self.turn_count == 0:
        #     if choice([0, 1]) == 1:
        #         self.s.sendall(bytes("SWAP\n", "utf-8"))
        #     else:
        #         # same as below
        #         choices = []
        #         for i in range(self.board_size):
        #             for j in range(self.board_size):
        #                 if self.board[i][j] == 0:
        #                     choices.append((i, j))
        #         pos = choice(choices)
        #         self.s.sendall(bytes(f"{pos[0]},{pos[1]}\n", "utf-8"))
        #         self.board[pos[0]][pos[1]] = self.colour
        # else:
        #     choices = []
        #     for i in range(self.board_size):
        #         for j in range(self.board_size):
        #             if self.board[i][j] == 0:
        #                 choices.append((i, j))
        #     pos = choice(choices)
        # print('working')
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

        self.s.sendall(bytes(f"{act_values[0]},{act_values[1]}\n", "utf-8"))
        self.board[act_values[0]][act_values[1]] = self.colour
        self.turn_count += 1

        self.remember(self.previousState, np.argmax(act_values[0]), 1, self.board, False)
        self.previousState = self.board

        self.save('naivemodel')
    # def replay(self, batch_size):
    #     minibatch = random.sample(self.memory, batch_size)
    #     for state, action, reward, next_state, done in minibatch:
    #         target = reward
    #         if not done:
    #             target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
    #         target_f = self.model.predict(state)
    #         target_f[0][action] = target
    #         self.model.fit(state, target_f, epochs=1, verbose=0)
    #     if self.epsilon > self.epsilon_min:
    #         self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

if (__name__ == "__main__"):
    agent = HexDQN()
    agent.run()