import numpy as np
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

import BestAgent

class HexDQN(BestAgent.NaiveAgent):
    HOST = "127.0.0.1"
    PORT = 1234

    def __init__(self, state_size, action_size):
        super().__init__(board_size=11)

        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.previousState = []

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def make_move(self):
        """Makes a move from the available pool of choices. If it can
  swap, chooses to do so 50% of the time.
  """

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

        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(self.board)

        self.s.sendall(bytes(f"{act_values[0]},{act_values[1]}\n", "utf-8"))
        self.board[act_values[0], act_values[1]] = self.colour
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
