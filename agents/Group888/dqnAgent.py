import socket
from random import choice
from time import sleep

import numpy as np
from keras.models import Sequential
from keras.src.layers import Dense, Conv2D, Flatten
from keras.optimizers import Adam


class HexAgent():
    """This class describes the default Hex agent. It will randomly send a
    valid move at each turn, and it will choose to swap with a 50% chance.
    """

    HOST = "127.0.0.1"
    PORT = 1234
    INFINITY = float("inf")

    def __init__(self, board_size=11):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect(("127.0.0.1", 1234))
        self.board_size = board_size
        self.board = [[0] * self.board_size for _ in range(self.board_size)]
        self.colour = ""
        self.max_depth = 1  # Depth
        self.evaluation_cache = {}

        # 定义模型相关的属性
        self.state_size = board_size * board_size
        self.action_size = board_size * board_size
        self.gamma = 0.95  # discount rate
        self.epsilon = 0.8  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.previousState = []
        self.randnum = 0
        self.prednum = 0

        # 创建模型
        self.model = self._build_model()

        # 使用随机数据通过模型，以便创建权重
        # dummy_input = np.zeros((1, self.state_size))
        # self.model.predict(dummy_input)

        # 现在加载权重
        self.model.load_weights("C:/Users/zhy20/Desktop/Hex_AI/agent1_weights.h5")

    def load(self, name):
        self.model.load_weights(name)

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()   
        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(11, 11, 1)))
        for _ in range(9):  
           model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(Flatten()) 
        model.add(Dense(self.action_size, activation='sigmoid'))  
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))  
        return model
    

    def run(self):
        """Reads data until it receives an END message or the socket closes."""

        while True:
            data = self.s.recv(1024)
            if not data:
                break
            # print(f"{self.colour} {data.decode('utf-8')}", end="")
            if (self.interpret_data(data)):
                break

        # print(f"Naive agent {self.colour} terminated")

    def interpret_data(self, data):
        """Checks the type of message and responds accordingly. Returns True
        if the game ended, False otherwise.
        """

        messages = data.decode("utf-8").strip().split("\n")
        messages = [x.split(";") for x in messages]
        # print(messages)
        for s in messages:
            if s[0] == "START":
                self.board_size = int(s[1])
                self.colour = s[2]
                self.board = [
                    [0]*self.board_size for i in range(self.board_size)]

                if self.colour == "R":
                    self.make_move()

            elif s[0] == "END":
                return True

            elif s[0] == "CHANGE":
                if s[3] == "END":
                    return True

                elif s[1] == "SWAP":
                    self.colour = self.opp_colour()
                    if s[3] == self.colour:
                        self.make_move()

                elif s[3] == self.colour:
                    action = [int(x) for x in s[1].split(",")]
                    self.board[action[0]][action[1]] = self.opp_colour()

                    self.make_move()

        return False

    def board_to_input(self):
        """将当前棋盘转换为模型输入。"""
        flat_board = np.array(self.board).flatten()
        input_board = np.where(flat_board == 'R', 1, flat_board)
        input_board = np.where(input_board == 'B', -1, input_board)
        input_board = np.where(input_board == 0, 0, input_board)
        return np.array(input_board, dtype='float32').reshape((1, 11, 11, 1))

    def make_move(self):
        """使用训练好的模型来选择最佳移动。"""
        current_state = self.board_to_input()
        q_values = self.model.predict(current_state)[0]

        # 将Q值最高的动作转换为棋盘上的位置
        possible_moves = self.get_possible_moves()
        best_move = None
        max_q_value = -float('inf')
        for move in possible_moves:
            idx = move[0] * self.board_size + move[1]
            if q_values[idx] > max_q_value:
                max_q_value = q_values[idx]
                best_move = move

        if best_move:
            self.execute_move(best_move)
        else:
            print("No valid move found!")


    def execute_move(self, move):
        self.board[move[0]][move[1]] = self.colour
        self.s.sendall(bytes(f"{move[0]},{move[1]}\n", "utf-8"))


    def opp_colour(self):
        """Returns the char representation of the colour opposite to the
        current one.
        """
        if self.colour == "R":
            return "B"
        elif self.colour == "B":
            return "R"
        else:
            return "None"

    def get_possible_moves(self):
        moves = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i][j] == 0:
                    moves.append((i, j))
        return moves


if (__name__ == "__main__"):
    agent = HexAgent()
    agent.run()
