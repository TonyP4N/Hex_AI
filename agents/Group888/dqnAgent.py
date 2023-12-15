import socket
import time
from random import choice
from time import sleep
import tensorflow as tf
import sys

sys.path.append("..")
from inputFormat import *

import numpy as np
from keras.models import Sequential
from keras.src.layers import Dense
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
        self.model = tf.keras.models.load_model("dqn_train_models/qlearn_5000_episode")
        self.swap_flag = True
        # self.search()

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
                    [0] * self.board_size for i in range(self.board_size)]

                if self.colour == "R":
                    self.swap_flag = False
                    self.search()
                    self.make_move()

            elif s[0] == "END":
                return True

            elif s[0] == "CHANGE":
                if s[3] == "END":
                    return True

                elif s[1] == "SWAP":
                    self.colour = self.opp_colour()
                    if s[3] == self.colour:
                        self.search()
                        self.make_move()

                elif s[3] == self.colour:
                    action = [int(x) for x in s[1].split(",")]
                    self.board[action[0]][action[1]] = self.opp_colour()
                    if self.swap_flag:
                        self.swap_move()
                    else:
                        self.search()
                        self.make_move()

        return False

    def stateToInput(self):
        board = self.board
        ret = new_game(len(board))
        padding = (input_size - len(board) + 1) // 2
        for i in range(len(board)):
            for j in range(len(board)):
                if board[i][j] == 'R':
                    play_cell(ret, (i + padding, j + padding), white)
                elif board[i][j] == 'B':
                    play_cell(ret, (i + padding, j + padding), black)
        return ret

    def search(self, time_budget=1):
        """
        Compute resistance for all moves in current state.
        """
        state = self.stateToInput()
        # get equivalent white to play game if black to play
        toplay = white if self.colour == "B" else "R"
        if(toplay == "R"):
            state = mirror_game(state)
        played = np.logical_or(state[white, padding:boardsize + padding, padding:boardsize + padding], \
                               state[black, padding:boardsize + padding, padding:boardsize + padding]).flatten()
        state = tf.convert_to_tensor(state, dtype=tf.float32)
        self.scores = self.model(tf.expand_dims(state, axis=0))
        # set value of played cells impossibly low so they are never picked
        played_indices = np.where(played)[0] // 11
        played_indices2 = np.where(played)[0] % 11
        # print(played_indices)
        # print(played_indices2)
        # Assuming 'scores' is your TensorFlow tensor
        self.scores = self.scores.numpy()  # Convert to NumPy array
        # set value of played cells impossibly low so they are never picked
        for x in range(len(played_indices)):
            self.scores[0][played_indices[x]][played_indices2[x]] = -1000

    # def board_to_input(self):
    #     """将当前棋盘转换为模型输入。"""
    #     flat_board = np.array(self.board).flatten()
    #     input_board = np.where(flat_board == 'R', 1, flat_board)
    #     input_board = np.where(input_board == 'B', -1, input_board)
    #     input_board = np.where(input_board == 0, 0, input_board)
    #     return np.array(input_board, dtype='float32').reshape((1, -1))

    def make_move(self):
        """使用训练好的模型来选择最佳移动。"""
        move = np.unravel_index(self.scores.argmax(), (boardsize, boardsize))
        # correct move for smaller boardsizes
        # flip returned move if black to play to get move in actual game
        self.execute_move(move)
        # current_state = self.board_to_input()
        # q_values = self.model.predict(current_state)[0]

        # # 将Q值最高的动作转换为棋盘上的位置
        # possible_moves = self.get_possible_moves()
        # best_move = None
        # max_q_value = -float('inf')
        # for move in possible_moves:
        #     idx = move[0] * self.board_size + move[1]
        #     if q_values[idx] > max_q_value:
        #         max_q_value = q_values[idx]
        #         best_move = move

        # if best_move:
        #     self.execute_move(best_move)
        # else:
        #     print("No valid move found!")

    def execute_move(self, move):
        print(move)
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
