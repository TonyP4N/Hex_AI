import socket
import tensorflow as tf
import numpy as np
from inputFormat import *
from resistance import *
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
        self.swap_flag = True
        self.model = tf.keras.models.load_model("dqn_train_models/qlearn_5000_episode")
        self.board_weights = [[0] * self.board_size for _ in range(self.board_size)]



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
        print(messages)
        for s in messages:
            if s[0] == "START":
                self.board_size = int(s[1])
                self.colour = s[2]
                self.board = [
                    [0] * self.board_size for i in range(self.board_size)]

                if self.colour == "R":
                    self.swap_flag = False
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
                    # if self.swap_flag:
                    #     self.swap_flag = False
                    #     self.swap_move()
                    # else:
                    self.search()
                    self.make_move()

        return False

    def make_move(self):
        """Use Alpha-Beta pruning to find the best move."""
        best_score = -self.INFINITY
        best_move = None
        alpha = -self.INFINITY
        beta = self.INFINITY

        for move in self.get_possible_moves():
            self.make_temporary_move(move)
            score = self.alphabeta(0, alpha, beta, False)
            self.undo_move(move)

            if score > best_score:
                best_score = score
                best_move = move
                alpha = max(alpha, score)
        print("score", best_score, "move", best_move)
        self.execute_move(best_move)

    # def swap_move(self):
    #     board = self.board
    #     # 00 01 02 10 11 20
    #     # 1010 1009 1008 0910 0909 0810
    #     not_swap = [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [2, 0], [10, 10], [10, 9], [10, 8], [9, 10], [9, 9],
    #                 [8, 10]]
    #     for i in range(self.board_size):
    #         for j in range(self.board_size):
    #             if board[i][j] != 0:
    #                 if [i, j] in not_swap:
    #                     self.make_move()
    #                 else:
    #                     self.colour = self.opp_colour()
    #                     self.s.sendall(bytes("SWAP\n", "utf-8"))


    def alphabeta(self, depth, alpha, beta, maximizingPlayer):
        """Alpha-Beta pruning logic."""
        # 生成当前棋盘的哈希值作为缓存键
        board_hash = self.hash_board()
        if board_hash in self.evaluation_cache:
            return self.evaluation_cache[board_hash]

        if depth == self.max_depth or self.is_game_over():
            score = self.evaluate_board()
            self.evaluation_cache[board_hash] = score
            return score
        if maximizingPlayer:
            max_eval = -self.INFINITY
            for move in self.get_possible_moves():
                self.make_temporary_move(move)
                eval = self.alphabeta(depth + 1, alpha, beta, False)
                self.undo_move(move)

                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = self.INFINITY
            for move in self.get_possible_moves():
                self.make_temporary_move(move)
                eval = self.alphabeta(depth + 1, alpha, beta, True)
                self.undo_move(move)

                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval

    def calculate_partial_line_score(self, colour):
        """评估形成部分连线的得分"""
        score = 0
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i][j] == colour:
                    # 为每个连续的、同色的棋子组增加分数
                    score += self.evaluate_partial_line(i, j, colour)
        return score

    def evaluate_partial_line(self, i, j, colour):
        """评估单个位置的连线潜力"""
        partial_line_score = 0
        # 检查各个方向上的连线潜力
        directions = [(1, 0), (0, 1), (1, 1), (-1, 1), (1, -1), (-1, 0)]
        for di, dj in directions:
            line_length = 1
            next_i, next_j = i + di, j + dj
            end_open = 0  # 连线端点的开放度
            while 0 <= next_i < self.board_size and 0 <= next_j < self.board_size and self.board[next_i][
                next_j] == colour:
                line_length += 1
                next_i += di
                next_j += dj
            # 检查连线两端是否开放
            if self.is_open_end(i - di, j - dj) or self.is_open_end(next_i, next_j):
                end_open = 1
            # 根据连线长度和端点开放度增加分数
            partial_line_score += line_length + end_open
        return partial_line_score

    def is_open_end(self, i, j):
        """检查给定位置是否为开放端点"""
        return 0 <= i < self.board_size and 0 <= j < self.board_size and self.board[i][j] == 0

    # 其他函数保持不变

    def evaluate_board(self):
        # Use DQN board weights in evaluation
        my_score = self.calculate_dijkstra_score(self.colour) + self.calculate_partial_line_score(self.colour) + np.sum(
            self.board_weights)
        # print(np.sum(self.board_weights))
        # my_score = self.calculate_dijkstra_score(self.colour) + self.calculate_partial_line_score(self.colour) + \
        #            np.sum(score(self.stateToInput(),white if self.colour == 'B' else black))
        #
        # opponent_score = self.calculate_dijkstra_score(self.opp_colour()) + self.calculate_partial_line_score(
        #     self.opp_colour()) -np.sum(score(self.stateToInput(),white if self.opp_colour() == 'B' else black))
        opponent_score = self.calculate_dijkstra_score(self.opp_colour()) + self.calculate_partial_line_score(
            self.opp_colour()) - np.sum(self.board_weights)
        # print(np.sum(self.board_weights))
        return my_score - opponent_score
    def search(self):
        """
        Compute resistance for all moves in current state.
        """
        state = self.stateToInput()
        state = tf.convert_to_tensor(state, dtype=tf.float32)
        scores = self.model(tf.expand_dims(state, axis=0)).numpy()

        # Update board weights
        for i in range(self.board_size):
            for j in range(self.board_size):
                self.board_weights[i][j] = scores[0][i][j]

        # set value of played cells impossibly low so they are never picked
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i][j] != 0:
                    self.board_weights[i][j] = -2

    def stateToInput(self):
        board = self.board
        ret = new_game(len(board))
        padding = (11 - len(board) + 1) // 2
        for i in range(len(board)):
            for j in range(len(board)):
                if board[i][j] == 'R':
                    play_cell(ret, (i + padding, j + padding), white)
                elif board[i][j] == 'B':
                    play_cell(ret, (i + padding, j + padding), black)
        return ret

    def DQN_board(self):
        self.search()
        return np.max(self.scores)

    def calculate_dijkstra_score(self, colour):

        # Dijkstra算法
        distance = [[self.INFINITY] * self.board_size for _ in range(self.board_size)]
        if colour == "R":
            start = 0
            end = self.board_size - 1
        else:
            start = 1
            end = self.board_size

        # 初始化距离矩阵
        distance = [[self.INFINITY] * self.board_size for _ in range(self.board_size)]
        # 初始化已访问矩阵
        visited = [[False] * self.board_size for _ in range(self.board_size)]
        # 初始化前驱矩阵
        predecessor = [[None] * self.board_size for _ in range(self.board_size)]
        # 初始化起点
        distance[start][0] = 0
        # 初始化队列
        queue = []
        queue.append((start, 0))
        while queue:
            # 取出队首元素
            u, v = queue.pop(0)
            # 标记已访问
            visited[u][v] = True
            # 遍历邻居
            for i in range(self.board_size):
                for j in range(self.board_size):
                    if self.board[i][j] == colour:
                        # 计算距离
                        if i == u:
                            dist = distance[u][v] + 1
                        elif j == v:
                            dist = distance[u][v] + 1
                        else:
                            dist = distance[u][v] + 2
                        # 更新距离
                        if dist < distance[i][j]:
                            distance[i][j] = dist
                            predecessor[i][j] = (u, v)
                        # 入队
                        if not visited[i][j]:
                            queue.append((i, j))

        score = min([min(row) for row in distance])  # 选择距离矩阵中的最小值
        return score if score != self.INFINITY else 0

    def hash_board(self):
        # 将当前棋盘状态转换为哈希值
        return hash(tuple(tuple(row) for row in self.board))

    def get_possible_moves(self):
        moves = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i][j] == 0:
                    moves.append((i, j))
        return moves

    def make_temporary_move(self, move):
        self.board[move[0]][move[1]] = self.colour

    def undo_move(self, move):
        self.board[move[0]][move[1]] = 0

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

    def is_game_over(self):
        return self.check_win("R") or self.check_win("B")

    def check_win(self, colour):
        """Returns True if the given colour has won, False otherwise."""
        if colour == "R":
            # 红色玩家赢得游戏的条件是水平方向上连线
            return self.check_horizontal_win(colour)
        elif colour == "B":
            # 蓝色玩家赢得游戏的条件是垂直方向上连线
            return self.check_vertical_win(colour)

    # 检查水平方向连线（红色玩家）
    def check_horizontal_win(self, colour):
        # 遍历每一行的起始位置
        for i in range(self.board_size):
            if self.board[i][0] == colour and self.check_horizontal_win_from(i, 0, colour):
                return True
        return False

    # 检查垂直方向连线（蓝色玩家）
    def check_vertical_win(self, colour):
        # 遍历每一列的起始位置
        for j in range(self.board_size):
            if self.board[0][j] == colour and self.check_vertical_win_from(0, j, colour):
                return True
        return False

    def check_horizontal_win_from(self, i, j, colour):
        """Returns True if the given colour has won horizontally from the
        given position, False otherwise.
        """
        if j == self.board_size:
            return True
        if self.board[i][j] != colour:
            return False
        return self.check_horizontal_win_from(i, j + 1, colour)

    def check_vertical_win_from(self, i, j, colour):
        """Returns True if the given colour has won vertically from the
        given position, False otherwise.
        """
        if i == self.board_size:
            return True
        if self.board[i][j] != colour:
            return False
        return self.check_vertical_win_from(i + 1, j, colour)

    def execute_move(self, move):
        self.board[move[0]][move[1]] = self.colour
        self.s.sendall(bytes(f"{move[0]},{move[1]}\n", "utf-8"))


if (__name__ == "__main__"):
    agent = HexAgent()
    agent.run()
