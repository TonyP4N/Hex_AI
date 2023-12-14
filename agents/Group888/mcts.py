import socket
from copy import deepcopy
import random
import math
import time
from time import sleep


class Node:
    def __init__(self, board, move, parent=None):
        self.board = board
        self.move = move
        self.parent = parent
        self.children = []
        self.wins = 0
        self.visits = 0
        self.untried_moves = self.get_possible_moves(board)
        self.player_just_moved = self.opp_colour(parent.player_just_moved) if parent else "R"

    def get_possible_moves(self, board):
        return [(i, j) for i in range(len(board)) for j in range(len(board)) if board[i][j] == 0]

    def uct_select_child(self):
        log_visits = math.log(self.visits)
        return max(self.children, key=lambda c: c.wins / c.visits + math.sqrt(2 * log_visits / c.visits))

    def add_child(self, m, b):
        n = Node(move=m, parent=self, board=b)
        self.untried_moves.remove(m)
        self.children.append(n)
        return n

    def update(self, result):
        self.visits += 1
        self.wins += result

    def opp_colour(self, colour):
        return "B" if colour == "R" else "R"


class MCTSAgent():
    """This class describes the default Hex agent. It will randomly send a
    valid move at each turn, and it will choose to swap with a 50% chance.
    """

    HOST = "127.0.0.1"
    PORT = 1234

    def __init__(self, board_size=11, time_limit=2):
        print('np')
        self.s = socket.socket(
            socket.AF_INET, socket.SOCK_STREAM
        )

        self.s.connect((self.HOST, self.PORT))

        self.board_size = board_size
        self.board = [[0] * self.board_size for _ in range(self.board_size)]
        self.colour = ""
        self.turn_count = 0
        self.time_limit = time_limit

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
                    self.make_move()

            elif s[0] == "END":
                return True

            elif s[0] == "CHANGE":
                if s[3] == "END":
                    return True

                elif s[1] == "SWAP":
                    self.colour = self.opp_colour(self.colour)
                    if s[3] == self.colour:
                        self.make_move()

                elif s[3] == self.colour:
                    action = [int(x) for x in s[1].split(",")]
                    self.board[action[0]][action[1]] = self.opp_colour(self.colour)

                    self.make_move()

        return False

    def make_move(self):
        root = Node(board=self.board, move=None)

        end_time = time.time() + self.time_limit
        while time.time() < end_time:
            node = root
            board = deepcopy(self.board)

            # Selection
            while node.untried_moves == [] and node.children != []:
                node = node.uct_select_child()
                board = self.get_next_board(board, node.move, node.player_just_moved)

            # Expansion
            if node.untried_moves != []:
                m = random.choice(node.untried_moves)
                board = self.get_next_board(board, m, node.player_just_moved)
                node = node.add_child(m, board)

            # Simulation
            result = self.simulation(deepcopy(board), node.player_just_moved)

            while node:
                node.update(result)
                node = node.parent

        self.execute_move(sorted(root.children, key=lambda c: c.visits)[-1].move)

    def execute_move(self, move):
        self.board[move[0]][move[1]] = self.colour
        self.s.sendall(bytes(f"{move[0]},{move[1]}\n", "utf-8"))

    def opp_colour(self, colour):
        return "B" if colour == "R" else "R"

    def get_possible_moves(self, board):
        p_moves = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                if board[i][j] == 0:
                    p_moves.append((i, j))
        return p_moves

    def get_next_board(self, board, move, colour):
        next_board = deepcopy(board)
        next_board[move[0]][move[1]] = colour
        return next_board

    def simulation(self, board, player_just_moved):
        moves = self.get_possible_moves(board)
        player_to_move = self.opp_colour(player_just_moved)
        while moves:
            move = random.choice(moves)
            board = self.get_next_board(board, move, player_to_move)
            moves = self.get_possible_moves(board)
            player_to_move = self.opp_colour(player_to_move)

        return self.evaluate_board(board, player_just_moved)

    def evaluate_board(self, board, colour):
        visited = set()

        def dfs(board, x, y, target):
            if x < 0 or x >= len(board) or y < 0 or y >= len(board) or board[x][y] != target:
                return False
            if (target == "R" and y == len(board) - 1) or (target == "B" and x == len(board) - 1):
                return True
            if (x, y) in visited:
                return False
            visited.add((x, y))

            directions = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
            for dx, dy in directions:
                if dfs(board, x + dx, y + dy, target):
                    return True
            return False

        for start_y in range(len(board)):
            if dfs(board, 0, start_y, "R"):
                return 1 if colour == "R" else -1
        for start_x in range(len(board)):
            if dfs(board, start_x, 0, "B"):
                return 1 if colour == "B" else -1

        return 0


if (__name__ == "__main__"):
    agent = MCTSAgent()
    agent.run()
