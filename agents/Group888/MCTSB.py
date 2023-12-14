import copy
import math
import socket
from random import choice


def make_simulated_move(state, move, colour):
    new_state = copy.deepcopy(state)
    new_state[move[0]][move[1]] = colour
    return new_state


class MCTSAgent():
    HOST = "127.0.0.1"
    PORT = 1234

    def __init__(self, board_size=11):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect((self.HOST, self.PORT))
        self.board_size = board_size
        self.board = [[0]*self.board_size for _ in range(self.board_size)]
        self.colour = ""
        self.turn_count = 0

    def run(self):
        while True:
            data = self.s.recv(1024)
            if not data:
                break
            if self.interpret_data(data):
                break

    def interpret_data(self, data):
        messages = data.decode("utf-8").strip().split("\n")
        messages = [x.split(";") for x in messages]
        for s in messages:
            if s[0] == "START":
                self.board_size = int(s[1])
                self.colour = s[2]
                self.board = [[0]*self.board_size for _ in range(self.board_size)]
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

    def make_move(self):
        best_move = self.run_mcts()
        if best_move is not None:  # Check if a best move is found
            self.board[best_move[0]][best_move[1]] = self.colour
            self.s.sendall(f"{best_move[0]},{best_move[1]}\n".encode("utf-8"))
            self.turn_count += 1
        else:
            print("No best move found.")

    def run_mcts(self, simulations=1000):
        root = Node(state=self.board, parent=None, move=None, colour=self.colour)
        for _ in range(simulations):
            node = self.select(root)
            winner = self.simulate(node.state)
            self.backpropagate(node, winner)
        return self.best_move(root)

    def get_possible_moves(self):
        """Returns a list of possible moves."""
        moves = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i][j] == 0:
                    moves.append((i, j))
        return moves

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

    # MCTS的选择阶段
    def select(self, node):
        while not self.is_terminal(node.state) and node.is_fully_expanded():
            node = self.find_best_node(node)
        return self.expand(node) if not self.is_terminal(node.state) else node

    def get_legal_moves(self, state):
        moves = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                if state[i][j] == 0:
                    moves.append((i, j))
        return moves

    def expand(self, node):
        moves = self.get_legal_moves(node.state)
        for move in moves:
            if move not in node.children:
                new_state = make_simulated_move(node.state, move, node.colour)

                new_node = Node(state=new_state, parent=node, move=move, colour=self.toggle_colour(node.colour))
                node.children[move] = new_node
                return new_node
        return node

    def toggle_colour(self, colour):
        if colour == "R":
            return "B"
        elif colour == "B":
            return "R"

    # MCTS的模拟阶段
    def simulate(self, state):
        # 随机模拟直到游戏结束
        current_state = state
        while not self.is_terminal(current_state):
            legal_moves = self.get_legal_moves(current_state)
            if not legal_moves:  # Check if there are legal moves left
                break
            move = choice(legal_moves)
            current_state = make_simulated_move(current_state, move, self.toggle_colour(self.colour))
        return self.get_winner(current_state)

    # MCTS的反向传播阶段
    def backpropagate(self, node, winner):
        while node is not None:
            node.visits += 1
            if node.move and winner == self.colour:
                node.wins += 1
            node = node.parent

    def best_move(self, root):
        # 选择访问次数最多的走法
        best_move, max_visits = None, -1
        for move, node in root.children.items():
            if node.visits > max_visits:
                best_move, max_visits = move, node.visits
        return best_move

    def find_best_node(self, node):
        # 使用UCT（Upper Confidence Bound applied to Trees）公式
        best_node, max_value = None, -float('inf')
        for child in node.children.values():
            ucb_value = child.wins / child.visits + math.sqrt(2) * math.sqrt(math.log(node.visits) / child.visits)
            if ucb_value > max_value:
                best_node, max_value = child, ucb_value
        return best_node

    def is_terminal(self, state):
        return self.get_winner(state) is not None

    def get_winner(self, state):
        if self.check_win("R"):
            return "R"
        if self.check_win("B"):
            return "B"
        return None

    def check_win(self, colour):
        """检查指定颜色是否获胜"""
        visited = set()
        for i in range(self.board_size):
            if colour == "R" and self.board[0][i] == colour:  # Change here for "R"
                if self.dfs(0, i, colour, visited):  # Change here for "R"
                    return True
            if colour == "B" and self.board[i][0] == colour:
                if self.dfs(i, 0, colour, visited):
                    return True
        return False

    def dfs(self, i, j, colour, visited):
        """使用深度优先搜索检查获胜条件"""
        if (i, j) in visited:
            return False
        visited.add((i, j))

        if colour == "R" and j == self.board_size - 1:
            return True
        if colour == "B" and i == self.board_size - 1:
            return True

        for di, dj in [(1, 0), (0, 1), (-1, 0), (0, -1), (1, -1), (-1, 1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < self.board_size and 0 <= nj < self.board_size:
                if self.board[ni][nj] == colour and self.dfs(ni, nj, colour, visited):
                    return True
        return False

    def get_legal_moves(self, state):
        moves = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                if state[i][j] == 0:
                    moves.append((i, j))
        return moves

    def make_simulated_move(self, state, move):
        new_state = copy.deepcopy(state)
        new_state[move[0]][move[1]] = self.colour
        return new_state


class Node:
    def __init__(self, state, parent, move, colour):
        self.board_size = 11
        self.state = state
        self.parent = parent
        self.move = move
        self.children = {}
        self.wins = 0
        self.visits = 0
        self.colour = colour

    def is_fully_expanded(self):
        return len(self.children) == len(self.get_legal_moves(self.state, self.colour))

    def get_legal_moves(self, state, colour):
        moves = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                if state[i][j] == 0:
                    moves.append((i, j))
        return moves

    def make_simulated_move(self, state, move, colour):
        new_state = copy.deepcopy(state)
        new_state[move[0]][move[1]] = colour
        return new_state


if (__name__ == "__main__"):
    agent = MCTSAgent()
    agent.run()