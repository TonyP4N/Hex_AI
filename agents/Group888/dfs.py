def check_win(self, colour):
    """检查指定颜色是否获胜"""
    visited = set()
    for i in range(self.board_size):
        if colour == "R" and self.board[i][0] == colour:
            if self.dfs(i, 0, colour, visited):
                return True
        if colour == "B" and self.board[0][i] == colour:
            if self.dfs(0, i, colour, visited):
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