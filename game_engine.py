import random

class MNKBoard:
    """
    Represents the MNK Game board (m-rows, n-cols, k-in-a-row).
    """
    def __init__(self, m, n, k):
        self.m = m  # Rows
        self.n = n  # Cols
        self.k = k  # Win condition
        self.board = [[0 for _ in range(n)] for _ in range(m)]
        self.empty_cells = set((r, c) for r in range(m) for c in range(n))
        self.occupied_cells = set()
        self.last_move = None
        self.winner = None
        self.init_zobrist()

    def make_move(self, row, col, player):
        """
        Attempts to make a move for the given player at (row, col).
        Returns True if successful, False if the cell is occupied.
        """
        if self.board[row][col] != 0:
            return False
            
        self.board[row][col] = player
        self.empty_cells.remove((row, col))
        self.occupied_cells.add((row, col))
        self.update_hash(row, col, player)
        self.last_move = (row, col)
        
        if self.check_win(row, col, player):
            self.winner = player
        
        return True

    def undo_move(self, row, col):
        """
        Undoes the move at (row, col).
        """
        p = self.board[row][col]
        self.board[row][col] = 0
        self.empty_cells.add((row, col))
        self.occupied_cells.remove((row, col))
        
        if p != 0:
            self.update_hash(row, col, p)
            
        self.winner = None
        # Note: last_move cannot be easily restored without a history stack.

    def is_full(self):
        """Returns True if the board is full (Draw)."""
        return len(self.empty_cells) == 0

    def check_win(self, row, col, player):
        """
        Checks if the move at (row, col) triggered a win for the player.
        """
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for dr, dc in directions:
            count = 1
            # Check positive direction
            for i in range(1, self.k):
                r, c = row + dr * i, col + dc * i
                if 0 <= r < self.m and 0 <= c < self.n and self.board[r][c] == player:
                    count += 1
                else:
                    break
            
            # Check negative direction
            for i in range(1, self.k):
                r, c = row - dr * i, col - dc * i
                if 0 <= r < self.m and 0 <= c < self.n and self.board[r][c] == player:
                    count += 1
                else:
                    break
            
            if count >= self.k:
                return True
        return False

    def get_valid_moves(self):
        """Returns a list of all valid (empty) moves."""
        return list(self.empty_cells)

    def get_relevant_moves(self, radius=1):
        """
        Returns a subset of empty cells that are adjacent to occupied cells within a given radius.
        Useful for optimization in AI search.
        """
        if not self.occupied_cells:
            return [(self.m // 2, self.n // 2)]
        
        relevant = set()
        
        for r, c in self.occupied_cells:
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.m and 0 <= nc < self.n:
                        if self.board[nr][nc] == 0:
                            relevant.add((nr, nc))
        
        if not relevant:
            return list(self.empty_cells)
            
        return list(relevant)

    def init_zobrist(self):
        """Initializes values for Zobrist hashing."""
        self.zobrist_table = {}
        # 2 players, m rows, n cols.
        # We need a random number for each cell for each player.
        for r in range(self.m):
            for c in range(self.n):
                for p in [1, 2]:
                    self.zobrist_table[(r, c, p)] = random.getrandbits(64)
        self.current_hash = 0

    def update_hash(self, row, col, player):
        """Updates the board hash incrementally."""
        if hasattr(self, 'zobrist_table'):
            self.current_hash ^= self.zobrist_table[(row, col, player)]


