import random
import copy

class MNKBoard:
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
        p = self.board[row][col]
        self.board[row][col] = 0
        self.empty_cells.add((row, col))
        self.occupied_cells.remove((row, col))
        if p != 0:
            self.update_hash(row, col, p)
        self.winner = None
        # Note: last_move cannot be easily restored without a history stack in a full game,
        # but for AI search we typically just unset.

    def is_full(self):
        return len(self.empty_cells) == 0

    def check_win(self, row, col, player):
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
        # Return all empty cells
        return list(self.empty_cells)

    def get_relevant_moves(self, radius=1):
        # If board is empty, return center
        if not self.occupied_cells:
            return [(self.m // 2, self.n // 2)]
        
        relevant = set()
        # Use simple heuristic: if occupied are few, iterate occupied
        # if empty are few, iterate empty?
        # For now, iterating occupied is standard for "relevant moves" logic
        
        for r, c in self.occupied_cells:
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    if dr == 0 and dc == 0: continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.m and 0 <= nc < self.n:
                        if self.board[nr][nc] == 0:
                            relevant.add((nr, nc))
        
        if not relevant:
             # Fallback if somehow no relevant moves found but board not full 
            return list(self.empty_cells)
            
        return list(relevant)

    def init_zobrist(self):
        # Initialize Zobrist hashing table
        # 2 players, m rows, n cols
        self.zobrist_table = {}
        for r in range(self.m):
            for c in range(self.n):
                for p in [1, 2]:
                    self.zobrist_table[(r, c, p)] = random.getrandbits(64)
        self.current_hash = 0

    def update_hash(self, row, col, player):
        if hasattr(self, 'zobrist_table'):
            self.current_hash ^= self.zobrist_table[(row, col, player)]


