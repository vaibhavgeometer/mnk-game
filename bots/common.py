import random
import time
from collections import defaultdict

# Centralized Opening Book
OPENING_BOOK = {
    (3, 3, 3): [(1, 1)],
    (15, 15, 5): [(7, 7), (7, 6), (6, 7), (8, 7), (7, 8)],
    (10, 10, 5): [(4, 4), (5, 5), (4, 5), (5, 4)],
    (19, 19, 5): [(9, 9)],
    (6, 7, 4): [(3, 2), (3, 3), (3, 4)], 
}

class TranspositionTable:
    def __init__(self):
        self.table = {}
        # Entry: z_hash -> (depth, score, flag, best_move)
        # Flags: 0=Exact, 1=Lowerbound (Alpha), 2=Upperbound (Beta)
    
    def store(self, z_hash, depth, score, flag, best_move):
        self.table[z_hash] = (depth, score, flag, best_move)
    
    def lookup(self, z_hash, depth, alpha, beta):
        if z_hash in self.table:
            entry = self.table[z_hash]
            e_depth, e_score, e_flag, e_move = entry
            
            if e_depth >= depth:
                if e_flag == 0:  # Exact
                    return e_score, e_move
                if e_flag == 1 and e_score > alpha:  # Lowerbound
                    alpha = e_score
                elif e_flag == 2 and e_score < beta:  # Upperbound
                    beta = e_score
                
                if alpha >= beta:
                    return e_score, e_move
            
            return None, e_move
        return None, None

class BaseBot:
    def __init__(self):
        self.name = "BaseBot"
    
    def get_move(self, board, player, time_limit=None):
        raise NotImplementedError

    def get_opening_move(self, board):
        cnt = len(board.occupied_cells)
        if cnt > 2: return None
        
        if (board.m, board.n, board.k) in OPENING_BOOK:
             options = OPENING_BOOK[(board.m, board.n, board.k)]
             valid_opts = [move for move in options if board.board[move[0]][move[1]] == 0]
             if valid_opts:
                 return random.choice(valid_opts)

        m, n = board.m, board.n
        cx, cy = m // 2, n // 2
        
        if cnt == 0:
            return (cx, cy) 
            
        if cnt == 1:
            if board.board[cx][cy] == 0: return (cx, cy)
            neighbors = [(cx-1, cy-1), (cx+1, cy+1), (cx-1, cy+1), (cx+1, cy-1)]
            valid = [mv for mv in neighbors if 0 <= mv[0] < m and 0 <= mv[1] < n and board.board[mv[0]][mv[1]] == 0]
            if valid: return random.choice(valid)
            
        return None

    def random_move(self, board):
        moves = board.get_valid_moves()
        return random.choice(moves) if moves else None

    def find_winning_move(self, board, player):
        moves = board.get_relevant_moves(1)
        for r, c in moves:
            if board.make_move(r, c, player):
                won = (board.winner == player)
                board.undo_move(r, c)
                if won: return (r, c)
        return None

    def get_random_central_move(self, board):
        moves = board.get_valid_moves()
        if not moves: return None
        cx, cy = board.m/2, board.n/2
        moves.sort(key=lambda m: abs(m[0]-cx) + abs(m[1]-cy))
        return random.choice(moves[:min(len(moves), 10)])

class SearchBot(BaseBot):
    def __init__(self):
        super().__init__()
        self.tt = TranspositionTable()
        self.nodes_visited = 0
        self.start_time = 0
        self.time_limit = 1.0 
        self.killer_moves = defaultdict(lambda: [None, None])
        self.history_heuristic = defaultdict(int)

    def evaluate(self, board, player):
        if board.winner == player: return 100000000
        if board.winner == (3 - player): return -100000000
        
        score = 0
        opponent = 3 - player
        b = board.board
        m, n, k = board.m, board.n, board.k
        
        for r in range(m):
            score += self.evaluate_line(b[r], k, player, opponent)
        for c in range(n):
            col = [b[r][c] for r in range(m)]
            score += self.evaluate_line(col, k, player, opponent)
        for d in range(-(m - 1), n):
            diag = []
            for r in range(m):
                c = r + d
                if 0 <= c < n: diag.append(b[r][c])
            if len(diag) >= k:
                score += self.evaluate_line(diag, k, player, opponent)
        for d in range(0, m + n - 1):
            diag = []
            for r in range(m):
                c = d - r
                if 0 <= c < n: diag.append(b[r][c])
            if len(diag) >= k:
                score += self.evaluate_line(diag, k, player, opponent)
        return score

    def evaluate_line(self, line, k, player, opponent):
        score = 0
        length = len(line)
        for i in range(length - k + 1):
            segment = line[i : i+k]
            p_cnt = segment.count(player)
            o_cnt = segment.count(opponent)
            if p_cnt > 0 and o_cnt > 0: continue 
            if p_cnt == 0 and o_cnt == 0: continue
            opens = 0
            if i > 0 and line[i-1] == 0: opens += 1
            if i + k < length and line[i+k] == 0: opens += 1
            if p_cnt > 0: score += self.get_pattern_score(p_cnt, k, opens, True)
            else: score -= self.get_pattern_score(o_cnt, k, opens, False)
        return score

    def get_pattern_score(self, count, k, opens, is_player):
        s = 0
        if count == k: return 100000000 
        if count == k - 1:
            if opens == 2: s = 5000000 
            elif opens == 1: s = 100000 
        elif count == k - 2:
            if opens == 2: s = 50000 
            elif opens == 1: s = 2000 
            else: s = 100
        elif count == k - 3:
            if opens == 2: s = 1000
            elif opens == 1: s = 100
        else:
            s = count * 10
        if not is_player: s = int(s * 1.5)
        return s

    def update_heuristics(self, depth, move, delta):
        if self.killer_moves[depth][0] != move:
            self.killer_moves[depth][1] = self.killer_moves[depth][0]
            self.killer_moves[depth][0] = move
        self.history_heuristic[move] += delta

    def alphabeta(self, board, depth, player, alpha, beta, maximizing):
        if self.nodes_visited & 127 == 0: 
            if time.time() - self.start_time > self.time_limit:
                raise TimeoutError
        self.nodes_visited += 1
        
        tt_val, tt_move = self.tt.lookup(board.current_hash, depth, alpha, beta)
        if tt_val is not None: return tt_val, tt_move

        if board.winner is not None: return self.evaluate(board, player), None
        if depth == 0 or board.is_full(): return self.evaluate(board, player), None

        radius = 2 if depth > 2 else 1
        moves = board.get_relevant_moves(radius)
        if not moves: moves = board.get_valid_moves()

        def score_move(m):
            if m == tt_move: return 100000000
            if m == self.killer_moves.get(depth, [None, None])[0]: return 9000000
            if m == self.killer_moves.get(depth, [None, None])[1]: return 8000000
            return self.history_heuristic[m]

        moves.sort(key=score_move, reverse=True)
        best_move = None
        original_alpha = alpha
        
        if maximizing:
            value = -float('inf')
            for move in moves:
                board.make_move(move[0], move[1], player)
                try:
                    current_val = self.alphabeta(board, depth - 1, player, alpha, beta, False)[0]
                except TimeoutError:
                    board.undo_move(move[0], move[1])
                    raise
                board.undo_move(move[0], move[1])
                if current_val > value:
                    value = current_val
                    best_move = move
                alpha = max(alpha, value)
                if alpha >= beta:
                    self.update_heuristics(depth, move, delta=depth*depth)
                    break
            flag = 0 
            if value <= original_alpha: flag = 2
            elif value >= beta: flag = 1
            self.tt.store(board.current_hash, depth, value, flag, best_move)
            return value, best_move
        else:
            opponent = 3 - player
            value = float('inf')
            for move in moves:
                board.make_move(move[0], move[1], opponent)
                try:
                    current_val = self.alphabeta(board, depth - 1, player, alpha, beta, True)[0]
                except TimeoutError:
                    board.undo_move(move[0], move[1])
                    raise
                board.undo_move(move[0], move[1])
                if current_val < value:
                    value = current_val
                    best_move = move
                beta = min(beta, value)
                if beta <= alpha:
                    self.update_heuristics(depth, move, delta=depth*depth)
                    break
            flag = 0
            if value <= original_alpha: flag = 2
            elif value >= beta: flag = 1
            self.tt.store(board.current_hash, depth, value, flag, best_move)
            return value, best_move
