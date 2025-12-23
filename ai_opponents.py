import random
import time
from collections import defaultdict

OPENING_BOOK = {
    # (m, n, k): [list of opening moves]
    (3, 3, 3): [(1, 1)],
    (10, 10, 5): [(4, 4), (5, 5), (4, 5), (5, 4), (3, 3), (6, 6)],
    (15, 15, 5): [(7, 7), (6, 6), (8, 8), (6, 8), (8, 6)],
    (19, 19, 5): [(9, 9)],
    (6, 7, 4): [(3, 2), (3, 3), (3, 4)], # Connect 4-ish
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
                if e_flag == 0: # Exact
                    return e_score, e_move
                if e_flag == 1 and e_score > alpha: # Lowerbound (failed high previously, effectively)
                    alpha = e_score
                elif e_flag == 2 and e_score < beta: # Upperbound
                    beta = e_score
                
                if alpha >= beta:
                    return e_score, e_move
            
            return None, e_move # Return move to help ordering
        return None, None

class AI:
    def __init__(self, level):
        try:
            self.level = int(level)
        except (ValueError, TypeError):
            if level == 'easy': self.level = 1
            elif level == 'medium': self.level = 4
            elif level == 'hard': self.level = 7
            else: self.level = 1
        self.level = max(1, min(8, self.level))
        
        self.tt = TranspositionTable()
        self.nodes_visited = 0
        self.start_time = 0
        self.time_limit = 5.0
        self.killer_moves = defaultdict(lambda: [None, None])
        self.history_heuristic = defaultdict(int)

    def get_move(self, board, player, time_limit=5.0):
        self.nodes_visited = 0
        
        # Check Opening Book First
        move = self.get_opening_move(board)
        if move: return move
        
        # Level Dispatch
        if self.level == 1: return self.random_move(board)
        if self.level == 2: return self.level_2(board, player)
        if self.level == 3: return self.level_3(board, player)
        if self.level == 4: return self.best_move_minimax(board, player, 2)
        if self.level == 5: return self.best_move_minimax(board, player, 3)
        if self.level == 6: return self.best_move_minimax(board, player, 4)
        if self.level == 7: return self.best_move_iterative(board, player, time_limit, max_depth=6)
        if self.level == 8: return self.best_move_iterative(board, player, time_limit, max_depth=20)
        
        return self.random_move(board)

    def get_opening_move(self, board):
        cnt = len(board.occupied_cells)
        if cnt > 2: return None
        
        if (board.m, board.n, board.k) in OPENING_BOOK:
             options = OPENING_BOOK[(board.m, board.n, board.k)]
             # Filter occupied
             valid_opts = [move for move in options if board.board[move[0]][move[1]] == 0]
             if valid_opts:
                 return random.choice(valid_opts)

        m, n = board.m, board.n
        
        if cnt == 0:
            return (m//2, n//2) # Always take center if not in book but empty
            
        if cnt == 1:
            cx, cy = m//2, n//2
            if board.board[cx][cy] == 0: return (cx, cy)
            return (cx-1, cy-1)
            
        return None

    def random_move(self, board):
        moves = board.get_valid_moves()
        return random.choice(moves) if moves else None

    def level_2(self, board, player):
        # Win if possible
        win = self.find_winning_move(board, player)
        if win: return win
        
        if random.random() < 0.7:
             return self.random_move(board)
        return self.get_random_central_move(board)

    def level_3(self, board, player):
        win = self.find_winning_move(board, player)
        if win: return win
        block = self.find_winning_move(board, 3 - player)
        if block: return block
        
        if random.random() < 0.6:
            return self.get_random_central_move(board)
        return self.random_move(board)

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
        return random.choice(moves[:min(len(moves), 5)])

    def best_move_minimax(self, board, player, depth):
        # Basic wrapper for fixed depth w/o time limit strictness
        self.time_limit = 100 # loose limit
        self.start_time = time.time()
        # Reset simple stats
        self.killer_moves.clear()
        self.history_heuristic.clear()
        val, move = self.alphabeta(board, depth, player, -float('inf'), float('inf'), True)
        return move if move else self.random_move(board)

    def best_move_iterative(self, board, player, time_limit, max_depth):
        self.start_time = time.time()
        self.time_limit = time_limit
        self.killer_moves.clear()
        self.history_heuristic.clear()
        # Don't clear TT between moves to preserve knowledge? 
        # Actually in this simple implementation, clearing might be safer to avoid collisions across games
        # But for strictly one game, keeping it is good.
        # However, due to memory, let's keep it but user might restart game.
        # Ideally we reset TT on game start. For now, we just use it.
        
        best_move = None
        valid_moves = board.get_relevant_moves(1)
        if not valid_moves: valid_moves = board.get_valid_moves()
        
        # Initial Sort
        cx, cy = board.m/2, board.n/2
        valid_moves.sort(key=lambda m: abs(m[0]-cx) + abs(m[1]-cy))
        
        try:
            for depth in range(1, max_depth + 1):
                score, move = self.alphabeta(board, depth, player, -float('inf'), float('inf'), True)
                
                if move:
                    best_move = move
                    
                # Winning score found (approx 100000000)
                if score > 90000000: 
                    break
                    
                if time.time() - self.start_time > time_limit * 0.6:
                    break
        except TimeoutError:
            pass # Return best move found so far
            
        return best_move if best_move else (valid_moves[0] if valid_moves else None)

    def alphabeta(self, board, depth, player, alpha, beta, maximizing):
        # Time Check
        if self.nodes_visited % 2000 == 0:
            if time.time() - self.start_time > self.time_limit:
                raise TimeoutError
        self.nodes_visited += 1
        
        # TT Lookup
        tt_val, tt_move = self.tt.lookup(board.current_hash, depth, alpha, beta)
        if tt_val is not None:
             return tt_val, tt_move

        if board.winner is not None:
             return self.evaluate(board, player), None
        if depth == 0 or board.is_full():
             return self.evaluate(board, player), None

        # Move Ordering
        moves = board.get_relevant_moves(1)
        if not moves: moves = board.get_valid_moves()

        def score_move(m):
            if m == tt_move: return 10000000
            if m in self.killer_moves[depth]: return 100000
            return self.history_heuristic.get(m, 0)

        moves.sort(key=score_move, reverse=True)
        
        best_move = None
        best_val = -float('inf') if maximizing else float('inf')
        current_flag = 1 if maximizing else 2 # Default to potentially failing low/high
        
        if maximizing:
            for move in moves:
                board.make_move(move[0], move[1], player)
                val = self.alphabeta(board, depth - 1, player, alpha, beta, False)[0]
                board.undo_move(move[0], move[1])
                
                if val > best_val:
                    best_val = val
                    best_move = move
                
                alpha = max(alpha, best_val)
                if beta <= alpha:
                    # Beta Cutoff
                    self.killer_moves[depth][1] = self.killer_moves[depth][0]
                    self.killer_moves[depth][0] = move
                    self.history_heuristic[move] += depth * depth
                    current_flag = 2 # STORE UPPERBOUND? No, if we cut off, we found a value >= beta. 
                    # This means the true value is at least beta.
                    # Standard TT: cutoff on max node -> value is lowerbound of truth?
                    # Wait: alpha-beta returns "at least beta". So it's a Lowerbound on the true value.
                    current_flag = 1 
                    break
            
            if best_val <= alpha and not (beta <= alpha): 
                 # We didn't improve alpha? No, alpha tracks max.
                 # If we searched all and fell plainly within window, it's Exact (0).
                 pass
            
            # Refine flag logic:
            # If val >= beta, Lowerbound (we stopped early, could be higher)
            # If val <= alpha, Upperbound (we couldn't find anything better than alpha, true val is <= alpha)
            # Else Exact.
            
            # Since we modify alpha in loop:
            # Re-check logic
            flag = 0
            if best_val >= beta: flag = 1 # Lowerbound
            elif best_val <= alpha: flag = 2 # Upperbound (original alpha)
            # Note: alpha changes in loop. We should use original alpha for checking fail-low?
            # Standard: if score <= original_alpha -> Upperbound
            # But here we just use best_val.
            
            self.tt.store(board.current_hash, depth, best_val, flag, best_move)
            return best_val, best_move

        else: # Minimizing
            opponent = 3 - player
            for move in moves:
                board.make_move(move[0], move[1], opponent)
                val = self.alphabeta(board, depth - 1, player, alpha, beta, True)[0]
                board.undo_move(move[0], move[1])
                
                if val < best_val:
                    best_val = val
                    best_move = move
                    
                beta = min(beta, best_val)
                if beta <= alpha:
                    # Alpha Cutoff
                    self.killer_moves[depth][1] = self.killer_moves[depth][0]
                    self.killer_moves[depth][0] = move
                    self.history_heuristic[move] += depth * depth
                    current_flag = 2 # Upperbound (Nodes value <= Alpha)
                    break
            
            flag = 0
            if best_val <= alpha: flag = 2 # Upperbound
            elif best_val >= beta: flag = 1 # Lowerbound
            
            self.tt.store(board.current_hash, depth, best_val, flag, best_move)
            return best_val, best_move

    def evaluate(self, board, player):
        if board.winner == player: return 100000000
        if board.winner == (3-player): return -100000000
        
        score = 0
        opponent = 3 - player
        b = board.board
        m, n, k = board.m, board.n, board.k
        
        # Optimize: Horizontal (Fastest with slicing)
        for r in range(m):
            row_data = b[r]
            for c in range(n - k + 1):
                chunk = row_data[c:c+k]
                p_cnt = chunk.count(player)
                o_cnt = chunk.count(opponent)
                
                if p_cnt > 0 and o_cnt > 0: continue
                if p_cnt == 0 and o_cnt == 0: continue
                
                opens = 0
                if c > 0 and row_data[c-1] == 0: opens += 1
                if c + k < n and row_data[c+k] == 0: opens += 1
                
                if p_cnt > 0: score += self.get_line_score(p_cnt, k, opens, True)
                else: score -= self.get_line_score(o_cnt, k, opens, False)

        # For vertical/diagonal, list comp is slow.
        # But iterating strictly is faster than creating list.
        # Vertical
        for c in range(n):
            for r in range(m - k + 1):
                p_cnt = 0
                o_cnt = 0
                # Manual loop
                for i in range(k):
                    val = b[r+i][c]
                    if val == player: p_cnt += 1
                    elif val == opponent: o_cnt += 1
                    if p_cnt > 0 and o_cnt > 0: break 
                
                if p_cnt > 0 and o_cnt > 0: continue
                if p_cnt == 0 and o_cnt == 0: continue
                
                opens = 0
                if r > 0 and b[r-1][c] == 0: opens += 1
                if r + k < m and b[r+k][c] == 0: opens += 1
                
                if p_cnt > 0: score += self.get_line_score(p_cnt, k, opens, True)
                else: score -= self.get_line_score(o_cnt, k, opens, False)

        # Diagonal \
        for r in range(m - k + 1):
            for c in range(n - k + 1):
                p_cnt = 0
                o_cnt = 0
                for i in range(k):
                    val = b[r+i][c+i]
                    if val == player: p_cnt += 1
                    elif val == opponent: o_cnt += 1
                    if p_cnt > 0 and o_cnt > 0: break
                
                if p_cnt > 0 and o_cnt > 0: continue
                if p_cnt == 0 and o_cnt == 0: continue
                
                opens = 0
                if r > 0 and c > 0 and b[r-1][c-1] == 0: opens += 1
                if r + k < m and c + k < n and b[r+k][c+k] == 0: opens += 1
                
                if p_cnt > 0: score += self.get_line_score(p_cnt, k, opens, True)
                else: score -= self.get_line_score(o_cnt, k, opens, False)
        
        # Diagonal /
        for r in range(k - 1, m):
            for c in range(n - k + 1):
                p_cnt = 0
                o_cnt = 0
                for i in range(k):
                    val = b[r-i][c+i]
                    if val == player: p_cnt += 1
                    elif val == opponent: o_cnt += 1
                    if p_cnt > 0 and o_cnt > 0: break # Optimization
                
                if p_cnt > 0 and o_cnt > 0: continue
                if p_cnt == 0 and o_cnt == 0: continue
                
                opens = 0
                if r + 1 < m and c > 0 and b[r+1][c-1] == 0: opens += 1
                if r - k >= 0 and c + k < n and b[r-k][c+k] == 0: opens += 1
                
                if p_cnt > 0: score += self.get_line_score(p_cnt, k, opens, True)
                else: score -= self.get_line_score(o_cnt, k, opens, False)

        return score

    def get_line_score(self, count, k, opens, is_player):
        # Weighted Scoring
        
        if count == k: return 100000000
        
        score = 0
        if count == k-1:
            # k-1 means 1 move to win.
            # If opens == 2 (Live 4), guaranteed win.
            # Otherwise (Dead 4), forced block.
            if opens == 2: score = 2000000 # Higher than any blocked 3
            else: score = 100000 # Dead 4 (opens=1 or 0 implies internal gap)
            
        elif count == k-2:
            # k-2 means 2 moves to win.
            if opens == 2: score = 50000 # Live 3 (2 ways to make Dead 4)
            elif opens == 1: score = 5000 # Dead 3
            else: score = 1000 # Blocked ends but internal gap exists (e.g. wall-P-gap-P-wall)
            
        elif count == k-3:
            if opens == 2: score = 1000
            elif opens == 1: score = 200
            else: score = 50
            
        else:
            score = count * 10
            
        if not is_player:
            # Increase defensive weight slightly to prioritize blocking
            score = int(score * 1.5)
            
        return score
