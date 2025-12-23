import random
import time
from collections import defaultdict

# Centralized Opening Book
OPENING_BOOK = {
    # (m, n, k): [list of opening moves]
    
    # 3x3 (Tic-Tac-Toe) - Center is best
    (3, 3, 3): [(1, 1)],
    
    # 15x15 Gomoku (Standard)
    # Pro / Long Pro openings (center start)
    (15, 15, 5): [(7, 7), (7, 6), (6, 7), (8, 7), (7, 8)],
    
    # 10x10
    (10, 10, 5): [(4, 4), (5, 5), (4, 5), (5, 4)],
    
    # 19x19
    (19, 19, 5): [(9, 9)],
    
    # 6x7 Connect 4-ish size
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
                if e_flag == 0: # Exact
                    return e_score, e_move
                if e_flag == 1 and e_score > alpha: # Lowerbound
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
            if level == 'easy': self.level = 2
            elif level == 'medium': self.level = 5
            elif level == 'hard': self.level = 8
            else: self.level = 1
        
        self.level = max(1, min(8, self.level))
        
        self.tt = TranspositionTable()
        self.nodes_visited = 0
        self.start_time = 0
        self.time_limit = 1.0 # Default
        self.killer_moves = defaultdict(lambda: [None, None])
        self.history_heuristic = defaultdict(int)

    def get_move(self, board, player, time_limit=None):
        self.nodes_visited = 0
        
        # 1. Check Opening Book
        move = self.get_opening_move(board)
        if move: return move
        
        # 2. Level Dispatch
        # Level 1: Pure Random
        if self.level == 1: 
            return self.random_move(board)
            
        # Level 2: Greedy (Win > Block > Random)
        if self.level == 2:
            return self.level_2_greedy(board, player)
            
        # Level 3: Shallow Search (Depth 2)
        if self.level == 3:
            return self.best_move_minimax(board, player, depth=2)
            
        # Level 4: Moderate Search (Depth 3)
        if self.level == 4:
            return self.best_move_minimax(board, player, depth=3)
            
        # Levels 5-8: Iterative Deepening with Time Management
        # Set time limits based on level if not provided (fallback)
        if time_limit is None:
            if self.level == 5: time_limit = 0.5
            elif self.level == 6: time_limit = 1.0
            elif self.level == 7: time_limit = 2.0
            elif self.level == 8: time_limit = 5.0
            
        # Adjust depth caps
        max_depth_cap = 4 + (self.level - 4) * 2 # 5->6, 6->8, 7->10, 8->12
        if self.level == 8: max_depth_cap = 40 # Try to go deep
        
        return self.best_move_iterative(board, player, time_limit, max_depth_cap)

    def get_opening_move(self, board):
        cnt = len(board.occupied_cells)
        # Only use book for extremely early game
        if cnt > 2: return None
        
        # Exact match
        if (board.m, board.n, board.k) in OPENING_BOOK:
             options = OPENING_BOOK[(board.m, board.n, board.k)]
             valid_opts = [move for move in options if board.board[move[0]][move[1]] == 0]
             if valid_opts:
                 return random.choice(valid_opts)

        # Generic heuristics if book fails but board is empty/near empty
        m, n = board.m, board.n
        cx, cy = m//2, n//2
        
        if cnt == 0:
            return (cx, cy) 
            
        if cnt == 1:
            if board.board[cx][cy] == 0: return (cx, cy)
            # If center taken, play near it diagonally
            neighbors = [(cx-1, cy-1), (cx+1, cy+1), (cx-1, cy+1), (cx+1, cy-1)]
            valid = [mv for mv in neighbors if 0 <= mv[0] < m and 0 <= mv[1] < n and board.board[mv[0]][mv[1]] == 0]
            if valid: return random.choice(valid)
            
        return None

    def random_move(self, board):
        moves = board.get_valid_moves()
        return random.choice(moves) if moves else None

    def level_2_greedy(self, board, player):
        # 1. Check for immediate win
        win = self.find_winning_move(board, player)
        if win: return win
        
        # 2. Check for immediate block
        block = self.find_winning_move(board, 3 - player)
        if block: return block
        
        # 3. Center biased random
        return self.get_random_central_move(board)

    def find_winning_move(self, board, player):
        # Scan all 1-distance neighbors of occupied cells
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
        # Sort by distance to center
        moves.sort(key=lambda m: abs(m[0]-cx) + abs(m[1]-cy))
        # Pick from top 10 to add slight variety
        return random.choice(moves[:min(len(moves), 10)])

    def best_move_minimax(self, board, player, depth):
        # Fixed depth search without strict time limit (but fast enough for low levels)
        self.time_limit = 999.0 
        self.start_time = time.time()
        self.killer_moves.clear()
        
        score, move = self.alphabeta(board, depth, player, -float('inf'), float('inf'), True)
        return move if move else self.random_move(board)

    def best_move_iterative(self, board, player, time_limit, max_depth):
        self.start_time = time.time()
        self.time_limit = time_limit
        self.killer_moves.clear()
        # We accumulate history heuristic across ID depths (or could clear)
        # For a new move, clear it to avoid bias from previous unrelated game states?
        # Actually in a persistent game, clearing is usually safer unless we track game state.
        self.history_heuristic.clear() 
        
        best_move = None
        
        # Get moves sorted by proximity to center initially
        valid_moves = board.get_relevant_moves(1)
        if not valid_moves: valid_moves = board.get_valid_moves()
        
        cx, cy = board.m/2, board.n/2
        valid_moves.sort(key=lambda m: abs(m[0]-cx) + abs(m[1]-cy))
        
        # Quick check for immediate win to skip search if obvious
        win = self.find_winning_move(board, player)
        if win: return win
        
        # Quick check for forced block
        block = self.find_winning_move(board, 3-player)
        if block: 
            # We still search to see if we can force a win instead of just blocking,
            # but we seed best_move with block just in case we timeout immediately.
            best_move = block
        
        try:
            for depth in range(1, max_depth + 1):
                # Search
                score, move = self.alphabeta(board, depth, player, -float('inf'), float('inf'), True)
                
                if move:
                    best_move = move
                
                # If we found a forced win, stop searching
                if score > 90000000:
                    break
                
                # Check time usage
                elapsed = time.time() - self.start_time
                if elapsed > time_limit * 0.5: # If used > 50% of time, unlikely to finish next depth
                    break
                    
        except TimeoutError:
            pass # Return the best move found
            
        return best_move if best_move else self.random_move(board)

    def alphabeta(self, board, depth, player, alpha, beta, maximizing):
        # 1. Time Check
        if self.nodes_visited & 127 == 0: 
            if time.time() - self.start_time > self.time_limit:
                raise TimeoutError
        self.nodes_visited += 1
        
        # 2. TT Lookup
        tt_val, tt_move = self.tt.lookup(board.current_hash, depth, alpha, beta)
        if tt_val is not None:
             return tt_val, tt_move

        # 3. Terminal Node Checks
        if board.winner is not None:
             return self.evaluate(board, player), None
        
        if depth == 0 or board.is_full():
             return self.evaluate(board, player), None

        # 4. Move Generation and Ordering
        radius = 2 if depth > 2 else 1
        moves = board.get_relevant_moves(radius)
        if not moves: moves = board.get_valid_moves()

        def score_move(m):
            if m == tt_move: return 100000000 # TT move first
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
                    # Beta Cutoff
                    self.update_heuristics(depth, move, delta=depth*depth)
                    break
            
            # TT Store
            flag = 0 
            if value <= original_alpha: flag = 2 # Upperbound
            elif value >= beta: flag = 1 # Lowerbound
            self.tt.store(board.current_hash, depth, value, flag, best_move)
            return value, best_move
            
        else: # Minimizing
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
                    # Alpha Cutoff
                    self.update_heuristics(depth, move, delta=depth*depth)
                    break

            flag = 0
            if value <= original_alpha: flag = 2 # Should logic be same?
            elif value >= beta: flag = 1
            # For minimizing:
            # We updated beta.
            # If val <= alpha -> Fail Low (Alpha Cutoff) -> Upperbound? No.
            # If val <= alpha -> It means the maximizing player won't choose this path.
            # Wait, standard TT flags:
            # EXACT(0): alpha < score < beta
            # LOWER(1): score >= beta (Fail High in Max node)
            # UPPER(2): score <= alpha (Fail Low in Max node)
            
            # In Min node:
            # If score <= alpha (Alpha Cutoff): This node is "too good" for Min? No "too bad" for Max.
            # Max won't choose it.
            # The True Value of this node is <= Alpha. So Upperbound.
            
            # If score >= beta: Min hasn't found a move good enough to lower beta below alpha yet.
            
            # Helper logic: always use 'value' relative to alpha/beta passed in.
            if value <= original_alpha: flag = 2 # Upperbound
            elif value >= beta: flag = 1 # Lowerbound
            
            self.tt.store(board.current_hash, depth, value, flag, best_move)
            return value, best_move

    def update_heuristics(self, depth, move, delta):
        # Killer moves
        if self.killer_moves[depth][0] != move:
            self.killer_moves[depth][1] = self.killer_moves[depth][0]
            self.killer_moves[depth][0] = move
        # History heuristic
        self.history_heuristic[move] += delta

    def evaluate(self, board, player):
        # High value for winning
        if board.winner == player: return 100000000
        if board.winner == (3-player): return -100000000
        
        score = 0
        opponent = 3 - player
        
        # Scan lines to evaluate patterns
        b = board.board
        m, n, k = board.m, board.n, board.k
        
        # Gather all lines (Horizontal, Vertical, Diagonals)
        # To optimize, we could do this once or use generator?
        # For now, explicit loops are fastest in pure python compared to generic abstraction overhead.

        # 1. Horizontal
        for r in range(m):
            score += self.evaluate_line(b[r], k, player, opponent)
            
        # 2. Vertical
        for c in range(n):
            col = [b[r][c] for r in range(m)]
            score += self.evaluate_line(col, k, player, opponent)
            
        # 3. Diagonals
        # (Top-left to Bottom-right)
        for d in range(-(m-1), n):
            diag = []
            for r in range(m):
                c = r + d
                if 0 <= c < n: diag.append(b[r][c])
            if len(diag) >= k:
                score += self.evaluate_line(diag, k, player, opponent)
                
        # (Top-right to Bottom-left)
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
        
        # Analyze segments of length k+2 (to see ends) or simply check patterns.
        # "Sliding Window" of length k is standard.
        
        for i in range(length - k + 1):
            segment = line[i : i+k]
            p_cnt = segment.count(player)
            o_cnt = segment.count(opponent)
            
            # If mixed, it's blocked.
            if p_cnt > 0 and o_cnt > 0:
                continue 
            
            if p_cnt == 0 and o_cnt == 0:
                continue

            # Check open ends
            # We need to look outside the window [i, i+k]
            # Left index: i-1, Right index: i+k
            opens = 0
            if i > 0 and line[i-1] == 0: opens += 1
            if i + k < length and line[i+k] == 0: opens += 1
            
            if p_cnt > 0:
                score += self.get_pattern_score(p_cnt, k, opens, True)
            else:
                score -= self.get_pattern_score(o_cnt, k, opens, False)
                
        return score

    def get_pattern_score(self, count, k, opens, is_player):
        # Base Scores for patterns
        s = 0
        
        if count == k: 
            return 100000000 # Win
            
        if count == k - 1:
            if opens == 2: s = 5000000 # Live 4 (Unstoppable usually)
            elif opens == 1: s = 100000 # Dead 4 (Forced Block)
            
        elif count == k - 2:
            if opens == 2: s = 50000 # Live 3 (Creates Live 4)
            elif opens == 1: s = 2000 # Dead 3
            else: s = 100 # Blocked 3 (gap inside) - Actually logic above filters fully blocked.
                          # But this catches "O O O" with ends blocked? No ends are handled by 'opens'.
            
        elif count == k - 3:
            if opens == 2: s = 1000
            elif opens == 1: s = 100
            
        else:
            s = count * 10
            
        # Defense Weighing
        if not is_player:
            # If opponent has a threat, we MUST block it.
            # Defense score > Offense score for same pattern?
            # Maximize (MyScore - OpponentScore)
            # If Opponent has Live 4, score is -5,000,000.
            # If I have Live 4, score is +5,000,000.
            # We want to favor Blocking 4 over making 3.
            # So 1.2x weight on defense ensures we prefer to stop them.
            s = int(s * 1.5)
            
        return s
