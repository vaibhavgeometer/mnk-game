
import threading
import copy
import time
from ai_opponents import AI
from game_engine import MNKBoard

class GameSession:
    """
    Manages a single game instance (board, players, timers, AI thread).
    """
    def __init__(self, m, n, k, p1_config, p2_config, time_limit, increment, match_id=None):
        self.match_id = match_id
        self.m = m
        self.n = n
        self.k = k
        self.p1_config = p1_config
        self.p2_config = p2_config
        
        self.board = MNKBoard(m, n, k)
        
        # Init AI
        self.ai_p1 = AI(p1_config['level']) if p1_config['is_ai'] else None
        self.ai_p2 = AI(p2_config['level']) if p2_config['is_ai'] else None
        
        self.p1_name = p1_config.get('name', 'Player 1')
        self.p2_name = p2_config.get('name', 'Player 2')
        
        self.turn = 1
        self.winner = None
        self.win_reason = ""
        
        self.timer_p1 = time_limit
        self.timer_p2 = time_limit
        self.time_limit = time_limit
        self.increment = increment
        
        self.ai_thread = None
        self.ai_move = None
        self.last_update_time = time.time()
        self.is_active = True
        self.move_history = []
        
    def update(self, now):
        """Standard update loop for this session."""
        self.events = [] # Reset events for this frame
        
        if not self.is_active or self.winner is not None:
             return
             
        dt = now - self.last_update_time
        self.last_update_time = now
        
        # Update Timers
        if self.turn == 1:
            self.timer_p1 -= dt
            if self.timer_p1 <= 0:
                self.winner = 2
                self.win_reason = "Time Out"
                self.events.append('game_over')
        else:
            self.timer_p2 -= dt
            if self.timer_p2 <= 0:
                self.winner = 1
                self.win_reason = "Time Out"
                self.events.append('game_over')
                
        if self.winner:
            return

        # AI Logic
        current_ai = self.ai_p1 if self.turn == 1 else self.ai_p2
        
        if current_ai:
            if self.ai_thread is None:
                board_copy = copy.deepcopy(self.board)
                self.ai_thread = threading.Thread(
                    target=self.run_ai_thread, 
                    args=(board_copy, current_ai, self.turn), 
                    daemon=True
                )
                self.ai_thread.start()
            elif not self.ai_thread.is_alive():
                self.ai_thread = None
                if self.ai_move:
                    r, c = self.ai_move
                    self.make_move(r, c)
                self.ai_move = None
    
    def run_ai_thread(self, board_copy, ai_instance, player_num):
        # AI Logic similar to main.py
        remaining = self.timer_p1 if player_num == 1 else self.timer_p2
        start_t = time.time()
        
        limit = max(0.5, min(20.0, remaining * 0.1))
        if limit > remaining - 0.5: limit = max(0.1, remaining - 0.5)
        if remaining < 10: limit = min(limit, 0.5)
        
        self.ai_move = ai_instance.get_move(board_copy, player_num, time_limit=limit)
        
        # Artificial delay for realism if too fast?
        elapsed = time.time() - start_t
        start_delay = 0.5
        if elapsed < start_delay:
             time.sleep(start_delay - elapsed)

    def make_move(self, r, c):
        if self.board.make_move(r, c, self.turn):
            self.events.append('place')
            
            # Record move
            not_str = self.get_notation(r, c)
            self.move_history.append(not_str)
            
            # Increment
            if self.turn == 1: self.timer_p1 += self.increment
            else: self.timer_p2 += self.increment
            
            # Check Win
            if self.board.winner:
                self.winner = self.turn
                self.win_reason = f"Connect {self.k}"
                self.events.append('game_over')
            elif self.board.is_full():
                self.winner = 0 # Draw
                self.win_reason = "Draw"
                self.events.append('game_over')
            else:
                self.turn = 3 - self.turn
            return True
        return False
        
    def get_notation(self, r, c):
        row_str = str(r + 1)
        if c < 26:
            col_str = chr(ord('A') + c)
        else:
            col_str = "A" + chr(ord('A') + (c - 26))
        return col_str + row_str

    def get_status_text(self):
        if self.winner is not None:
            if self.winner == 0: return "Draw"
            return f"Win: P{self.winner}"
        return f"P{self.turn}"
