from .common import SearchBot
import time

class IterativeDeepBot(SearchBot):
    def get_move(self, board, player, time_limit=None):
        move = self.get_opening_move(board)
        if move: return move

        if time_limit is None: time_limit = 2.0
        max_depth = 10
        
        self.start_time = time.time()
        self.time_limit = time_limit
        self.killer_moves.clear()
        self.history_heuristic.clear() 
        
        best_move = None
        win = self.find_winning_move(board, player)
        if win: return win
        
        block = self.find_winning_move(board, 3 - player)
        if block: best_move = block
        
        try:
            for depth in range(1, max_depth + 1):
                score, move = self.alphabeta(board, depth, player, -float('inf'), float('inf'), True)
                if move: best_move = move
                if score > 90000000: break
                
                if time.time() - self.start_time > time_limit * 0.5: break
                    
        except TimeoutError:
            pass
            
        return best_move if best_move else self.random_move(board)
