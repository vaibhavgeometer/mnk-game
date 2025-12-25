from .common import SearchBot
import time

class ModerateBot(SearchBot):
    def get_move(self, board, player, time_limit=None):
        move = self.get_opening_move(board)
        if move: return move
        
        self.time_limit = 999.0 
        self.start_time = time.time()
        self.killer_moves.clear()
        
        score, move = self.alphabeta(board, 3, player, -float('inf'), float('inf'), True)
        return move if move else self.random_move(board)
