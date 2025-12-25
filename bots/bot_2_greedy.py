from .common import BaseBot

class GreedyBot(BaseBot):
    def get_move(self, board, player, time_limit=None):
        move = self.get_opening_move(board)
        if move: return move
        
        win = self.find_winning_move(board, player)
        if win: return win
        
        block = self.find_winning_move(board, 3 - player)
        if block: return block
        
        return self.get_random_central_move(board) or self.random_move(board)
