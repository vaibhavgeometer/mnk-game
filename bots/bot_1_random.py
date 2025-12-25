from .common import BaseBot

class RandomBot(BaseBot):
    def get_move(self, board, player, time_limit=None):
        move = self.get_opening_move(board)
        if move: return move
        return self.random_move(board)
