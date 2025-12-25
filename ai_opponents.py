from bots.bot_1_random import RandomBot
from bots.bot_2_greedy import GreedyBot
from bots.bot_3_shallow import ShallowBot
from bots.bot_4_moderate import ModerateBot
from bots.bot_5_iterative_fast import IterativeFastBot
from bots.bot_6_iterative_normal import IterativeNormalBot
from bots.bot_7_iterative_deep import IterativeDeepBot
from bots.bot_8_iterative_extreme import IterativeExtremeBot

class AI:
    """
    AI Opponent Wrapper that delegates to specific bot implementations.
    """
    def __init__(self, level):
        try:
            self.level = int(level)
        except (ValueError, TypeError):
            # Fallback for string inputs
            if level == 'easy': self.level = 2
            elif level == 'medium': self.level = 5
            elif level == 'hard': self.level = 8
            else: self.level = 1
        
        self.level = max(1, min(8, self.level))
        
        # Instantiate the correct bot
        if self.level == 1: self.bot = RandomBot()
        elif self.level == 2: self.bot = GreedyBot()
        elif self.level == 3: self.bot = ShallowBot()
        elif self.level == 4: self.bot = ModerateBot()
        elif self.level == 5: self.bot = IterativeFastBot()
        elif self.level == 6: self.bot = IterativeNormalBot()
        elif self.level == 7: self.bot = IterativeDeepBot()
        elif self.level == 8: self.bot = IterativeExtremeBot()
        else: self.bot = RandomBot()

    def get_move(self, board, player, time_limit=None):
        """
        Determines the best move for the given board state.
        Delegates to the specific bot instance.
        """
        return self.bot.get_move(board, player, time_limit)
