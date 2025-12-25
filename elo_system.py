import json
import os
import math

ELO_FILE = "elo_ratings.json"
INITIAL_ELO = 2000
MIN_ELO = 0
MAX_ELO = 4000

class EloManager:
    def __init__(self):
        self.ratings = {}
        self.load_ratings()

    def load_ratings(self):
        if os.path.exists(ELO_FILE):
            try:
                with open(ELO_FILE, 'r') as f:
                    self.ratings = json.load(f)
            except:
                self.ratings = {}
        
        # Ensure all levels 1-8 exist with initial ELO if not present
        changed = False
        for i in range(1, 9):
            key = str(i)
            if key not in self.ratings:
                self.ratings[key] = INITIAL_ELO
                changed = True
        
        if changed:
            self.save_ratings()

    def save_ratings(self):
        with open(ELO_FILE, 'w') as f:
            json.dump(self.ratings, f, indent=4)

    def get_rating(self, level):
        return self.ratings.get(str(level), INITIAL_ELO)

    def calculate_k(self, m, n, k_win):
        """
        Calculate K-factor based on game complexity and board size.
        Complexity is roughly proportional to board size (m * n).
        """
        base_k = 32
        
        # Complexity factor: larger boards = higher K (more "weight")
        # Example: 
        # 3x3 (9) -> +0.9 -> K=32.9
        # 10x10 (100) -> +10 -> K=42
        # 20x20 (400) -> +40 -> K=72
        complexity_score = m * n
        
        # Adjust K
        k_factor = base_k + (complexity_score / 10.0)
        
        # Cap K factor to prevent extreme swings on huge boards
        return min(k_factor, 100)

    def update_ratings(self, p1_level, p2_level, winner, m, n, k_win):
        """
        Update ratings for two bots.
        winner: 1 (p1), 2 (p2), 0 (draw)
        """
        r1 = self.get_rating(p1_level)
        r2 = self.get_rating(p2_level)
        
        K = self.calculate_k(m, n, k_win)
        
        # Expected scores
        qa = 10 ** (r1 / 400)
        qb = 10 ** (r2 / 400)
        
        e1 = qa / (qa + qb)
        e2 = qb / (qa + qb)
        
        # Actual scores
        if winner == 1:
            s1 = 1
            s2 = 0
        elif winner == 2:
            s1 = 0
            s2 = 1
        else: # Draw
            s1 = 0.5
            s2 = 0.5
            
        # New ratings
        curr_r1 = r1 + K * (s1 - e1)
        curr_r2 = r2 + K * (s2 - e2)
        
        # Integer constraint
        new_r1 = int(round(curr_r1))
        new_r2 = int(round(curr_r2))
        
        # Floor and Ceiling
        new_r1 = max(MIN_ELO, min(MAX_ELO, new_r1))
        new_r2 = max(MIN_ELO, min(MAX_ELO, new_r2))
        
        self.ratings[str(p1_level)] = new_r1
        self.ratings[str(p2_level)] = new_r2
        
        self.save_ratings()
        
        return new_r1, new_r2
