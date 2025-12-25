
import random
import time
import math

class TournamentPlayer:
    def __init__(self, id, name, is_ai=False, ai_level=0):
        self.id = id
        self.name = name
        self.is_ai = is_ai
        self.ai_level = ai_level
        self.points = 0
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.played = 0

class BaseTournament:
    def __init__(self, player_configs):
        self.players = []
        for i, config in enumerate(player_configs):
            self.players.append(TournamentPlayer(
                id=i, 
                name=config.get('name', f"Player {i+1}"), 
                is_ai=config.get('is_ai', False), 
                ai_level=config.get('level', 0)
            ))
        self.matches = []
        self.completed_matches = []
        self.active_matches = [] # list of match objects/game states if integrated? 
        # Actually, Tournament class just provides schedule. "Active" is game engine concept.
        
    def get_pending_matches(self):
        """Returns list of matches that are ready to be played."""
        raise NotImplementedError

    def record_result(self, p1_idx, p2_idx, winner_idx):
        """Records result of a specific match."""
        raise NotImplementedError

    def is_finished(self):
        raise NotImplementedError
        
    def get_standings(self):
        # Default for most
        return sorted(self.players, key=lambda p: (p.points, p.wins), reverse=True)

class RoundRobinTournament(BaseTournament):
    def __init__(self, player_configs, rounds=1):
        super().__init__(player_configs)
        self.rounds_multiplier = rounds
        self.schedule = [] # List of match tuples (p1_idx, p2_idx)
        self.completed_indices = set()
        self.generate_schedule()
        
    def generate_schedule(self):
        ids = [p.id for p in self.players]
        if len(ids) % 2 != 0: ids.append(-1) # Dummy
        n = len(ids)
        base_matches = []
        
        for round_idx in range(n - 1):
            round_matches = []
            for i in range(n // 2):
                p1 = ids[i]
                p2 = ids[n - 1 - i]
                if p1 != -1 and p2 != -1:
                    if round_idx % 2 == 0: round_matches.append((p1, p2))
                    else: round_matches.append((p2, p1))
            base_matches.extend(round_matches)
            ids = [ids[0]] + [ids[-1]] + ids[1:-1]
            
        for r in range(self.rounds_multiplier):
            chunk = list(base_matches)
            if r % 2 != 0: chunk = [(p2, p1) for (p1, p2) in chunk]
            self.schedule.extend(chunk)
            
    def get_pending_matches(self, limit=1):
        """Returns the next 'limit' matches from schedule that are not completed."""
        pending = []
        for i, (p1, p2) in enumerate(self.schedule):
            if i not in self.completed_indices:
                 pending.append((i, self.players[p1], self.players[p2]))
                 if len(pending) >= limit:
                     break
        return pending

    def record_result(self, match_id, winner_idx):
        if match_id in self.completed_indices:
             return

        self.completed_indices.add(match_id)
        
        match = self.schedule[match_id] # (p1_idx, p2_idx)
        p1 = self.players[match[0]]
        p2 = self.players[match[1]]
        
        p1.played += 1
        p2.played += 1
        
        if winner_idx == -1:
            p1.draws += 1; p2.draws += 1
            p1.points += 0.5; p2.points += 0.5
        elif winner_idx == p1.id:
            p1.wins += 1; p1.points += 1.0
            p2.losses += 1
        elif winner_idx == p2.id:
            p2.wins += 1; p2.points += 1.0
            p1.losses += 1

    def is_finished(self):
        return len(self.completed_indices) >= len(self.schedule)


class KnockoutTournament(BaseTournament):
    def __init__(self, player_configs):
        super().__init__(player_configs)
        # Pad players to power of 2
        self.brackets = [] # List of rounds. Each round is list of matches.
        # Match: {'p1': id, 'p2': id, 'winner': None, 'id': unique_str}
        self.generate_bracket()
        
    def generate_bracket(self):
        n = len(self.players)
        p2 = 1
        while p2 < n: p2 *= 2
        
        # Byes
        byes = p2 - n
        
        # Round 1
        round1 = []
        ids = [p.id for p in self.players]
        random.shuffle(ids) # Random seeding
        
        # First (n - byes) matches? No.
        # Standardalgo: 
        # Top 'byes' players get byes. Bottom (n-byes)*2 play?
        # E.g. 3 players. p2=4. byes=1. 
        # 1 bye, 1 match (2 players). Total 3. Correct.
        
        # IDs map
        current_round_players = ids
        
        # We'll actually structure it dynamically.
        # self.matches_graph = { match_id: {p1, p2, next_match_id} }
        # Simplified: Just keep rounds.
        
        # But wait, we need to return matches that can be played *now*.
        # Knockout is recursive.
        
        self.match_counter = 0
        self.tree = self.build_tree(ids)
        self.active_matches = {} # match_id -> match_node

    class MatchNode:
        def __init__(self, p1=None, p2=None, mid=0):
            self.p1 = p1 # Player ID or None (if waiting)
            self.p2 = p2
            self.mid = mid
            self.winner = None
            self.left = None # Source match for p1
            self.right = None # Source match for p2
            
    def build_tree(self, player_ids):
        # Recursive build
        if len(player_ids) == 1:
            return player_ids[0] # Just an ID
            
        mid = len(player_ids) // 2
        node = self.MatchNode(mid=self.match_counter)
        self.match_counter += 1
        
        # Split
        left_ids = player_ids[:mid]
        right_ids = player_ids[mid:]
        
        node.left = self.build_tree(left_ids)
        node.right = self.build_tree(right_ids)
        
        # Optimization: If leaf is player, bubble up immediately
        if isinstance(node.left, int): node.p1 = node.left
        if isinstance(node.right, int): node.p2 = node.right
        
        return node
        
    def get_pending_matches(self, limit=99):
        # Traverse tree to find ready matches
        ready = []
        
        def traverse(node):
            if isinstance(node, int): return
            
            # If already played, ignore
            if node.winner is not None: return
            
            # Check children first
            if isinstance(node.left, self.MatchNode): traverse(node.left)
            if isinstance(node.right, self.MatchNode): traverse(node.right)
            
            # Check availability
            # If both players present and not played
            if node.p1 is not None and node.p2 is not None and node.winner is None:
                # Check if this match is already being played?
                # We assume caller manages "in progress" state or we just return it.
                # Let's assume stateless here for simplicity: "Ready to Start"
                ready.append(node)

        traverse(self.tree)
        
        result = []
        for r in ready:
             match_info = (r.mid, self.players[r.p1], self.players[r.p2])
             result.append(match_info)
        return result[:limit]

    def record_result(self, match_id, winner_idx):
        # Find node
        node = self.find_node(self.tree, match_id)
        if node:
            node.winner = winner_idx
            
            # Propagate to parent?
            # We need parent pointers or re-traverse to find who points to this node
            self.propagate_winner(self.tree, node, winner_idx)
            
            # Update stats
            if winner_idx != -1: # Draw not really allowed in KO usually?
                 # If draw, simple random advance or rematch?
                 # Assuming strict win.
                 # Let's assume random if draw for now to keep tree moving.
                 if winner_idx == -1: 
                     winner_idx = random.choice([node.p1, node.p2])
                 
                 w_p = self.players[winner_idx]
                 l_p = self.players[node.p1 if winner_idx == node.p2 else node.p2]
                 w_p.wins += 1; w_p.played += 1
                 l_p.losses += 1; l_p.played += 1
            else:
                 # Force decision
                 winner_idx = random.choice([node.p1, node.p2])
                 self.propagate_winner(self.tree, node, winner_idx)

    def find_node(self, node, mid):
        if isinstance(node, int): return None
        if node.mid == mid: return node
        
        l = self.find_node(node.left, mid)
        if l: return l
        return self.find_node(node.right, mid)

    def propagate_winner(self, root, target_child, winner_id):
        if isinstance(root, int): return False
        
        if root.left == target_child:
            root.p1 = winner_id
            return True
        if root.right == target_child:
            root.p2 = winner_id
            return True
            
        if self.propagate_winner(root.left, target_child, winner_id): return True
        return self.propagate_winner(root.right, target_child, winner_id)

    def is_finished(self):
        return self.tree.winner is not None
