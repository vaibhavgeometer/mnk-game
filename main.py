import sys
import os
import time
import math
import random
import threading
import copy
import json
from collections import defaultdict
import pygame

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================

SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 800
FPS = 60
DEFAULT_TIME_LIMIT = 180 
ELO_FILE = "elo_ratings.json"
HISTORY_FILE = "game_history.json"
INITIAL_ELO = 2000
MIN_ELO = 0
MAX_ELO = 4000

# ==========================================
# UI COMPONENTS
# ==========================================

class COLORS:
    """Color palette for the UI (Lichess Dark Mode Style)."""
    BACKGROUND = (22, 21, 18)      # #161512
    SURFACE = (38, 36, 33)         # #262421
    SURFACE_LIGHT = (48, 46, 43)   # Lighter surface
    PRIMARY = (60, 179, 113)       # Medium Sea Green (Active elements)
    SECONDARY = (200, 80, 80)      # Soft Red
    ACCENT = (54, 154, 204)        # Lichess Blue
    TEXT = (186, 186, 186)         # #bababa
    TEXT_BRIGHT = (255, 255, 255)
    TEXT_DIM = (100, 100, 100)
    HIGHLIGHT = (60, 60, 60)
    SHADOW = (10, 10, 10)
    TIMER_ACTIVE = (75, 120, 75)   # Green tint for active timer
    TIMER_INACTIVE = (40, 40, 40)   # Dark for inactive

class UIElement:
    """Base class for all UI elements."""
    def __init__(self, x, y, w, h):
        self.rect = pygame.Rect(x, y, w, h)
        self.hovered = False
    
    def update(self, mouse_pos):
        self.hovered = self.rect.collidepoint(mouse_pos)
        return self.hovered

    def draw(self, surface):
        pass

class Button(UIElement):
    """Standard interactable button."""
    def __init__(self, x, y, w, h, text, font, action=None, bg_color=COLORS.SURFACE, hover_color=COLORS.HIGHLIGHT):
        super().__init__(x, y, w, h)
        self.text = text
        self.font = font
        self.action = action
        self.bg_color = bg_color
        self.hover_color = hover_color
        self.click_sound = None

    def draw(self, surface):
        color = self.hover_color if self.hovered else self.bg_color
        # Shadow
        pygame.draw.rect(surface, COLORS.SHADOW, (self.rect.x + 4, self.rect.y + 4, self.rect.w, self.rect.h), border_radius=12)
        # Main body
        pygame.draw.rect(surface, color, self.rect, border_radius=12)
        # Border
        pygame.draw.rect(surface, COLORS.PRIMARY if self.hovered else COLORS.SURFACE, self.rect, width=2, border_radius=12)
        
        text_surf = self.font.render(self.text, True, COLORS.TEXT)
        text_rect = text_surf.get_rect(center=self.rect.center)
        surface.blit(text_surf, text_rect)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and self.hovered:
            if self.click_sound:
                self.click_sound.play()
            if self.action:
                self.action()
            return True
        return False

class Slider(UIElement):
    """Horizontal slider for selecting values."""
    def __init__(self, x, y, w, h, min_val, max_val, initial_val, font, label_prefix="Val"):
        super().__init__(x, y, w, h)
        self.min_val = min_val
        self.max_val = max_val
        self.val = initial_val
        self.font = font
        self.label_prefix = label_prefix
        self.dragging = False

    def get_value(self):
        return int(self.val)

    def update(self, mouse_pos):
        super().update(mouse_pos)
        if self.dragging:
            rel_x = mouse_pos[0] - self.rect.x
            ratio = max(0, min(1, rel_x / self.rect.w))
            self.val = self.min_val + (self.max_val - self.min_val) * ratio
        return self.hovered

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.hovered:
                self.dragging = True
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False

    def draw(self, surface):
        # Label
        if hasattr(self, 'custom_label') and self.custom_label:
            label_text = self.custom_label
        else:
            label_text = f"{self.label_prefix}: {int(self.val)}"
            
        text_surf = self.font.render(label_text, True, COLORS.TEXT)
        surface.blit(text_surf, (self.rect.x, self.rect.y - 30))

        # Track
        pygame.draw.rect(surface, COLORS.SHADOW, (self.rect.x, self.rect.center[1]-4, self.rect.w, 8), border_radius=4)
        pygame.draw.rect(surface, COLORS.TEXT_DIM, (self.rect.x, self.rect.center[1]-4, self.rect.w, 8), width=1, border_radius=4)
        
        # Handle
        ratio = (self.val - self.min_val) / (self.max_val - self.min_val) if (self.max_val > self.min_val) else 0
        handle_x = self.rect.x + (self.rect.w * ratio)
        handle_color = COLORS.PRIMARY if self.dragging or self.hovered else COLORS.TEXT
        pygame.draw.circle(surface, handle_color, (int(handle_x), self.rect.center[1]), 12)

# ==========================================
# GAME ENGINE
# ==========================================

class MNKBoard:
    """Represents the MNK Game board (m-rows, n-cols, k-in-a-row)."""
    def __init__(self, m, n, k):
        self.m = m  # Rows
        self.n = n  # Cols
        self.k = k  # Win condition
        self.board = [[0 for _ in range(n)] for _ in range(m)]
        self.empty_cells = set((r, c) for r in range(m) for c in range(n))
        self.occupied_cells = set()
        self.last_move = None
        self.winner = None
        self.winning_cells = []
        self.init_zobrist()

    def make_move(self, row, col, player):
        if self.board[row][col] != 0:
            return False
            
        self.board[row][col] = player
        self.empty_cells.remove((row, col))
        self.occupied_cells.add((row, col))
        self.update_hash(row, col, player)
        self.last_move = (row, col)
        
        if self.check_win(row, col, player):
            self.winner = player
        
        return True

    def undo_move(self, row, col):
        p = self.board[row][col]
        self.board[row][col] = 0
        self.empty_cells.add((row, col))
        self.occupied_cells.remove((row, col))
        
        if p != 0:
            self.update_hash(row, col, p)
            
        if p != 0:
            self.update_hash(row, col, p)
            
        self.winner = None
        self.winning_cells = []

    def is_full(self):
        return len(self.empty_cells) == 0

    def check_win(self, row, col, player):
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in directions:
            winning_line = [(row, col)]
            # Check positive direction
            for i in range(1, self.k):
                r, c = row + dr * i, col + dc * i
                if 0 <= r < self.m and 0 <= c < self.n and self.board[r][c] == player:
                    winning_line.append((r, c))
                else:
                    break
            # Check negative direction
            for i in range(1, self.k):
                r, c = row - dr * i, col - dc * i
                if 0 <= r < self.m and 0 <= c < self.n and self.board[r][c] == player:
                    winning_line.append((r, c))
                else:
                    break
            if len(winning_line) >= self.k:
                self.winning_cells = winning_line
                return True
        return False

    def get_valid_moves(self):
        return list(self.empty_cells)

    def get_relevant_moves(self, radius=1):
        if not self.occupied_cells:
            return [(self.m // 2, self.n // 2)]
        
        relevant = set()
        for r, c in self.occupied_cells:
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    if dr == 0 and dc == 0: continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.m and 0 <= nc < self.n:
                        if self.board[nr][nc] == 0:
                            relevant.add((nr, nc))
        
        if not relevant:
            return list(self.empty_cells)
        return list(relevant)

    def init_zobrist(self):
        self.zobrist_table = {}
        for r in range(self.m):
            for c in range(self.n):
                for p in [1, 2]:
                    self.zobrist_table[(r, c, p)] = random.getrandbits(64)
        self.current_hash = 0

    def update_hash(self, row, col, player):
        if hasattr(self, 'zobrist_table'):
            self.current_hash ^= self.zobrist_table[(row, col, player)]

# ==========================================
# BOT IMPLEMENTATION
# ==========================================

OPENING_BOOK = {
    (3, 3, 3): [(1, 1)],
    (15, 15, 5): [(7, 7), (7, 6), (6, 7), (8, 7), (7, 8)],
    (10, 10, 5): [(4, 4), (5, 5), (4, 5), (5, 4)],
    (19, 19, 5): [(9, 9)],
    (6, 7, 4): [(3, 2), (3, 3), (3, 4)], 
}

class TranspositionTable:
    def __init__(self):
        self.table = {}
    
    def store(self, z_hash, depth, score, flag, best_move):
        self.table[z_hash] = (depth, score, flag, best_move)
    
    def lookup(self, z_hash, depth, alpha, beta):
        if z_hash in self.table:
            entry = self.table[z_hash]
            e_depth, e_score, e_flag, e_move = entry
            
            if e_depth >= depth:
                if e_flag == 0: return e_score, e_move
                if e_flag == 1 and e_score > alpha: alpha = e_score
                elif e_flag == 2 and e_score < beta: beta = e_score
                
                if alpha >= beta: return e_score, e_move
            return None, e_move
        return None, None

class BaseBot:
    def __init__(self):
        self.name = "BaseBot"
    
    def get_move(self, board, player, time_limit=None):
        raise NotImplementedError

    def get_opening_move(self, board):
        cnt = len(board.occupied_cells)
        if cnt > 2: return None
        
        if (board.m, board.n, board.k) in OPENING_BOOK:
             options = OPENING_BOOK[(board.m, board.n, board.k)]
             valid_opts = [move for move in options if board.board[move[0]][move[1]] == 0]
             if valid_opts: return random.choice(valid_opts)

        m, n = board.m, board.n
        cx, cy = m // 2, n // 2
        
        if cnt == 0: return (cx, cy) 
        if cnt == 1:
            if board.board[cx][cy] == 0: return (cx, cy)
            neighbors = [(cx-1, cy-1), (cx+1, cy+1), (cx-1, cy+1), (cx+1, cy-1)]
            valid = [mv for mv in neighbors if 0 <= mv[0] < m and 0 <= mv[1] < n and board.board[mv[0]][mv[1]] == 0]
            if valid: return random.choice(valid)
        return None

    def random_move(self, board):
        moves = board.get_valid_moves()
        return random.choice(moves) if moves else None

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
        return random.choice(moves[:min(len(moves), 10)])

class SearchBot(BaseBot):
    def __init__(self):
        super().__init__()
        self.tt = TranspositionTable()
        self.nodes_visited = 0
        self.start_time = 0
        self.time_limit = 1.0 
        self.killer_moves = defaultdict(lambda: [None, None])
        self.history_heuristic = defaultdict(int)

    def evaluate(self, board, player):
        if board.winner == player: return 100000000
        if board.winner == (3 - player): return -100000000
        
        score = 0
        opponent = 3 - player
        b = board.board
        m, n, k = board.m, board.n, board.k
        
        for r in range(m): score += self.evaluate_line(b[r], k, player, opponent)
        for c in range(n):
            col = [b[r][c] for r in range(m)]
            score += self.evaluate_line(col, k, player, opponent)
        for d in range(-(m - 1), n):
            diag = []
            for r in range(m):
                c = r + d
                if 0 <= c < n: diag.append(b[r][c])
            if len(diag) >= k: score += self.evaluate_line(diag, k, player, opponent)
        for d in range(0, m + n - 1):
            diag = []
            for r in range(m):
                c = d - r
                if 0 <= c < n: diag.append(b[r][c])
            if len(diag) >= k: score += self.evaluate_line(diag, k, player, opponent)
        return score

    def evaluate_line(self, line, k, player, opponent):
        score = 0
        length = len(line)
        for i in range(length - k + 1):
            segment = line[i : i+k]
            p_cnt = segment.count(player)
            o_cnt = segment.count(opponent)
            if p_cnt > 0 and o_cnt > 0: continue 
            if p_cnt == 0 and o_cnt == 0: continue
            opens = 0
            if i > 0 and line[i-1] == 0: opens += 1
            if i + k < length and line[i+k] == 0: opens += 1
            if p_cnt > 0: score += self.get_pattern_score(p_cnt, k, opens, True)
            else: score -= self.get_pattern_score(o_cnt, k, opens, False)
        return score

    def get_pattern_score(self, count, k, opens, is_player):
        s = 0
        if count == k: return 100000000 
        if count == k - 1:
            if opens == 2: s = 5000000 
            elif opens == 1: s = 100000 
        elif count == k - 2:
            if opens == 2: s = 50000 
            elif opens == 1: s = 2000 
            else: s = 100
        elif count == k - 3:
            if opens == 2: s = 1000
            elif opens == 1: s = 100
        else:
            s = count * 10
        if not is_player: s = int(s * 1.5)
        return s

    def update_heuristics(self, depth, move, delta):
        if self.killer_moves[depth][0] != move:
            self.killer_moves[depth][1] = self.killer_moves[depth][0]
            self.killer_moves[depth][0] = move
        self.history_heuristic[move] += delta

    def alphabeta(self, board, depth, player, alpha, beta, maximizing):
        if self.nodes_visited & 127 == 0: 
            if time.time() - self.start_time > self.time_limit:
                raise TimeoutError
        self.nodes_visited += 1
        
        tt_val, tt_move = self.tt.lookup(board.current_hash, depth, alpha, beta)
        if tt_val is not None: return tt_val, tt_move

        if board.winner is not None: return self.evaluate(board, player), None
        if depth == 0 or board.is_full(): return self.evaluate(board, player), None

        radius = 2 if depth > 2 else 1
        moves = board.get_relevant_moves(radius)
        if not moves: moves = board.get_valid_moves()

        def score_move(m):
            if m == tt_move: return 100000000
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
                    self.update_heuristics(depth, move, delta=depth*depth)
                    break
            flag = 0 
            if value <= original_alpha: flag = 2
            elif value >= beta: flag = 1
            self.tt.store(board.current_hash, depth, value, flag, best_move)
            return value, best_move
        else:
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
                    self.update_heuristics(depth, move, delta=depth*depth)
                    break
            flag = 0
            if value <= original_alpha: flag = 2
            elif value >= beta: flag = 1
            self.tt.store(board.current_hash, depth, value, flag, best_move)
            return value, best_move

# SPECIFIC BOTS
class IterativeBot(SearchBot):
    def __init__(self, max_depth, default_time):
        super().__init__()
        self.max_depth = max_depth
        self.default_time = default_time

    def get_move(self, board, player, time_limit=None):
        move = self.get_opening_move(board)
        if move: return move

        if time_limit is None: time_limit = self.default_time
        
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
            for depth in range(1, self.max_depth + 1):
                score, move = self.alphabeta(board, depth, player, -float('inf'), float('inf'), True)
                if move: best_move = move
                if score > 90000000: break
                if time.time() - self.start_time > time_limit * 0.5: break
        except TimeoutError:
            pass
        return best_move if best_move else self.random_move(board)

class AI:
    """AI Opponent Wrapper that delegates to specific bot implementations."""
    def __init__(self, level):
        try:
            self.level = int(level)
        except (ValueError, TypeError):
            self.level = 1
        self.level = max(1, min(4, self.level))
        
        if self.level == 1: 
            # Level 1: Elo ~600 (Greedy / Depth 1)
            self.bot = IterativeBot(1, 0.5)
        elif self.level == 2: 
            # Level 2: Elo ~1200 (Depth 3 - Avoids blunders, sees traps)
            self.bot = IterativeBot(3, 1.0)
        elif self.level == 3: 
            # Level 3: Elo ~1800 (Depth 6 - Strong play)
            self.bot = IterativeBot(6, 3.0)
        elif self.level == 4: 
            # Level 4: Max Strength (Deep search)
            self.bot = IterativeBot(40, 10.0)
        else:
            self.bot = IterativeBot(1, 0.5)

    def get_move(self, board, player, time_limit=None):
        return self.bot.get_move(board, player, time_limit)

# ==========================================
# SYSTEM MANAGERS (ELO, RECORDS, SOUND)
# ==========================================

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
        
        defaults = {'1': 600, '2': 1200, '3': 1800, '4': 2400}
        for i in range(1, 5):
            key = str(i)
            if key not in self.ratings:
                self.ratings[key] = defaults.get(key, INITIAL_ELO)

    def save_ratings(self):
        try:
            with open(ELO_FILE, 'w') as f:
                json.dump(self.ratings, f, indent=4)
        except: pass

    def get_rating(self, level):
        return self.ratings.get(str(level), INITIAL_ELO)

    def calculate_k(self, m, n, k_win):
        base_k = 32
        complexity_score = m * n
        k_factor = base_k + (complexity_score / 10.0)
        return min(k_factor, 100)

    def update_ratings(self, p1_level, p2_level, winner, m, n, k_win):
        r1 = self.get_rating(p1_level)
        r2 = self.get_rating(p2_level)
        
        K = self.calculate_k(m, n, k_win)
        
        qa = 10 ** (r1 / 400)
        qb = 10 ** (r2 / 400)
        
        e1 = qa / (qa + qb)
        e2 = qb / (qa + qb)
        
        if winner == 1: s1, s2 = 1, 0
        elif winner == 2: s1, s2 = 0, 1
        else: s1, s2 = 0.5, 0.5
            
        curr_r1 = r1 + K * (s1 - e1)
        curr_r2 = r2 + K * (s2 - e2)
        
        new_r1 = max(MIN_ELO, min(MAX_ELO, int(round(curr_r1))))
        new_r2 = max(MIN_ELO, min(MAX_ELO, int(round(curr_r2))))
        
        self.ratings[str(p1_level)] = new_r1
        self.ratings[str(p2_level)] = new_r2
        self.save_ratings()

class RecordManager:
    """Handles saving and loading of game records."""
    def __init__(self):
        self.filename = HISTORY_FILE
        self.history = self.load_history()

    def load_history(self):
        if not os.path.exists(self.filename):
            return {"games": []}
        try:
            with open(self.filename, "r") as f:
                return json.load(f)
        except:
            return {"games": []}

    def save_history(self):
        try:
            with open(self.filename, "w") as f:
                json.dump(self.history, f, indent=4)
        except IOError: pass

    def add_game_record(self, record):
        record['timestamp'] = record.get('timestamp', time.time())
        self.history["games"].append(record)
        self.save_history()

class SoundManager:
    """Handles loading and playing of sound effects and music."""
    def __init__(self):
        self.sounds = {}
        self.music_playing = False
        self.use_generated = False
        try:
            import numpy as np
            self.np = np
            self.use_generated = True
        except ImportError:
            self.use_generated = False
        
        try:
            import array
            import struct
            self.array = array
            self.struct = struct
            self.use_fallback = True
        except:
             self.use_fallback = False

        try:
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
            self.generate_defaults()
        except pygame.error as e:
            print(f"Sound system error: {e}")

    def generate_defaults(self):
        if self.use_generated:
            self.generate_numpy_sounds()
        elif self.use_fallback:
            self.generate_fallback_sounds()

    def generate_numpy_sounds(self):
        def make_tone(freq, dur, vol=0.1):
            sample_rate = 44100
            n_samples = int(sample_rate * dur)
            t = self.np.linspace(0, dur, n_samples, False)
            wave = self.np.sin(2 * self.np.pi * freq * t) * 32767 * vol
            wave = wave.astype(self.np.int16)
            return pygame.sndarray.make_sound(self.np.column_stack((wave, wave)))

        self.sounds['click'] = make_tone(600, 0.05)
        self.sounds['place'] = make_tone(400, 0.1)
        self.sounds['win'] = make_tone(800, 0.3)
        self.sounds['lose'] = make_tone(200, 0.4)
        
    def generate_fallback_sounds(self):
        def make_tone(freq, dur, vol=0.1):
            sample_rate = 44100
            n_samples = int(sample_rate * dur)
            # Square wave generation
            period = sample_rate // freq
            width = period // 2
            
            data = self.array.array('h')
            for i in range(n_samples):
                val = 32767 * vol if (i % period) < width else -32767 * vol
                data.append(int(val))
                data.append(int(val)) # Stereo
                
            return pygame.mixer.Sound(buffer=data)

        try:
            self.sounds['click'] = make_tone(600, 0.05)
            self.sounds['place'] = make_tone(400, 0.1)
            self.sounds['win'] = make_tone(800, 0.3)
            self.sounds['lose'] = make_tone(200, 0.4)
        except Exception as e:
            print(f"Fallback sound generation failed: {e}")

    def play(self, name):
        if name in self.sounds:
            self.sounds[name].play()
    
    def set_music_volume(self, vol):
        try: pygame.mixer.music.set_volume(vol)
        except: pass

    def load_music(self, path):
        if os.path.exists(path):
            try:
                pygame.mixer.music.load(path)
                pygame.mixer.music.play(-1)
                pygame.mixer.music.set_volume(0.5)
                self.music_playing = True
            except: pass

# ==========================================
# GAME SESSION & APP
# ==========================================

class GameSession:
    """Manages a single game instance (board, players, timers, AI thread)."""
    def __init__(self, m, n, k, p1_config, p2_config, time_limit, increment):
        self.m = m
        self.n = n
        self.k = k
        self.p1_config = p1_config
        self.p2_config = p2_config
        
        self.board = MNKBoard(m, n, k)
        
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
        self.events = []
        
    def update(self, now):
        if not hasattr(self, 'events'): self.events = []
        if not self.is_active or self.winner is not None: return
             
        dt = now - self.last_update_time
        self.last_update_time = now
        
        # Update Timers
        if self.turn == 1:
            self.timer_p1 -= dt
            if self.timer_p1 <= 0:
                self.winner = 2; self.win_reason = "Time Out"; self.events.append('game_over')
        else:
            self.timer_p2 -= dt
            if self.timer_p2 <= 0:
                self.winner = 1; self.win_reason = "Time Out"; self.events.append('game_over')
                
        if self.winner: return

        # AI Logic
        current_ai = self.ai_p1 if self.turn == 1 else self.ai_p2
        if current_ai:
            if self.ai_thread is None:
                board_copy = copy.deepcopy(self.board)
                self.ai_thread = threading.Thread(target=self.run_ai_thread, args=(board_copy, current_ai, self.turn), daemon=True)
                self.ai_thread.start()
            elif not self.ai_thread.is_alive():
                self.ai_thread = None
                if self.ai_move:
                    self.make_move(*self.ai_move)
                self.ai_move = None
    
    def run_ai_thread(self, board_copy, ai_instance, player_num):
        remaining = self.timer_p1 if player_num == 1 else self.timer_p2
        start_t = time.time()
        
        limit = max(0.5, min(20.0, remaining * 0.1))
        if limit > remaining - 0.5: limit = max(0.1, remaining - 0.5)
        if remaining < 10: limit = min(limit, 0.5)
        
        self.ai_move = ai_instance.get_move(board_copy, player_num, time_limit=limit)
        
        elapsed = time.time() - start_t
        start_delay = 0.5
        if elapsed < start_delay: time.sleep(start_delay - elapsed)

    def make_move(self, r, c):
        if self.board.make_move(r, c, self.turn):
            self.events.append('place')
            self.move_history.append(self.get_notation(r, c))
            
            if self.turn == 1: self.timer_p1 += self.increment
            else: self.timer_p2 += self.increment
            
            if self.board.winner:
                self.winner = self.turn; self.win_reason = f"Connect {self.k}"; self.events.append('game_over')
            elif self.board.is_full():
                self.winner = 0; self.win_reason = "Draw"; self.events.append('game_over')
            else:
                self.turn = 3 - self.turn
            return True
        return False
        
    def get_notation(self, r, c):
        row_str = str(r + 1)
        col_str = chr(ord('A') + c) if c < 26 else "A" + chr(ord('A') + (c - 26))
        return col_str + row_str

class GameApp:
    def __init__(self):
        pygame.init()
        self.width = SCREEN_WIDTH
        self.height = SCREEN_HEIGHT
        self.windowed_size = (self.width, self.height)
        self.fullscreen = False
        self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
        pygame.display.set_caption("MNK Game Deluxe - AI Edition")
        self.clock = pygame.time.Clock()
        
        self.font_title = pygame.font.SysFont("Segoe UI", 60, bold=True)
        self.font_ui = pygame.font.SysFont("Segoe UI", 24)
        self.font_timer = pygame.font.SysFont("Consolas", 32, bold=True)
        self.font_small = pygame.font.SysFont("Segoe UI", 16)
        
        self.sound_manager = SoundManager()

        self.record_manager = RecordManager()
        self.elo_manager = EloManager()

        self.state = "MENU"
        
        self.m = 15
        self.n = 15
        self.k = 5
        self.p1_level = 0
        self.p2_level = 2
        self.time_limit = DEFAULT_TIME_LIMIT 
        self.time_increment = 2
        self.sessions = []
        
        self.init_ui()

    def init_ui(self):
        cx, cy = self.width // 2, self.height // 2
        s_m = self.slider_m.get_value() if hasattr(self, 'slider_m') else self.m
        s_n = self.slider_n.get_value() if hasattr(self, 'slider_n') else self.n
        s_k = self.slider_k.get_value() if hasattr(self, 'slider_k') else self.k
        s_p1 = min(4, self.slider_p1.get_value()) if hasattr(self, 'slider_p1') else min(4, self.p1_level)
        s_p2 = min(4, self.slider_p2.get_value()) if hasattr(self, 'slider_p2') else min(4, self.p2_level)
        s_time = self.slider_time.get_value() if hasattr(self, 'slider_time') else 3
        s_inc = self.slider_increment.get_value() if hasattr(self, 'slider_increment') else 2

        
        # Menu Buttons (Centered and evenly spaced)
        btn_start_y = cy - 110
        btn_gap = 70
        self.btn_play = Button(cx - 100, btn_start_y, 200, 50, "PLAY", self.font_ui, lambda: self.start_single_game())
        self.btn_history = Button(cx - 100, btn_start_y + btn_gap, 200, 50, "HISTORY", self.font_ui, lambda: self.set_state("HISTORY"))
        self.btn_settings = Button(cx - 100, btn_start_y + btn_gap*2, 200, 50, "SETTINGS", self.font_ui, lambda: self.set_state("SETTINGS"))
        self.btn_quit = Button(cx - 100, btn_start_y + btn_gap*3, 200, 50, "QUIT", self.font_ui, lambda: self.quit_game())
        self.btn_back = Button(cx - 100, self.height - 100, 200, 50, "BACK", self.font_ui, lambda: self.set_state("MENU"))
        
        sy = cy - 250
        self.slider_m = Slider(cx - 150, sy, 300, 20, 1, 32, s_m, self.font_ui, "Rows (M)")
        self.slider_n = Slider(cx - 150, sy + 70, 300, 20, 1, 32, s_n, self.font_ui, "Cols (N)")
        self.slider_k = Slider(cx - 150, sy + 140, 300, 20, 1, 7, s_k, self.font_ui, "Win (K)")
        self.slider_p1 = Slider(cx - 150, sy + 210, 300, 20, 0, 4, s_p1, self.font_ui, "Player 1")
        self.slider_p2 = Slider(cx - 150, sy + 280, 300, 20, 0, 4, s_p2, self.font_ui, "Player 2")
        self.slider_time = Slider(cx - 150, sy + 350, 300, 20, 1, 30, s_time, self.font_ui, "Time (Mins)")
        self.slider_increment = Slider(cx - 150, sy + 420, 300, 20, 0, 60, s_inc, self.font_ui, "Increment (Sec)")

        
        self.btn_rematch = Button(cx - 100, cy + 80, 200, 50, "PLAY AGAIN", self.font_ui, lambda: self.start_single_game())
        self.btn_menu = Button(cx - 100, cy + 150, 200, 50, "MENU", self.font_ui, lambda: self.set_state("MENU"))
        
    def set_state(self, state):
        self.state = state
        
    def quit_game(self):
        pygame.quit()
        sys.exit()

    def start_single_game(self):
        self.m = self.slider_m.get_value()
        self.n = self.slider_n.get_value()
        self.k = self.slider_k.get_value()
        self.time_limit = self.slider_time.get_value() * 60
        self.time_increment = self.slider_increment.get_value()
        if self.k > max(self.m, self.n): self.k = max(self.m, self.n)
        
        p1_config = {'name': 'Player 1', 'is_ai': self.slider_p1.get_value() > 0, 'level': self.slider_p1.get_value()}
        p2_config = {'name': 'Player 2', 'is_ai': self.slider_p2.get_value() > 0, 'level': self.slider_p2.get_value()}
        
        session = GameSession(self.m, self.n, self.k, p1_config, p2_config, self.time_limit, self.time_increment)
        self.sessions = [session]
        self.set_state("GAME")
        
    def run(self):
        while True:
            self.handle_input()
            self.update()
            self.render()
            self.clock.tick(FPS)
            
    def handle_input(self):
        mouse_pos = pygame.mouse.get_pos()
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT: self.quit_game()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_F11:
                    self.fullscreen = not self.fullscreen
                    if self.fullscreen:
                        self.windowed_size = (self.width, self.height)
                        self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
                    else:
                        w, h = self.windowed_size
                        self.screen = pygame.display.set_mode((w, h), pygame.RESIZABLE)
                    self.width, self.height = self.screen.get_size()
                    self.init_ui()

            if event.type == pygame.VIDEORESIZE:
                if not self.fullscreen:
                    self.width, self.height = event.w, event.h
                    self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
                    self.init_ui()
            
            if self.state == "MENU":
                for b in [self.btn_play, self.btn_history, self.btn_settings, self.btn_quit]:
                     b.handle_event(event)

            elif self.state == "SETTINGS":
                 for s in [self.slider_m, self.slider_n, self.slider_k, self.slider_p1, self.slider_p2, self.slider_time, self.slider_increment]:
                     s.handle_event(event)
                 self.btn_back.handle_event(event)
            
            elif self.state == "HISTORY":
                 self.btn_back.handle_event(event)

            elif self.state in ["GAME", "GAMEOVER"]:
                handled = False
                if self.sessions and self.sessions[0].winner is not None:
                     for b in [self.btn_rematch, self.btn_menu]: 
                         if b.handle_event(event): handled = True
                if handled: continue

                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                     self.handle_game_click(mouse_pos)

    def handle_game_click(self, pos):
        layout = self.get_layout(len(self.sessions))
        for i, session in enumerate(self.sessions):
            rect, cell_size = layout[i]
            if session.winner is None:
                 is_p1_human = (session.turn == 1 and not session.ai_p1)
                 is_p2_human = (session.turn == 2 and not session.ai_p2)
                 if is_p1_human or is_p2_human:
                      bx, by, bw, bh = rect
                      if bx <= pos[0] < bx + bw and by <= pos[1] < by + bh:
                          c = int((pos[0] - bx) // cell_size)
                          r = int((pos[1] - by) // cell_size)
                          session.make_move(r, c)
        
    def update(self):
        now = time.time()
        if self.state == "SETTINGS":
            self.update_settings_sliders(pygame.mouse.get_pos())
            self.btn_back.update(pygame.mouse.get_pos())
        elif self.state == "MENU":
            for b in [self.btn_play, self.btn_history, self.btn_settings, self.btn_quit]: 
                b.update(pygame.mouse.get_pos())
        elif self.state == "HISTORY":
             self.btn_back.update(pygame.mouse.get_pos())
        elif self.state == "GAME" or self.state == "GAMEOVER":
             all_finished = True
             for session in self.sessions:
                 if session.is_active:
                     session.update(now)
                     for e in getattr(session, 'events', []):
                         if e == 'place': self.sound_manager.play('place')
                         elif e == 'game_over': 
                             self.sound_manager.play('win' if session.winner else 'lose')
                             self.save_game_result(session)
                             session.is_active = False 
                     session.events = []
                     if session.winner is None: all_finished = False

             if all_finished: self.state = "GAMEOVER"

             if self.state == "GAMEOVER":
                 for b in [self.btn_rematch, self.btn_menu]: b.update(pygame.mouse.get_pos())

    def save_game_result(self, session):
        rec = {
            'timestamp': time.time(),
            'p1_name': session.p1_name,
            'p1_level': session.p1_config.get('level', 0),
            'p2_name': session.p2_name,
            'p2_level': session.p2_config.get('level', 0),
            'm': session.m, 'n': session.n, 'k': session.k,
            'winner': session.winner,
            'win_reason': session.win_reason,
            'moves': session.move_history,
            'mode': 'Quick'
        }
        self.record_manager.add_game_record(rec)

        p1_ai = session.p1_config.get('is_ai', False)
        p2_ai = session.p2_config.get('is_ai', False)
        if p1_ai and p2_ai:
            p1_lvl = session.p1_config.get('level')
            p2_lvl = session.p2_config.get('level')
            if p1_lvl and p2_lvl:
                self.elo_manager.update_ratings(p1_lvl, p2_lvl, session.winner, session.m, session.n, session.k)

    def get_layout(self, num_sessions):
        # We enforce a single game view for this UI style
        if num_sessions == 0: return []
        
        # 3-Column Layout: [Left Info 20%] [Board 55%] [Right Timer/Moves 25%]
        total_w, total_h = self.width, self.height
        
        # Margins
        margin = 10
        
        # Panel Dimensions
        left_w = 250
        right_w = 300
        center_w = total_w - left_w - right_w - (4 * margin)
        
        # Center Board Calculation to maintain aspect ratio
        max_h = total_h - 2 * margin
        
        # We need to return a list of (rect, cell_size) tuples for the existing render logic to use
        # But for the new UI, we only really support one main session well.
        # We will calculate the board rect here.
        
        session = self.sessions[0]
        cs = min(center_w // session.n, max_h // session.m)
        if cs < 10: cs = 10
        
        bw, bh = cs * session.n, cs * session.m
        bx = left_w + 2 * margin + (center_w - bw) // 2
        by = margin + (max_h - bh) // 2
        
        # Main legacy return format for the board rendering loop
        return [((bx, by, bw, bh), cs)]

    def render_lichess_ui(self, session, layout_info):
        board_rect_tuple, cs = layout_info
        bx, by, bw, bh = board_rect_tuple
        
        # --- LEFT PANEL (Game Info) ---
        left_rect = pygame.Rect(10, 10, 240, self.height - 20)
        pygame.draw.rect(self.screen, COLORS.SURFACE, left_rect, border_radius=4)
        
        # Header
        pygame.draw.rect(self.screen, COLORS.SURFACE_LIGHT, (left_rect.x, left_rect.y, left_rect.w, 50), border_radius=4)
        title = self.font_ui.render(f"MNK ({session.k})", True, COLORS.TEXT_BRIGHT)
        self.screen.blit(title, (left_rect.x + 15, left_rect.y + 10))
        
        # Game Details
        y_off = 70
        info_lines = [
            f"Rated • Rapid", # Placeholder
            f"{int(session.time_limit//60)}+{session.increment}",
            f"Grid: {session.m}x{session.n}"
        ]
        for line in info_lines:
            txt = self.font_small.render(line, True, COLORS.TEXT)
            self.screen.blit(txt, (left_rect.x + 15, left_rect.y + y_off))
            y_off += 25
            
        # Chat Placeholder (Visual only)
        chat_y = self.height - 250
        pygame.draw.rect(self.screen, COLORS.BACKGROUND, (left_rect.x + 10, chat_y, left_rect.w - 20, 230), border_radius=4)
        chat_lbl = self.font_small.render("Spectator Room", True, COLORS.TEXT_DIM)
        self.screen.blit(chat_lbl, (left_rect.x + 20, chat_y + 10))

        # --- RIGHT PANEL (Timers & Moves) ---
        right_x = self.width - 310
        right_rect = pygame.Rect(right_x, 10, 300, self.height - 20)
        
        # Top Player (Player 2)
        self.render_player_card(right_rect.x, right_rect.y, right_rect.w, session, 2)
        
        # Move List (Middle)
        mid_y = right_rect.y + 100
        mid_h = self.height - 220
        self.render_move_list(right_rect.x, mid_y, right_rect.w, mid_h, session)
        
        # Bottom Player (Player 1)
        self.render_player_card(right_rect.x, self.height - 100, right_rect.w, session, 1)

    def render_player_card(self, x, y, w, session, player_num):
        is_active = (session.turn == player_num) and (session.winner is None)
        timer_val = session.timer_p1 if player_num == 1 else session.timer_p2
        name = session.p1_name if player_num == 1 else session.p2_name
        rating = self.elo_manager.get_rating(session.p1_config.get('level', 0) if player_num == 1 else session.p2_config.get('level', 0))
        level_label = f" ({rating})"
        
        # Background
        # pygame.draw.rect(self.screen, COLORS.SURFACE, (x, y, w, 90), border_radius=4)
        
        # Timer Box
        t_bg = COLORS.TIMER_ACTIVE if is_active else COLORS.TIMER_INACTIVE
        t_rect = pygame.Rect(x + w - 100, y + 25, 90, 40)
        pygame.draw.rect(self.screen, t_bg, t_rect, border_radius=4)
        
        # Time Text
        t_str = self.format_time(timer_val)
        t_surf = self.font_timer.render(t_str, True, COLORS.TEXT_BRIGHT) # if is_active else COLORS.TEXT)
        t_dest = t_surf.get_rect(center=t_rect.center)
        self.screen.blit(t_surf, t_dest)
        
        # Name & Info
        name_surf = self.font_ui.render(name, True, COLORS.TEXT_BRIGHT)
        self.screen.blit(name_surf, (x + 10, y + 20))
        
        rating_surf = self.font_small.render(level_label, True, COLORS.TEXT_DIM)
        self.screen.blit(rating_surf, (x + 10, y + 50))
        
        # Indicator Dot
        dot_color = COLORS.PRIMARY if is_active else COLORS.SURFACE_LIGHT
        pygame.draw.circle(self.screen, dot_color, (x + w - 110, y + 45), 5)

    def render_move_list(self, x, y, w, h, session):
        pygame.draw.rect(self.screen, COLORS.SURFACE, (x, y, w, h), border_radius=4)
        
        # Header
        pygame.draw.rect(self.screen, COLORS.SURFACE_LIGHT, (x, y, w, 30), border_radius=4)
        # headers = ["#", "White", "Black"]
        # for i, txt in enumerate(headers):
        #    s = self.font_small.render(txt, True, COLORS.TEXT_DIM)
        #     self.screen.blit(s, (x + 10 + i*80, y + 5))
            
        # Draw Moves
        moves = session.move_history
        start_idx = max(0, len(moves) - 20) # Show last 10 rounds (20 moves)
        
        row_h = 24
        curr_y = y + 35
        
        turn_num = start_idx // 2 + 1
        
        for i in range(start_idx, len(moves), 2):
            if curr_y + row_h > y + h: break
            
            # Row Background for alternating
            if turn_num % 2 == 0:
                pygame.draw.rect(self.screen, COLORS.BACKGROUND, (x, curr_y, w, row_h))
            
            # Move Number
            num_surf = self.font_small.render(f"{turn_num}.", True, COLORS.TEXT_DIM)
            self.screen.blit(num_surf, (x + 10, curr_y + 2))
            
            # P1 Move
            p1_m = moves[i]
            p1_col = COLORS.TEXT_BRIGHT if (i == len(moves)-1 and session.turn == 2) else COLORS.TEXT
            s1 = self.font_small.render(p1_m, True, p1_col)
            self.screen.blit(s1, (x + 60, curr_y + 2))
            
            # P2 Move
            if i + 1 < len(moves):
                p2_m = moves[i+1]
                p2_col = COLORS.TEXT_BRIGHT if (i + 1 == len(moves)-1 and session.turn == 1) else COLORS.TEXT
                s2 = self.font_small.render(p2_m, True, p2_col)
                self.screen.blit(s2, (x + 160, curr_y + 2))
                
            curr_y += row_h
            turn_num += 1

    def format_time(self, seconds):
        if seconds < 0: seconds = 0
        return f"{int(seconds // 60):02}:{int(seconds % 60):02}"

    def render(self):
        self.screen.fill(COLORS.BACKGROUND)
        
        if self.state == "MENU":
            self.render_menu()
        elif self.state == "SETTINGS":
            self.render_settings()
        elif self.state == "HISTORY":
            self.render_history()
        elif self.state in ["GAME", "GAMEOVER"]:
            layout = self.get_layout(len(self.sessions))
            # Since we focused on single game view, we typically have 1 session
            for i, session in enumerate(self.sessions):
                layout_info = layout[i]
                board_rect_tuple, cs = layout_info
                bx, by, bw, bh = board_rect_tuple
                
                # Draw Lichess UI Chrome
                self.render_lichess_ui(session, layout_info)

                # --- BOARD RENDERING (Preserved Logic, new position) ---
                # Board Background
                # pygame.draw.rect(self.screen, COLORS.SURFACE, (bx-10, by-10, bw+20, bh+20), border_radius=5)
                
                for r in range(session.m):
                    for c in range(session.n):
                        cell_rect = pygame.Rect(bx + c * cs, by + r * cs, cs, cs)
                        
                        # Checkerboard pattern for grid background? Optional, but keeping simple line grid as per request "don't change board"
                        # Actually Lichess uses checkerboard. But user said "do not change board".
                        # So we keep the simple outline logic or whatever was there.
                        # The original code was: pygame.draw.rect(self.screen, COLORS.BACKGROUND, cell_rect, 1)
                        # We might need to adjust the color since Background is now the board background.
                        
                        pygame.draw.rect(self.screen, (60, 60, 60), cell_rect, 1)  # Subtle grid lines
                        
                        val = session.board.board[r][c]
                        if val != 0:
                            center = cell_rect.center
                            rad = cs // 3
                            if val == 1:
                                color = COLORS.TEXT_BRIGHT # White pieces for P1 usually
                                # Or stick to the original Primary/Secondary if the user liked colors.
                                # Lichess is B/W. Let's stick to the requested "Don't change board" strictness, 
                                # but maybe update the colors to be visible on dark bg.
                                color = COLORS.PRIMARY 
                                pygame.draw.line(self.screen, color, (center[0]-rad, center[1]-rad), (center[0]+rad, center[1]+rad), 3)
                                pygame.draw.line(self.screen, color, (center[0]-rad, center[1]+rad), (center[0]+rad, center[1]-rad), 3)
                            else:
                                color = COLORS.SECONDARY
                                pygame.draw.circle(self.screen, color, center, rad, 3)
                        
                        # Highlight Winning Line
                        if (r, c) in session.board.winning_cells:
                            pygame.draw.rect(self.screen, (255, 215, 0), cell_rect, 3) # Gold border
                            s = pygame.Surface((cs, cs))
                            s.set_alpha(60)
                            s.fill((255, 215, 0))
                            self.screen.blit(s, cell_rect.topleft)

                        # Last Move Highlight
                        elif session.board.last_move == (r, c):
                            pygame.draw.rect(self.screen, (100, 200, 100), cell_rect, 2)

                # Mouse Hover / Ghost Piece
                if session.winner is None and not self.state == "GAMEOVER":
                     is_p1_human = (session.turn == 1 and not session.ai_p1)
                     is_p2_human = (session.turn == 2 and not session.ai_p2)
                     if is_p1_human or is_p2_human:
                        mx, my = pygame.mouse.get_pos()
                        if bx <= mx < bx + bw and by <= my < by + bh:
                            hc = int((mx - bx) // cs)
                            hr = int((my - by) // cs)
                            if 0 <= hr < session.m and 0 <= hc < session.n:
                                if session.board.board[hr][hc] == 0:
                                    h_rect = pygame.Rect(bx + hc * cs, by + hr * cs, cs, cs)
                                    h_center = h_rect.center
                                    h_rad = cs // 3
                                    s_ghost = pygame.Surface((cs, cs), pygame.SRCALPHA)
                                    g_color = COLORS.PRIMARY if session.turn == 1 else COLORS.SECONDARY
                                    # Draw ghost shape transparently
                                    if session.turn == 1:
                                         pygame.draw.line(s_ghost, (*g_color, 128), (cs//2 - h_rad, cs//2 - h_rad), (cs//2 + h_rad, cs//2 + h_rad), 2)
                                         pygame.draw.line(s_ghost, (*g_color, 128), (cs//2 - h_rad, cs//2 + h_rad), (cs//2 + h_rad, cs//2 - h_rad), 2)
                                    else:
                                         pygame.draw.circle(s_ghost, (*g_color, 128), (cs//2, cs//2), h_rad, 2)
                                    self.screen.blit(s_ghost, h_rect.topleft)
                                    pygame.draw.rect(self.screen, (*COLORS.TEXT_BRIGHT, 50), h_rect, 1)
            
            if self.state == "GAMEOVER":
                 self.render_overlay()
                 self.render_gameover()

        pygame.display.flip()

    def render_menu(self):
        # Background Pattern
        self.screen.fill(COLORS.BACKGROUND)
        
        # Title
        title = self.font_title.render(f"MNK GAME ({self.k}-in-row)", True, COLORS.PRIMARY)
        title_rect = title.get_rect(center=(self.width // 2, 80))
        self.screen.blit(title, title_rect)

        # Settings Summary
        p1_type = "Human" if self.slider_p1.get_value() == 0 else f"AI-{int(self.slider_p1.get_value())}"
        p2_type = "Human" if self.slider_p2.get_value() == 0 else f"AI-{int(self.slider_p2.get_value())}"
        summary = f"{int(self.slider_m.get_value())}x{int(self.slider_n.get_value())} Grid | {p1_type} vs {p2_type}"
        sum_surf = self.font_small.render(summary, True, COLORS.TEXT_DIM)
        sum_rect = sum_surf.get_rect(center=(self.width // 2, 125))
        self.screen.blit(sum_surf, sum_rect)
        
        # Menu Panel
        panel_rect = pygame.Rect(0, 0, 300, 330)
        panel_rect.center = (self.width // 2, self.height // 2 + 20)
        pygame.draw.rect(self.screen, COLORS.SURFACE, panel_rect, border_radius=15)
        pygame.draw.rect(self.screen, COLORS.HIGHLIGHT, panel_rect, width=2, border_radius=15)
        
        for b in [self.btn_play, self.btn_history, self.btn_settings, self.btn_quit]: b.draw(self.screen)
        
    def render_history(self):
        title = self.font_title.render("GAME HISTORY", True, COLORS.ACCENT)
        self.screen.blit(title, title.get_rect(center=(self.width // 2, 60)))
        
        recents = self.record_manager.history.get('games', [])[-13:]
        recents.reverse()
        
        y = 120
        headers = ["Time", "P1", "P2", "Winner", "Size", "Moves"]
        x_positions = [50, 200, 350, 500, 650, 750]
        
        for i, h in enumerate(headers):
             self.screen.blit(self.font_ui.render(h, True, COLORS.PRIMARY), (x_positions[i], y))
        y += 40
        
        for g in recents:
             ts = time.strftime("%H:%M", time.localtime(g['timestamp']))
             w_val = g['winner']
             if w_val == 1: winner = "P1 Win"
             elif w_val == 2: winner = "P2 Win"
             elif w_val == 0: winner = "Draw"
             else: winner = "?"
             
             row = [ts, g['p1_name'], g['p2_name'], winner, f"{g['m']}x{g['n']} ({g['k']})", str(len(g.get('moves', [])))]
             for i, d in enumerate(row):
                 self.screen.blit(self.font_small.render(str(d)[:15], True, COLORS.TEXT), (x_positions[i], y))
             y += 30
             
        self.btn_back.draw(self.screen)
        
    def render_settings(self):
        title = self.font_title.render("SETTINGS", True, COLORS.ACCENT)
        self.screen.blit(title, title.get_rect(center=(self.width // 2, 80)))
        
        # Settings Panel
        panel_rect = pygame.Rect(0, 0, 400, 480)
        panel_rect.center = (self.width // 2, self.height // 2 - 30)
        pygame.draw.rect(self.screen, COLORS.SURFACE, panel_rect, border_radius=15)
        
        for s in [self.slider_m, self.slider_n, self.slider_k, self.slider_p1, self.slider_p2, self.slider_time, self.slider_increment]:
            s.draw(self.screen)
        self.btn_back.draw(self.screen)

    def render_overlay(self):
        s = pygame.Surface((self.width, self.height))
        s.set_alpha(150); s.fill((0,0,0))
        self.screen.blit(s, (0,0))

    def render_gameover(self):
        winner = self.sessions[0].winner if self.sessions else 0
        if winner == 0: msg = "DRAW!"; color = COLORS.TEXT
        elif winner == 1: msg = "PLAYER 1 WINS!"; color = COLORS.PRIMARY
        else: msg = "PLAYER 2 WINS!"; color = COLORS.SECONDARY
        text = self.font_title.render(msg, True, color)
        rect = text.get_rect(center=(self.width // 2, self.height // 2 - 60))
        pygame.draw.rect(self.screen, COLORS.SURFACE, rect.inflate(60, 40), border_radius=10)
        self.screen.blit(text, rect)
        self.btn_rematch.draw(self.screen)
        self.btn_menu.draw(self.screen)

    def update_settings_sliders(self, mouse_pos):
        for s in [self.slider_m, self.slider_n, self.slider_k, self.slider_p1, self.slider_p2, self.slider_time, self.slider_increment]:
            s.update(mouse_pos)
            
        val1 = self.slider_p1.get_value()
        elo1 = f" ({self.elo_manager.get_rating(val1)})" if val1 > 0 else ""
        self.slider_p1.custom_label = f"Player 1: {'Human' if val1==0 else f'AI Lvl {val1}{elo1}'}"
        
        val2 = self.slider_p2.get_value()
        elo2 = f" ({self.elo_manager.get_rating(val2)})" if val2 > 0 else ""
        self.slider_p2.custom_label = f"Player 2: {'Human' if val2==0 else f'AI Lvl {val2}{elo2}'}"
        
        self.slider_time.custom_label = f"Time Limit: {self.slider_time.get_value()} Minutes"
        self.slider_increment.custom_label = f"Increment: {self.slider_increment.get_value()} Seconds"

if __name__ == "__main__":
    app = GameApp()
    app.run()
