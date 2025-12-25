import sys
import os
import time
import math
import random
import threading
import copy
import pygame

# Import local modules
from game_engine import MNKBoard
from ai_opponents import AI
from ui_components import Button, Slider, COLORS
from tournament import RoundRobinTournament, KnockoutTournament
from game_session import GameSession
from records import RecordManager
from elo_system import EloManager

# Constants
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 800
FPS = 60
DEFAULT_TIME_LIMIT = 300 

class SoundManager:
    """
    Handles loading and playing of sound effects and music.
    """
    def __init__(self):
        self.sounds = {}
        self.music_playing = False
        self.use_generated = False
        try:
            import numpy as np
            self.np = np
            self.use_generated = True
        except ImportError:
            pass 

        try:
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
            self.generate_defaults()
        except pygame.error as e:
            print(f"Sound system error: {e}")

    def generate_defaults(self):
        if not self.use_generated: return
            
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
        self.sounds['timer_low'] = make_tone(1000, 0.1, 0.05)

    def play(self, name):
        if name in self.sounds:
            self.sounds[name].play()
    
    def set_music_volume(self, vol):
        try:
            pygame.mixer.music.set_volume(vol)
        except:
            pass

    def load_music(self, path):
        if os.path.exists(path):
            try:
                pygame.mixer.music.load(path)
                pygame.mixer.music.play(-1)
                pygame.mixer.music.set_volume(0.5)
                self.music_playing = True
            except (pygame.error, FileNotFoundError):
                pass
                
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
        
        # Fonts
        self.font_title = pygame.font.SysFont("Segoe UI", 60, bold=True)
        self.font_ui = pygame.font.SysFont("Segoe UI", 24)
        self.font_timer = pygame.font.SysFont("Consolas", 32, bold=True)
        self.font_small = pygame.font.SysFont("Segoe UI", 16)
        
        # Audio
        self.sound_manager = SoundManager()
        if os.path.exists("background_music.wav"):
            self.sound_manager.load_music("background_music.wav")
        elif os.path.exists("music.mp3"): 
            self.sound_manager.load_music("music.mp3")

        # Managers
        self.record_manager = RecordManager()
        self.elo_manager = EloManager()

        # Application State
        self.state = "MENU"
        self.state_timer = 0
        
        # Game Settings
        self.m = 10
        self.n = 10
        self.k = 5
        self.p1_level = 0
        self.p2_level = 0
        self.time_limit = DEFAULT_TIME_LIMIT 
        self.time_increment = 0
        
        # Active Sessions
        self.sessions = [] # List of GameSession objects
        self.tournament = None
        self.tournament_type = "RR" # RR or KO
        
        # UI
        self.tour_configs = [{'is_ai': False, 'level': 3} for _ in range(20)]
        self.init_ui()

    def init_ui(self):
        cx, cy = self.width // 2, self.height // 2
        
        # Save state if exists (preserve slider values during resize)
        s_tp = self.slider_tour_players.get_value() if hasattr(self, 'slider_tour_players') else 4
        s_tr = self.slider_tour_rounds.get_value() if hasattr(self, 'slider_tour_rounds') else 1
        s_tpi = self.slider_tour_p_idx.get_value() if hasattr(self, 'slider_tour_p_idx') else 1
        s_tpt = self.slider_tour_p_type.get_value() if hasattr(self, 'slider_tour_p_type') else 0
        s_tpl = self.slider_tour_p_level.get_value() if hasattr(self, 'slider_tour_p_level') else 3
        
        s_m = self.slider_m.get_value() if hasattr(self, 'slider_m') else self.m
        s_n = self.slider_n.get_value() if hasattr(self, 'slider_n') else self.n
        s_k = self.slider_k.get_value() if hasattr(self, 'slider_k') else self.k
        s_p1 = self.slider_p1.get_value() if hasattr(self, 'slider_p1') else self.p1_level
        s_p2 = self.slider_p2.get_value() if hasattr(self, 'slider_p2') else self.p2_level
        s_time = self.slider_time.get_value() if hasattr(self, 'slider_time') else 5
        s_inc = self.slider_increment.get_value() if hasattr(self, 'slider_increment') else 0
        s_vol = self.slider_music.get_value() if hasattr(self, 'slider_music') else 50
        
        # Menu
        self.btn_play = Button(cx - 100, cy - 120, 200, 50, "PLAY", self.font_ui, lambda: self.start_single_game())
        self.btn_tournament = Button(cx - 100, cy - 50, 200, 50, "TOURNAMENT", self.font_ui, lambda: self.set_state("TOURNAMENT_SETUP"))
        self.btn_history = Button(cx - 100, cy + 20, 200, 50, "HISTORY", self.font_ui, lambda: self.set_state("HISTORY"))
        self.btn_settings = Button(cx - 100, cy + 90, 200, 50, "SETTINGS", self.font_ui, lambda: self.set_state("SETTINGS"))
        self.btn_quit = Button(cx - 100, cy + 160, 200, 50, "QUIT", self.font_ui, lambda: self.quit_game())
        
        # Tournament Setup
        self.slider_tour_players = Slider(cx - 150, cy - 180, 300, 20, 2, 8, s_tp, self.font_ui, "Total Players")
        self.slider_tour_rounds = Slider(cx - 150, cy - 130, 300, 20, 1, 4, s_tr, self.font_ui, "Rounds")
        
        tour_label = self.btn_tour_type.text if hasattr(self, 'btn_tour_type') else "Type: Round Robin"
        self.btn_tour_type = Button(cx - 150, cy - 80, 300, 30, tour_label, self.font_ui, lambda: self.toggle_tour_type())
        
        self.slider_tour_p_idx = Slider(cx - 150, cy + 10, 300, 20, 1, 4, s_tpi, self.font_ui, "Edit Player")
        self.slider_tour_p_type = Slider(cx - 150, cy + 60, 300, 20, 0, 1, s_tpt, self.font_ui, "Type") 
        self.slider_tour_p_level = Slider(cx - 150, cy + 110, 300, 20, 1, 8, s_tpl, self.font_ui, "AI Level")
        
        self.btn_tour_start = Button(cx - 100, cy + 180, 200, 50, "START", self.font_ui, lambda: self.start_tournament())
        
        back_y = self.height - 100
        self.btn_back = Button(cx - 100, back_y, 200, 50, "BACK", self.font_ui, lambda: self.set_state("MENU"))
        
        # Tournament Hub
        self.btn_tour_play = Button(cx - 100, self.height - 150, 200, 50, "PLAY PENDING", self.font_ui, lambda: self.play_pending_matches())
        self.btn_tour_back = Button(cx - 100, back_y, 200, 50, "EXIT TO MENU", self.font_ui, lambda: self.set_state("MENU"))
        
        # Settings
        # Center settings block
        sy = cy - 250
        self.slider_m = Slider(cx - 150, sy, 300, 20, 3, 32, s_m, self.font_ui, "Rows (M)")
        self.slider_n = Slider(cx - 150, sy + 70, 300, 20, 3, 32, s_n, self.font_ui, "Cols (N)")
        self.slider_k = Slider(cx - 150, sy + 140, 300, 20, 3, 32, s_k, self.font_ui, "Win (K)")
        self.slider_p1 = Slider(cx - 150, sy + 210, 300, 20, 0, 8, s_p1, self.font_ui, "Player 1")
        self.slider_p2 = Slider(cx - 150, sy + 280, 300, 20, 0, 8, s_p2, self.font_ui, "Player 2")
        self.slider_time = Slider(cx - 150, sy + 350, 300, 20, 1, 30, s_time, self.font_ui, "Time (Mins)")
        self.slider_increment = Slider(cx - 150, sy + 420, 300, 20, 0, 60, s_inc, self.font_ui, "Increment (Sec)")
        self.slider_music = Slider(cx - 150, sy + 490, 300, 20, 0, 100, s_vol, self.font_ui, "Music Volume")
        
        # Game Over (Single Game)
        self.btn_rematch = Button(cx - 100, cy + 80, 200, 50, "PLAY AGAIN", self.font_ui, lambda: self.start_single_game())
        self.btn_menu = Button(cx - 100, cy + 150, 200, 50, "MENU", self.font_ui, lambda: self.set_state("MENU"))
        self.btn_sessions_back = Button(cx - 100, self.height - 80, 200, 50, "RETURN", self.font_ui, lambda: self.return_from_game())
        
    def set_state(self, state):
        self.state = state
        self.state_timer = time.time()
        
    def quit_game(self):
        pygame.quit()
        sys.exit()

    def toggle_tour_type(self):
        if self.tournament_type == "RR":
            self.tournament_type = "KO"
            self.btn_tour_type.text = "Type: Knockout"
        else:
            self.tournament_type = "RR"
            self.btn_tour_type.text = "Type: Round Robin"

    def start_single_game(self):
        # Apply settings
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
        
    def start_tournament(self):
        n_players = self.slider_tour_players.get_value()
        n_rounds = self.slider_tour_rounds.get_value()
        
        configs = []
        for i in range(n_players):
            c = self.tour_configs[i]
            configs.append({
                'name': f"Bot {i+1}" if c['is_ai'] else f"Player {i+1}",
                'is_ai': c['is_ai'],
                'level': c['level']
            })
            
        if self.tournament_type == "RR":
            self.tournament = RoundRobinTournament(configs, n_rounds)
        else:
            self.tournament = KnockoutTournament(configs)
            
        self.sessions = []
        self.set_state("TOURNAMENT_HUB")
        
    def play_pending_matches(self):
        if not self.tournament: return
        matches = self.tournament.get_pending_matches(limit=4) # allow up to 4 concurrent
        
        if not matches: return
        
        for m_data in matches:
            mid, p1, p2 = m_data
            
            # Check if match already running
            already_running = False
            for s in self.sessions:
                if s.match_id == mid and s.is_active:
                    already_running = True
            
            if already_running: continue
            
            # Create session
            # Use settings for board size? Or fixed? Default to current settings.
            p1_conf = {'name': p1.name, 'is_ai': p1.is_ai, 'level': p1.ai_level}
            p2_conf = {'name': p2.name, 'is_ai': p2.is_ai, 'level': p2.ai_level}
            
            # For tournament, maybe enforce fast time limit for AI vs AI?
            # Or use global settings. Using global for now.
            
            session = GameSession(self.m, self.n, self.k, p1_conf, p2_conf, self.time_limit, self.time_increment, match_id=mid)
            self.sessions.append(session)
            
        self.set_state("GAME")

    def return_from_game(self):
        if self.tournament:
            self.set_state("TOURNAMENT_HUB")
        else:
            self.set_state("MENU")

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
                        # Save current size before going full
                        self.windowed_size = (self.width, self.height)
                        self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
                    else:
                        w, h = self.windowed_size
                        self.screen = pygame.display.set_mode((w, h), pygame.RESIZABLE)
                    self.width, self.height = self.screen.get_size()
                    self.init_ui()

            if event.type == pygame.VIDEORESIZE:
                # Only update if not fullscreen (to avoid conflict if system sends resize events during fullscreen toggle)
                if not self.fullscreen:
                    self.width, self.height = event.w, event.h
                    self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
                    self.init_ui()
            
            if self.state == "MENU":
                for b in [self.btn_play, self.btn_tournament, self.btn_history, self.btn_settings, self.btn_quit]:
                     b.handle_event(event)

            elif self.state == "TOURNAMENT_SETUP":
                for s in [self.slider_tour_players, self.slider_tour_rounds, self.slider_tour_p_idx, self.slider_tour_p_type, self.slider_tour_p_level]:
                    s.handle_event(event)
                for b in [self.btn_tour_start, self.btn_back, self.btn_tour_type]:
                    b.handle_event(event)
                    
            elif self.state == "TOURNAMENT_HUB":
                for b in [self.btn_tour_play, self.btn_tour_back]:
                    b.handle_event(event)
                    
            elif self.state == "SETTINGS":
                 for s in [self.slider_m, self.slider_n, self.slider_k, self.slider_p1, self.slider_p2, self.slider_time, self.slider_increment, self.slider_music]:
                     s.handle_event(event)
                 self.btn_back.handle_event(event)
            
            elif self.state == "HISTORY":
                 self.btn_back.handle_event(event)

                 
            elif self.state in ["GAME", "GAMEOVER"]:
                handled = False
                # If single game and over, handle buttons
                if not self.tournament and self.sessions and self.sessions[0].winner is not None:
                     for b in [self.btn_rematch, self.btn_menu]: 
                         if b.handle_event(event): handled = True
                
                elif self.tournament:
                    if self.btn_sessions_back.handle_event(event): handled = True
                
                if handled: continue

                # Handle clicks on boards
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                     self.handle_game_click(mouse_pos)

    def handle_game_click(self, pos):
        # Dispatch click to appropriate session board
        # Need to know layout
        layout = self.get_layout(len(self.sessions))
        
        for i, session in enumerate(self.sessions):
            rect, cell_size = layout[i]
            # Check collision with rect
             # Adjust pos relative to rect
            if session.winner is None:
                 # Check human turn
                 is_p1_human = (session.turn == 1 and not session.ai_p1)
                 is_p2_human = (session.turn == 2 and not session.ai_p2)
                 
                 if is_p1_human or is_p2_human:
                     # Translate click
                      # rect is (x, y, w, h) of board area
                      bx, by, bw, bh = rect
                      if bx <= pos[0] < bx + bw and by <= pos[1] < by + bh:
                          c = int((pos[0] - bx) // cell_size)
                          r = int((pos[1] - by) // cell_size)
                          session.make_move(r, c)
        
    def update(self):
        now = time.time()
        
        # Update Menu Sliders logic in TOURNAMENT_SETUP
        if self.state == "TOURNAMENT_SETUP":
            self.update_tour_sliders(pygame.mouse.get_pos())
            for b in [self.btn_tour_start, self.btn_back, self.btn_tour_type]: b.update(pygame.mouse.get_pos())
            
        elif self.state == "SETTINGS":
            self.update_settings_sliders(pygame.mouse.get_pos())
            self.btn_back.update(pygame.mouse.get_pos())
        
        elif self.state == "MENU":
            for b in [self.btn_play, self.btn_tournament, self.btn_history, self.btn_settings, self.btn_quit]: 
                b.update(pygame.mouse.get_pos())

        elif self.state == "HISTORY":
             self.btn_back.update(pygame.mouse.get_pos())

        elif self.state == "TOURNAMENT_HUB":
            for b in [self.btn_tour_play, self.btn_tour_back]: b.update(pygame.mouse.get_pos())
            
            # Auto play pending AI matches if any?
            # User asked for simultaneous pending bot games if multiple.
            # We check periodically.
            if self.tournament and not self.tournament.is_finished():
                # If we have < 4 active sessions, fetch more?
                active_count = sum(1 for s in self.sessions if s.is_active)
                if active_count < 4:
                    # Check for pending AI-only matches
                    pending = self.tournament.get_pending_matches(limit=10)
                    # Filter for AI only and not running
                    for m in pending:
                        mid, p1, p2 = m
                        if p1.is_ai and p2.is_ai:
                            # Start it if not running
                             running = any(s.match_id == mid for s in self.sessions if s.is_active)
                             if not running:
                                 self.play_pending_matches() # This handles getting them
                                 break # One at a time to avoid spam or rely on method
        
        elif self.state == "GAME" or self.state == "GAMEOVER":
             # Update all sessions
             all_finished = True
             for session in self.sessions:
                 if session.is_active:
                     session.update(now)
                     # Handle events
                     for e in getattr(session, 'events', []):
                         if e == 'place': self.sound_manager.play('place')
                         elif e == 'game_over': 
                             self.sound_manager.play('win' if session.winner else 'lose')
                             # Save Record
                             self.save_game_result(session)
                             if self.tournament and session.match_id is not None:
                                 self.tournament.record_result(session.match_id, session.winner if session.winner > 0 else -1)
                             session.is_active = False # Stop updating
                     
                     session.events = []
                     
                     if session.winner is None: 
                         all_finished = False

             if not self.tournament:
                 if all_finished: self.state = "GAMEOVER"

             if self.tournament:
                 self.btn_sessions_back.update(pygame.mouse.get_pos())
             else:
                 if self.state == "GAMEOVER":
                     for b in [self.btn_rematch, self.btn_menu]: b.update(pygame.mouse.get_pos())

    def save_game_result(self, session):
        # Convert winner (1/2) to name/id
        winner_val = session.winner
        
        rec = {
            'timestamp': time.time(),
            'p1_name': session.p1_name,
            'p1_level': session.p1_config.get('level', 0),
            'p2_name': session.p2_name,
            'p2_level': session.p2_config.get('level', 0),
            'm': session.m, 'n': session.n, 'k': session.k,
            'winner': winner_val,
            'win_reason': session.win_reason,
            'moves': session.move_history,
            'mode': 'Tournament' if self.tournament else 'Quick'
        }
        self.record_manager.add_game_record(rec)

        # Update ELO if both are AI
        p1_ai = session.p1_config.get('is_ai', False)
        p2_ai = session.p2_config.get('is_ai', False)
        
        if p1_ai and p2_ai:
            p1_lvl = session.p1_config.get('level')
            p2_lvl = session.p2_config.get('level')
            if p1_lvl and p2_lvl:
                self.elo_manager.update_ratings(p1_lvl, p2_lvl, winner_val, session.m, session.n, session.k)

    def get_layout(self, num_sessions):
        # Returns list of (rect, cell_size) for each session
        # Layouts: 1=(1x1), 2=(2x1), 3-4=(2x2), 5-6=(3x2), 7-9=(3x3)
        
        # Margin
        margin = 20
        header = 60
        area_w = self.width - 2*margin
        area_h = self.height - 2*margin - header
        
        cols = 1
        rows = 1
        if num_sessions >= 2: cols=2
        if num_sessions >= 3: cols=2; rows=2
        if num_sessions >= 5: cols=3; rows=2
        if num_sessions >= 7: cols=3; rows=3
        
        # Calculate cell size
        panel_w = area_w // cols
        panel_h = area_h // rows
        
        results = []
        for i in range(num_sessions):
            r = i // cols
            c = i % cols
            x = margin + c * panel_w
            y = margin + header + r * panel_h
            
            # Board needs to fit in panel_w, panel_h with some padding
            # Aspect ratio of board is n/m
            # Maximize cell_size
            # (n * cs) <= panel_w - pad
            # (m * cs) <= panel_h - pad
            max_w = panel_w - 20
            max_h = panel_h - 40 # Space for text
            
            cs = min(max_w // self.n, max_h // self.m)
            if cs < 5: cs = 5
            
            bw = cs * self.n
            bh = cs * self.m
            
            bx = x + (panel_w - bw) // 2
            by = y + (panel_h - bh) // 2 + 10
            
            results.append(((bx, by, bw, bh), cs))
            
        return results

    def format_time(self, seconds):
        if seconds < 0: seconds = 0
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m:02}:{s:02}"

    def render(self):
        self.screen.fill(COLORS.BACKGROUND)
        
        if self.state == "MENU":
            self.render_menu()
        elif self.state == "SETTINGS":
            self.render_settings()
        elif self.state == "HISTORY":
            self.render_history()
        elif self.state == "TOURNAMENT_SETUP":
            self.render_tournament_setup()
        elif self.state == "TOURNAMENT_HUB":
            self.render_tournament_hub()
        elif self.state in ["GAME", "GAMEOVER"]:
            layout = self.get_layout(len(self.sessions))
            
            for i, session in enumerate(self.sessions):
                rect, cs = layout[i]
                bx, by, bw, bh = rect
                
                # Draw Panel Background
                panel_rect = pygame.Rect(bx - 5, by - 60, bw + 10, bh + 70)
                pygame.draw.rect(self.screen, COLORS.SURFACE, panel_rect, border_radius=8)
                
                # Draw Header Info with Details
                t1 = self.format_time(session.timer_p1)
                t2 = self.format_time(session.timer_p2)
                
                # Make names decorative
                def get_p_str(name, conf):
                    if conf.get('is_ai'): return f"{name}"
                    return f"{name}"
                
                info1 = f"{get_p_str(session.p1_name, session.p1_config)} ({t1})"
                info2 = f"{get_p_str(session.p2_name, session.p2_config)} ({t2})"
                
                header_text = f"{info1} vs {info2}"
                if session.winner is not None:
                     if session.winner == 0: header_text = "DRAW"
                     else: header_text = f"WINNER: {session.p1_name if session.winner==1 else session.p2_name}"
                
                # Status line
                status = f"Move: {len(session.move_history)//2 + 1}"
                if session.winner is None:
                     status += f" | Turn: {session.p1_name if session.turn==1 else session.p2_name}"
                     
                txt = self.font_small.render(header_text, True, COLORS.TEXT)
                # Center text above board
                txt_rect = txt.get_rect(center=(bx + bw // 2, by - 45))
                self.screen.blit(txt, txt_rect)
                
                txt2 = self.font_small.render(status, True, COLORS.HIGHLIGHT)
                txt2_rect = txt2.get_rect(center=(bx + bw // 2, by - 20))
                self.screen.blit(txt2, txt2_rect)
                
                # Draw Board
                for r in range(session.m):
                    for c in range(session.n):
                        cell_rect = pygame.Rect(bx + c * cs, by + r * cs, cs, cs)
                        pygame.draw.rect(self.screen, COLORS.BACKGROUND, cell_rect, 1)
                        
                        val = session.board.board[r][c]
                        if val != 0:
                            center = cell_rect.center
                            rad = cs // 3
                            if val == 1:
                                color = COLORS.PRIMARY
                                pygame.draw.line(self.screen, color, (center[0]-rad, center[1]-rad), (center[0]+rad, center[1]+rad), 2)
                                pygame.draw.line(self.screen, color, (center[0]-rad, center[1]+rad), (center[0]+rad, center[1]-rad), 2)
                            else:
                                color = COLORS.SECONDARY
                                pygame.draw.circle(self.screen, color, center, rad, 2)
                        
                        if session.board.last_move == (r, c):
                            pygame.draw.rect(self.screen, COLORS.ACCENT, cell_rect, 2)
            
            if self.tournament:
                self.btn_sessions_back.draw(self.screen)
            elif self.state == "GAMEOVER":
                 self.render_overlay()
                 self.render_gameover()

        pygame.display.flip()

    # Reuse render helpers
    def render_menu(self):
        title = self.font_title.render(f"MNK GAME ({self.k}-in-row)", True, COLORS.PRIMARY)
        title_rect = title.get_rect(center=(self.width // 2, 100))
        self.screen.blit(title, title_rect)
        self.btn_play.draw(self.screen)
        self.btn_tournament.draw(self.screen)
        self.btn_history.draw(self.screen)
        self.btn_settings.draw(self.screen)
        self.btn_quit.draw(self.screen)
        
    def render_history(self):
        title = self.font_title.render("GAME HISTORY", True, COLORS.ACCENT)
        title_rect = title.get_rect(center=(self.width // 2, 60))
        self.screen.blit(title, title_rect)
        
        # Show last 13 games
        games = self.record_manager.history.get('games', [])
        recents = games[-13:]
        recents.reverse()
        
        y = 120
        # Headers
        headers = ["Time", "P1", "P2", "Winner", "Size", "Moves"]
        x_positions = [50, 200, 350, 500, 650, 750]
        
        for i, h in enumerate(headers):
             s = self.font_ui.render(h, True, COLORS.PRIMARY)
             self.screen.blit(s, (x_positions[i], y))
        y += 40
        
        for g in recents:
             ts = time.strftime("%H:%M", time.localtime(g['timestamp']))
             p1 = g['p1_name']
             p2 = g['p2_name']
             w_val = g['winner']
             if w_val == 1: winner = "P1 Win"
             elif w_val == 2: winner = "P2 Win"
             elif w_val == 0: winner = "Draw"
             else: winner = "?"
             
             size = f"{g['m']}x{g['n']} ({g['k']})"
             moves_cnt = str(len(g.get('moves', [])))
             
             row = [ts, p1, p2, winner, size, moves_cnt]
             for i, d in enumerate(row):
                 s = self.font_small.render(str(d)[:15], True, COLORS.TEXT)
                 self.screen.blit(s, (x_positions[i], y))
             y += 30
             
        self.btn_back.draw(self.screen)
        
    def render_settings(self):
        title = self.font_title.render("SETTINGS", True, COLORS.ACCENT)
        title_rect = title.get_rect(center=(self.width // 2, 80))
        self.screen.blit(title, title_rect)
        self.slider_m.draw(self.screen)
        self.slider_n.draw(self.screen)
        self.slider_k.draw(self.screen)
        self.slider_p1.draw(self.screen)
        self.slider_p2.draw(self.screen)
        self.slider_time.draw(self.screen)
        self.slider_increment.draw(self.screen)
        self.slider_music.draw(self.screen)
        self.btn_back.draw(self.screen)

    def render_tournament_setup(self):
        title = self.font_title.render("TOURNAMENT SETUP", True, COLORS.ACCENT)
        title_rect = title.get_rect(center=(self.width // 2, 80))
        self.screen.blit(title, title_rect)
        self.slider_tour_players.draw(self.screen)
        self.slider_tour_rounds.draw(self.screen)
        self.btn_tour_type.draw(self.screen)
        pygame.draw.line(self.screen, COLORS.HIGHLIGHT, (100, 230), (self.width - 100, 230), 2)
        self.slider_tour_p_idx.draw(self.screen)
        self.slider_tour_p_type.draw(self.screen)
        if self.tour_configs[self.slider_tour_p_idx.get_value() - 1]['is_ai']:
             self.slider_tour_p_level.draw(self.screen)
        self.btn_tour_start.draw(self.screen)
        self.btn_back.draw(self.screen)
        
    def render_tournament_hub(self):
        title = self.font_title.render("TOURNAMENT STANDINGS", True, COLORS.ACCENT)
        title_rect = title.get_rect(center=(self.width // 2, 60))
        self.screen.blit(title, title_rect)
        
        y = 120
        headers = ["Rank", "Name", "P", "W", "D", "L", "Pts"]
        x_positions = [100, 200, 400, 500, 600, 700, 800]
        for i, h in enumerate(headers):
             s = self.font_ui.render(h, True, COLORS.PRIMARY)
             self.screen.blit(s, (x_positions[i], y))
        y += 40
        standings = self.tournament.get_standings()
        for i, p in enumerate(standings):
            row_data = [str(i+1), p.name, str(p.played), str(p.wins), str(p.draws), str(p.losses), str(p.points)]
            for j, d in enumerate(row_data):
                color = COLORS.TEXT
                if i == 0 and self.tournament.is_finished(): color = (255, 215, 0)
                s = self.font_ui.render(d, True, color)
                self.screen.blit(s, (x_positions[j], y))
            y += 30
            
        self.btn_tour_play.draw(self.screen)
        self.btn_tour_back.draw(self.screen)

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

    def update_tour_sliders(self, mouse_pos):
        self.slider_tour_players.update(mouse_pos)
        self.slider_tour_rounds.update(mouse_pos)
        self.slider_tour_p_idx.update(mouse_pos)
        self.slider_tour_p_type.update(mouse_pos)
        self.slider_tour_p_level.update(mouse_pos)
        
        # Logic sync
        max_p = self.slider_tour_players.get_value()
        self.slider_tour_p_idx.max_val = max_p
        if self.slider_tour_p_idx.val > max_p: self.slider_tour_p_idx.val = max_p
        
        current_idx = self.slider_tour_p_idx.get_value() - 1
        if not hasattr(self, 'last_tour_idx'): self.last_tour_idx = current_idx
        
        if current_idx != self.last_tour_idx:
             conf = self.tour_configs[current_idx]
             self.slider_tour_p_type.val = 1 if conf['is_ai'] else 0
             self.slider_tour_p_level.val = conf['level']
             self.last_tour_idx = current_idx
        else:
             self.tour_configs[current_idx]['is_ai'] = (self.slider_tour_p_type.get_value() == 1)
             self.tour_configs[current_idx]['level'] = self.slider_tour_p_level.get_value()
             
        # Update labels
        self.slider_tour_players.custom_label = f"Total Players: {max_p}"
        self.slider_tour_rounds.custom_label = f"Rounds: {self.slider_tour_rounds.get_value()}"
        self.slider_tour_p_idx.custom_label = f"Editing Config: Player {current_idx + 1}"
        is_ai = self.tour_configs[current_idx]['is_ai']
        self.slider_tour_p_type.custom_label = f"Type: {'AI' if is_ai else 'Human'}"
        self.slider_tour_p_type.custom_label = f"Type: {'AI' if is_ai else 'Human'}"
        if is_ai: 
            lvl = self.tour_configs[current_idx]['level']
            elo = self.elo_manager.get_rating(lvl)
            self.slider_tour_p_level.custom_label = f"AI Strength: Level {lvl} (ELO: {elo})"
        else: self.slider_tour_p_level.custom_label = "AI Strength: N/A"

    def update_settings_sliders(self, mouse_pos):
        for s in [self.slider_m, self.slider_n, self.slider_k, self.slider_p1, self.slider_p2, self.slider_time, self.slider_increment, self.slider_music]:
            s.update(mouse_pos)
            
        val1 = self.slider_p1.get_value()
        elo1 = f" ({self.elo_manager.get_rating(val1)})" if val1 > 0 else ""
        self.slider_p1.custom_label = f"Player 1: {'Human' if val1==0 else f'AI Lvl {val1}{elo1}'}"
        
        val2 = self.slider_p2.get_value()
        elo2 = f" ({self.elo_manager.get_rating(val2)})" if val2 > 0 else ""
        self.slider_p2.custom_label = f"Player 2: {'Human' if val2==0 else f'AI Lvl {val2}{elo2}'}"
        self.slider_time.custom_label = f"Time Limit: {self.slider_time.get_value()} Minutes"
        self.slider_increment.custom_label = f"Increment: {self.slider_increment.get_value()} Seconds"
        mus_val = self.slider_music.get_value()
        self.slider_music.custom_label = f"Music Volume: {mus_val}%"
        self.sound_manager.set_music_volume(mus_val / 100.0)

if __name__ == "__main__":
    app = GameApp()
    app.run()
