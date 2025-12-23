import sys
import os
import subprocess
import time
import math
import random
import threading
import copy

# Bootstrap: Fix for Python 3.13t (Free-Threaded) incompatibility
if "python3.13t" in sys.executable.lower() or "t.exe" in sys.executable.lower():
    print("(!) Detected incompatible Python 3.13t. Restarting with standard Python...", file=sys.stderr)
    standard_python = os.path.join(os.path.dirname(sys.executable), "python.exe")
    if not os.path.exists(standard_python):
        standard_python = "python"
    try:
        subprocess.check_call([standard_python, os.path.abspath(__file__)] + sys.argv[1:])
        sys.exit(0)
    except (OSError, subprocess.SubprocessError):
        sys.exit(1)

import pygame

# Import local modules
from game_engine import MNKBoard
from ai_opponents import AI
from ui_components import Button, Slider, COLORS

# Constants
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 800
FPS = 60
DEFAULT_TIME_LIMIT = 300 # 5 minutes in seconds

class SoundManager:
    def __init__(self):
        self.sounds = {}
        self.music_playing = False
        self.use_generated = False
        try:
            import numpy as np
            self.np = np
            self.use_generated = True
        except ImportError:
            pass # Numpy optional

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
        if name in self.sounds: self.sounds[name].play()
    
    def set_music_volume(self, vol):
        try:
            pygame.mixer.music.set_volume(vol)
        except: pass

    def load_music(self, path):
        if os.path.exists(path):
            try:
                pygame.mixer.music.load(path)
                pygame.mixer.music.play(-1)
                pygame.mixer.music.set_volume(0.5)
                self.music_playing = True
            except (pygame.error, FileNotFoundError): pass

class GameApp:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("MNK Game Deluxe - AI Edition")
        self.clock = pygame.time.Clock()
        
        # Fonts
        self.font_title = pygame.font.SysFont("Segoe UI", 60, bold=True)
        self.font_ui = pygame.font.SysFont("Segoe UI", 24)
        self.font_timer = pygame.font.SysFont("Consolas", 32, bold=True)
        
        # Audio
        self.sound_manager = SoundManager()
        if os.path.exists("background_music.wav"):
            self.sound_manager.load_music("background_music.wav")
        elif os.path.exists("music.mp3"): 
            self.sound_manager.load_music("music.mp3")

        # Game State
        self.state = "MENU"
        self.m = 10
        self.n = 10
        self.k = 5
        self.p1_level = 0 # 0 = Human, 1-8 = AI Levels
        self.p2_level = 0 # 0 = Human, 1-8 = AI Levels
        
        # Timers (in seconds)
        self.time_limit = DEFAULT_TIME_LIMIT 
        self.timer_p1 = self.time_limit
        self.timer_p2 = self.time_limit
        self.last_frame_time = 0
        
        # Game Objects
        self.board = None
        self.ai_p1 = None
        self.ai_p2 = None
        self.ai_thread = None
        self.ai_move = None
        self.turn = 1
        self.winner = None
        self.win_reason = ""
        
        # UI Elements
        self.init_ui()

    def init_ui(self):
        cx, cy = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2
        
        # Menu Buttons
        self.btn_play = Button(cx - 100, cy - 50, 200, 50, "PLAY", self.font_ui, lambda: self.start_game())
        self.btn_settings = Button(cx - 100, cy + 20, 200, 50, "SETTINGS", self.font_ui, lambda: self.set_state("SETTINGS"))
        self.btn_quit = Button(cx - 100, cy + 90, 200, 50, "QUIT", self.font_ui, lambda: self.quit_game())
        
        # Settings UI
        self.slider_m = Slider(cx - 150, 120, 300, 20, 3, 20, self.m, self.font_ui, "Rows (M)")
        self.slider_n = Slider(cx - 150, 190, 300, 20, 3, 20, self.n, self.font_ui, "Cols (N)")
        self.slider_k = Slider(cx - 150, 260, 300, 20, 3, 10, self.k, self.font_ui, "Win (K)")
        
        # Player Levels (0-8)
        self.slider_p1 = Slider(cx - 150, 330, 300, 20, 0, 8, self.p1_level, self.font_ui, "Player 1")
        self.slider_p2 = Slider(cx - 150, 400, 300, 20, 0, 8, self.p2_level, self.font_ui, "Player 2")
        
        # Time Control
        self.slider_time = Slider(cx - 150, 470, 300, 20, 1, 30, 5, self.font_ui, "Time (Mins)")

        # Music Volume (0-100)
        self.slider_music = Slider(cx - 150, 540, 300, 20, 0, 100, 50, self.font_ui, "Music Volume")

        self.btn_back = Button(cx - 100, 650, 200, 50, "BACK", self.font_ui, lambda: self.set_state("MENU"))
        
        # Game Over UI
        self.btn_rematch = Button(cx - 100, cy + 80, 200, 50, "PLAY AGAIN", self.font_ui, lambda: self.start_game())
        self.btn_menu = Button(cx - 100, cy + 150, 200, 50, "MENU", self.font_ui, lambda: self.set_state("MENU"))

    def set_state(self, state):
        self.state = state
        self.sound_manager.play('click')

    def start_game(self):
        self.m = self.slider_m.get_value()
        self.n = self.slider_n.get_value()
        self.k = self.slider_k.get_value()
        
        # Update Player Selection
        self.p1_level = self.slider_p1.get_value()
        self.p2_level = self.slider_p2.get_value()
        
        self.time_limit = self.slider_time.get_value() * 60
        self.timer_p1 = self.time_limit
        self.timer_p2 = self.time_limit
        self.last_frame_time = time.time()
        
        if self.k > max(self.m, self.n):
            self.k = max(self.m, self.n)

        print(f"\n{'='*30}")
        print(f"NEW GAME STARTED")
        print(f"Board Size: {self.m}x{self.n}")
        print(f"Win Condition: {self.k}-in-a-row")
        print(f"Player 1: {'Human' if self.p1_level == 0 else f'AI Level {self.p1_level}'}")
        print(f"Player 2: {'Human' if self.p2_level == 0 else f'AI Level {self.p2_level}'}")
        print(f"Time Limit: {self.slider_time.get_value()} mins")
        print(f"{'='*30}\n")

        self.board = MNKBoard(self.m, self.n, self.k)
        
        self.ai_p1 = AI(self.p1_level) if self.p1_level > 0 else None
        self.ai_p2 = AI(self.p2_level) if self.p2_level > 0 else None
            
        self.turn = 1
        self.winner = None
        self.win_reason = ""
        self.set_state("GAME")

    def quit_game(self):
        pygame.quit()
        sys.exit()

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
            if event.type == pygame.QUIT:
                self.quit_game()
            
            if self.state == "MENU":
                for btn in [self.btn_play, self.btn_settings, self.btn_quit]:
                    btn.click_sound = self.sound_manager.sounds.get('click')
                    btn.handle_event(event)
            
            elif self.state == "SETTINGS":
                for slider in [self.slider_m, self.slider_n, self.slider_k, self.slider_p1, self.slider_p2, self.slider_time, self.slider_music]:
                    slider.handle_event(event)

                if self.btn_back.handle_event(event):
                    # Save settings logic if needed
                    pass

            elif self.state == "GAME":
                if self.winner is None and event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    # Allow click only if Human Turn
                    is_p1_human = (self.turn == 1 and not self.ai_p1)
                    is_p2_human = (self.turn == 2 and not self.ai_p2)
                    
                    if is_p1_human or is_p2_human:
                        self.handle_board_click(mouse_pos)
            
            elif self.state == "GAMEOVER":
                for btn in [self.btn_rematch, self.btn_menu]:
                    btn.click_sound = self.sound_manager.sounds.get('click')
                    btn.handle_event(event)

    def handle_board_click(self, pos):
        # Calculate grid parameters
        cell_size = min(600 // self.m, 800 // self.n, 60)
        board_w = cell_size * self.n
        board_h = cell_size * self.m
        start_x = (SCREEN_WIDTH - board_w) // 2
        start_y = (SCREEN_HEIGHT - board_h) // 2 + 30 # Offset for header
        
        if pos[0] >= start_x and pos[0] < start_x + board_w and \
           pos[1] >= start_y and pos[1] < start_y + board_h:
            
            c = (pos[0] - start_x) // cell_size
            r = (pos[1] - start_y) // cell_size
            
            if self.board.make_move(r, c, self.turn):
                self.sound_manager.play('place')
                t_left = self.timer_p1 if self.turn == 1 else self.timer_p2
                print(f"[Player {self.turn}] Move: ({r}, {c}) | Time Left: {int(t_left//60)}:{int(t_left%60):02d}")
                self.check_game_end()
                if not self.winner:
                    self.turn = 3 - self.turn
                    self.render()
                    # Prepare for AI move in main loop

    def check_game_end(self):
        if self.board.winner:
            self.winner = self.turn
            self.win_reason = "Connect " + str(self.k)
            self.sound_manager.play('win' if self.turn == 1 else 'lose')
            self.state = "GAMEOVER"
        elif self.board.is_full():
            self.winner = 0
            self.win_reason = "Draw"
            self.sound_manager.play('lose')
            self.state = "GAMEOVER"
            
        if self.state == "GAMEOVER":
             print(f"\nGAME OVER: {self.win_reason}")
             if self.winner: print(f"Winner: Player {self.winner}")
             else: print("Result: Draw")
             print("-" * 30)

    def run_ai_thread(self, board_copy, ai_instance, player_num):
        start_time = time.time()
        
        # Simple time management: Use 10% of remaining time, or at least 1.0s, max 20s
        # If timer is very low, play faster
        remaining = self.timer_p1 if player_num == 1 else self.timer_p2
        
        # Base calculation
        limit = max(0.5, min(20.0, remaining * 0.1))
        
        # Hard Safety Clamp: Never allow AI to use more time than we have left minus buffer
        if limit > remaining - 0.5:
             limit = max(0.1, remaining - 0.5)

        # If deeply low on time, go very fast
        if remaining < 10: 
            limit = min(limit, 0.5)
        
        self.ai_move = ai_instance.get_move(board_copy, player_num, time_limit=limit)
        elapsed = time.time() - start_time
        if elapsed < 0.5:
            time.sleep(0.5 - elapsed) 

    def update(self):
        now = time.time()
        dt = now - self.last_frame_time
        self.last_frame_time = now
        
        if self.state == "GAME" and self.winner is None:
            # Update Timers
            if self.turn == 1:
                self.timer_p1 -= dt
                if self.timer_p1 <= 0:
                    self.winner = 2
                    self.win_reason = "Time Out"
                    self.state = "GAMEOVER"
            else: # Player 2
                self.timer_p2 -= dt
                if self.timer_p2 <= 0:
                    self.winner = 1
                    self.win_reason = "Time Out"
                    self.state = "GAMEOVER"

            # AI Logic
            current_ai = self.ai_p1 if self.turn == 1 else self.ai_p2
            if self.winner is None and current_ai:
                if self.ai_thread is None:
                    # Start thinking
                    board_copy = copy.deepcopy(self.board)
                    self.ai_thread = threading.Thread(target=self.run_ai_thread, args=(board_copy, current_ai, self.turn), daemon=True)
                    self.ai_thread.start()
                
                elif not self.ai_thread.is_alive():
                    # Finished thinking
                    self.ai_thread = None
                    if self.ai_move:
                        r, c = self.ai_move
                        # Only make move if it's still AI's turn (game check logic might have ended it?)
                        # Actually check if move is valid
                        if self.board.make_move(r, c, self.turn):
                             self.sound_manager.play('place')
                             time_left = self.timer_p1 if self.turn == 1 else self.timer_p2
                             print(f"[Player {self.turn} (AI Lvl {current_ai.level})] Move: ({r}, {c}) | Time Left: {int(time_left//60)}:{int(time_left%60):02d}")
                             self.check_game_end()
                             if not self.winner:
                                 self.turn = 3 - self.turn
                                 self.render()
                    
                    self.ai_move = None

        mouse_pos = pygame.mouse.get_pos()
        if self.state == "MENU":
            for btn in [self.btn_play, self.btn_settings, self.btn_quit]:
                btn.update(mouse_pos)
        elif self.state == "SETTINGS":
            # Just reading value here to show labels
            pass 
            # We want to force slider labels update before drawing
            val1 = self.slider_p1.get_value()
            if val1 == 0: self.slider_p1.custom_label = "Player 1: Human"
            else: self.slider_p1.custom_label = f"Player 1: AI Lvl {val1}"

            val2 = self.slider_p2.get_value()
            if val2 == 0: self.slider_p2.custom_label = "Player 2: Human"
            else: self.slider_p2.custom_label = f"Player 2: AI Lvl {val2}"
            
            # Same for time
            t_val = self.slider_time.get_value()
            self.slider_time.custom_label = f"Time Limit: {t_val} Minutes"

            # Music
            mus_val = self.slider_music.get_value()
            self.slider_music.custom_label = f"Music Volume: {mus_val}%"
            self.sound_manager.set_music_volume(mus_val / 100.0)

            for slider in [self.slider_m, self.slider_n, self.slider_k, self.slider_p1, self.slider_p2, self.slider_time, self.slider_music]:
                slider.update(mouse_pos)
            self.btn_back.update(mouse_pos)
        elif self.state == "GAMEOVER":
             for btn in [self.btn_rematch, self.btn_menu]:
                btn.update(mouse_pos)

    def render(self):
        self.screen.fill(COLORS.BACKGROUND)
        if self.state == "MENU":
            self.render_menu()
        elif self.state == "SETTINGS":
            self.render_settings()
        elif self.state == "GAME":
            self.render_game()
        elif self.state == "GAMEOVER":
            self.render_game()
            self.render_overlay()
            self.render_gameover()
        pygame.display.flip()

    def render_menu(self):
        title = self.font_title.render(f"MNK GAME ({self.k}-in-row)", True, COLORS.PRIMARY)
        title_rect = title.get_rect(center=(SCREEN_WIDTH // 2, 150))
        self.screen.blit(title, title_rect)
        self.btn_play.draw(self.screen)
        self.btn_settings.draw(self.screen)
        self.btn_quit.draw(self.screen)

    def render_settings(self):
        title = self.font_title.render("SETTINGS", True, COLORS.ACCENT)
        title_rect = title.get_rect(center=(SCREEN_WIDTH // 2, 80))
        self.screen.blit(title, title_rect)
        
        self.slider_m.draw(self.screen)
        self.slider_n.draw(self.screen)
        self.slider_k.draw(self.screen)
        self.slider_p1.draw(self.screen)
        self.slider_p2.draw(self.screen)
        self.slider_time.draw(self.screen)
        self.slider_music.draw(self.screen)
        
        self.btn_back.draw(self.screen)

    def render_game(self):
        # Draw Timers
        self.draw_timer(20, 20, 1, self.timer_p1)
        self.draw_timer(SCREEN_WIDTH - 220, 20, 2, self.timer_p2)

        # Status Text (Centered Top)
        status_text = f"Player {self.turn}'s Turn"
        current_ai = self.ai_p1 if self.turn == 1 else self.ai_p2
        if current_ai: status_text = f"Player {self.turn} (AI) Thinking..."
        text_surf = self.font_ui.render(status_text, True, COLORS.TEXT)
        self.screen.blit(text_surf, (SCREEN_WIDTH//2 - text_surf.get_width()//2, 30))

        # Board Calculation
        cell_size = min(600 // self.m, 800 // self.n, 60)
        board_w = cell_size * self.n
        board_h = cell_size * self.m
        start_x = (SCREEN_WIDTH - board_w) // 2
        start_y = (SCREEN_HEIGHT - board_h) // 2 + 30
        
        pygame.draw.rect(self.screen, COLORS.SURFACE, (start_x - 10, start_y - 10, board_w + 20, board_h + 20), border_radius=10)

        for r in range(self.m):
            for c in range(self.n):
                rect = pygame.Rect(start_x + c * cell_size, start_y + r * cell_size, cell_size, cell_size)
                pygame.draw.rect(self.screen, COLORS.BACKGROUND, rect, 1)
                
                val = self.board.board[r][c]
                if val != 0:
                    center = rect.center
                    radius = cell_size // 3
                    if val == 1: # X
                        color = COLORS.PRIMARY
                        start_pos1 = (center[0] - radius, center[1] - radius)
                        end_pos1 = (center[0] + radius, center[1] + radius)
                        start_pos2 = (center[0] - radius, center[1] + radius)
                        end_pos2 = (center[0] + radius, center[1] - radius)
                        pygame.draw.line(self.screen, color, start_pos1, end_pos1, 4)
                        pygame.draw.line(self.screen, color, start_pos2, end_pos2, 4)
                    else: # O
                        color = COLORS.SECONDARY
                        pygame.draw.circle(self.screen, color, center, radius, 4)
                        
                if self.board.last_move == (r, c):
                    pygame.draw.rect(self.screen, COLORS.ACCENT, rect, 2)

    def draw_timer(self, x, y, player, time_left):
        minutes = int(max(0, time_left) // 60)
        seconds = int(max(0, time_left) % 60)
        time_str = f"{minutes:02}:{seconds:02}"
        color = COLORS.PRIMARY if player == 1 else COLORS.SECONDARY
        
        # Highlight if low time and active
        if time_left < 30 and self.turn == player:
            if int(time.time() * 2) % 2 == 0: # Flash
                color = (255, 50, 50)
                
        # Card background
        rect = pygame.Rect(x, y, 200, 60)
        pygame.draw.rect(self.screen, COLORS.SURFACE, rect, border_radius=8)
        pygame.draw.rect(self.screen, color, rect, 2, border_radius=8)
        
        name = "PLAYER 1" if player == 1 else "PLAYER 2"
        if player == 1 and self.ai_p1: name += " (AI)"
        if player == 2 and self.ai_p2: name += " (AI)"
        name_surf = self.font_ui.render(name, True, COLORS.TEXT)
        time_surf = self.font_timer.render(time_str, True, color)
        
        self.screen.blit(name_surf, (x + 10, y + 5))
        self.screen.blit(time_surf, (x + 10, y + 28))

    def render_overlay(self):
        s = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        s.set_alpha(150)
        s.fill((0,0,0))
        self.screen.blit(s, (0,0))

    def render_gameover(self):
        reason = self.win_reason if self.win_reason else ""
        if self.winner == 0:
            msg = "DRAW!"
            color = COLORS.TEXT
        elif self.winner == 1:
            msg = "PLAYER 1 WINS!"
            color = COLORS.PRIMARY
        else:
            msg = "PLAYER 2 WINS!"
            color = COLORS.SECONDARY
            
        text = self.font_title.render(msg, True, color)
        subtext = self.font_ui.render(reason, True, COLORS.TEXT)
        
        rect = text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 60))
        rect_sub = subtext.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
        
        # Background panel for text
        panel_rect = rect.union(rect_sub).inflate(60, 40)
        pygame.draw.rect(self.screen, COLORS.SURFACE, panel_rect, border_radius=10)
        pygame.draw.rect(self.screen, color, panel_rect, 2, border_radius=10)
        
        self.screen.blit(text, rect)
        self.screen.blit(subtext, rect_sub)
        
        self.btn_rematch.draw(self.screen)
        self.btn_menu.draw(self.screen)

if __name__ == "__main__":
    app = GameApp()
    app.run()
