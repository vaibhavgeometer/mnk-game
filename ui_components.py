import pygame

class COLORS:
    BACKGROUND = (30, 30, 40)
    SURFACE = (50, 50, 65)
    PRIMARY = (100, 180, 255)    # Soft Blue
    SECONDARY = (255, 100, 100)  # Soft Red
    ACCENT = (100, 255, 150)     # Soft Green
    TEXT = (230, 230, 230)
    TEXT_DIM = (150, 150, 160)
    HIGHLIGHT = (70, 70, 90)
    SHADOW = (20, 20, 25)

class UIElement:
    def __init__(self, x, y, w, h):
        self.rect = pygame.Rect(x, y, w, h)
        self.hovered = False
    
    def update(self, mouse_pos):
        self.hovered = self.rect.collidepoint(mouse_pos)
        return self.hovered

    def draw(self, surface):
        pass

class Button(UIElement):
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
