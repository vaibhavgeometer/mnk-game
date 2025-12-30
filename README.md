# MNK Game Deluxe

> **A Modern, Feature-Rich Implementation of the Generalized Tic-Tac-Toe (m,n,k-game)**

![Game UI](https://img.shields.io/badge/Status-Active-success) ![Python](https://img.shields.io/badge/Made%20with-Python-blue) ![Pygame](https://img.shields.io/badge/Library-Pygame-red)

A highly customizable, competitive table-top game built from scratch in Python. It features a completely redesigned, **Lichess-inspired dark UI**, a powerful AI engine, and full game history tracking.

---

## ✨ Features

### 🎨 **New "Pro" User Interface**

- **Dark Mode Aesthetic**: A sleek, distraction-free dark theme inspired by professional chess platforms.
- **3-Panel Layout**:
  - **Left**: Game Info & Chat placeholder.
  - **Center**: The main board area.
  - **Right**: Player cards, live timers with active indicators, and a scrollable move history list.
- **Visual Cues**: Last move highlighting, winning line gold glow, and interactive ghost pieces.

### 🎮 **Customizable Gameplay**

- **Board Configuration**:
  - **Rows ($M$) & Cols ($N$)**: Adjust grid size from **1x1** up to **32x32**.
  - **Win Condition ($K$)**: Set the target line length from **1** to **7** (and beyond).
- **Time Controls**: Full support for Fischer timing (Initial Time + Increment per move).
- **Default Standards**: Pre-configured defaults for competitive play (15x15 board, 5-in-a-row, 3 min + 2 sec).

### 🤖 **Adjustable AI Opponents**

Challenge the computer with 4 distinct difficulty levels:

- **Level 1 (Beginner)**: Elo ~600. Good for learning the rules.
- **Level 2 (Intermediate)**: Elo ~1200. Avoids simple blunders (Default opponent).
- **Level 3 (Advanced)**: Elo ~1800. Strong tactical play.
- **Level 4 (Master)**: Max strength using deep search, Alpha-Beta pruning, and Zobrist Hashing.

### 📜 **Robust System**

- **Single-File Architecture**: The entire game engine, UI, and AI logic are consolidated into `main.py` for easy portability.
- **Game History**: Every match is logged to `game_history.json`.
- **Elo Rating System**: Tracks AI performance over time (stored in `elo_ratings.json`).

---

## 🚀 Installation & Setup

1. **Clone the repository**:

   ```bash
   git clone <repository-url>
   cd mnk-game
   ```

2. **Install Dependencies**:
   You only need `pygame`. `numpy` is optional but recommended for better sound generation.

   ```bash
   pip install pygame numpy
   ```

3. **Run the Game**:
   ```bash
   python main.py
   ```

---

## 🕹️ Controls

- **Mouse**: Entirely mouse-driven.
  - **Click**: Place a piece, navigate menus, adjust sliders.
  - **Drag**: Adjust sliders in the Settings menu.
- **Keyboard**:
  - **F11**: Toggle Fullscreen Mode.

---

## 🛠️ Technology Stack

- **Language**: Python 3.12+
- **Rendering**: Pygame Community Edition (CE)
- **Audio**: Procedural sound generation (Sine/Square waves) via Numpy/Pygame.

---

## 👥 Credits

**Lead Developer**: Vaibhav  
**AI Collaborator**: Antigravity (Google DeepMind)
