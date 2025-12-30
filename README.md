# MNK Game Deluxe

> **A Professional, Feature-Rich Implementation of the Generalized Tic-Tac-Toe (m,n,k-game)**

![Game UI](https://img.shields.io/badge/Status-Active-success) ![Python](https://img.shields.io/badge/Made%20with-Python-blue) ![Pygame](https://img.shields.io/badge/Library-Pygame--CE-red)

A highly customizable, competitive table-top game built with Python and Pygame. It features a premium **Lichess-inspired dark UI**, a powerful AI engine with 8 difficulty levels, dynamic Elo rating tracking, and a robust game history system.

---

## ✨ Features

### 🎨 **Professional User Interface**

- **Lichess-Inspired Design**: A sleek, distraction-free dark theme optimized for clarity and focus.
- **3-Panel Strategic Layout**:
  - **Left Panel**: Game metadata, current grid settings, and quick-access menu.
  - **Center Panel**: The responsive game board with high-visibility pieces and winning line animations.
  - **Right Panel**: Active player cards with live Fischer timers, Elo indicators, and a scrollable move list.
- **Visual Excellence**: Subtle micro-animations, last-move highlighting, and interactive "ghost" pieces for effortless placement.

### 🤖 **8-Level AI Engine**

Challenge a sophisticated AI using Alpha-Beta pruning, Zobrist Hashing, and Iterative Deepening. The AI strength scales linearly:

- **Level 1 (Novice)**: ~300 Elo. Basic play, perfect for beginners.
- **Levels 2-4 (Intermediate)**: ~600 to ~1200 Elo. Avoids obvious blunders and starts looking ahead.
- **Levels 5-7 (Advanced)**: ~1500 to ~2100 Elo. Strong tactical awareness and positional play.
- **Level 8 (Master)**: Max Strength. Deep search depth designed to challenge even experienced players.

### 📈 **Elo & History System**

- **Bot Ratings Graph**: Track the performance of all 8 AI levels via an integrated line graph.
- **Dynamic Elo Tracking**: Watch AI ratings fluctuate based on their performance against each other.
- **Move History**: Complete record of every match stored in `game_history.json`, with a scrollable in-game move list.
- **Data Management**: Full control to reset history or recalibrate bot ratings at any time.

### ⚙️ **Customizable Gameplay**

- **Board Configuration**:
  - **Grid Size ($M \times N$)**: Fully adjustable from **3x3** to **32x32**.
  - **Win Condition ($K$)**: Set the target line length (e.g., Gomoku is 15x15, 5).
- **Time Controls**: Professional Fischer timing (Initial Time + Increment per move).
- **Portability**: Single-file architecture (`main.py`) makes it easy to run and share.

---

## 🚀 Installation & Setup

1. **Clone the repository**:

   ```bash
   git clone https://github.com/vaibhavgeometer/mnk-game.git
   cd mnk-game
   ```

2. **Install Dependencies**:

   While `pygame` is the only requirement, `numpy` is recommended for high-quality procedurally generated audio.

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
  - **Click**: Place pieces, navigate menus, and adjust settings.
  - **Sliders**: Drag to fine-tune board size, win condition, and time controls.
- **Keyboard**:
  - **F11**: Toggle Fullscreen Mode.

---

## 🛠️ Technology Stack

- **Language**: Python 3.12+
- **Framework**: Pygame Community Edition (CE)
- **Audio**: Procedural Sine/Square wave generation via Numpy.
- **Data**: JSON-based persistent storage for ratings and history.

---

## 👥 Credits

**Lead Developer**: Vaibhav (@vaibhavgeometer)  
**AI Technical Partner**: Antigravity (Google DeepMind)
