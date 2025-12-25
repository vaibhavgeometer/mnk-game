# MNK Game Deluxe

A highly customizable, advanced version of the m,n,k-game (generalized Tic Tac Toe) built with Python and Pygame. Challenge yourself against sophisticated AI opponents or play with friends.

## Features

### 🎮 Gameplay Customization

- **Flexible Board Rules**: Customize board dimensions ($M \times N$) and win condition ($K$-in-a-row).
- **Match Configurations**: Adjustable time limits and **time increments** (Fischer timing).
- **Game Modes**: Support for Human vs. Human, Human vs. AI, and AI vs. AI battles.

### 🤖 Advanced AI

- **8 Difficulty Levels**: From random moves (Elo 0) to an unbeatable deep search engine (Elo 2500+).
- **Modern Algorithms**: Powered by Minimax with Alpha-Beta Pruning, Iterative Deepening, and Zobrist Hashing.
- **Transposition Table**: Efficiently caches board states to speed up decision-making.
- **Opening Book**: Includes optimized opening moves for common board sizes (e.g., Gomoku, Tic-Tac-Toe).

### 🎨 Modern Experience

- **Sleek UI**: Clean interface with interactive sliders, real-time timer updates, and dynamic visuals.
- **Audio Experience**: Procedural sound generation for game events (moves, wins) and background music support.
- **Robustness**: Optimized for performance and stability.

## Installation

1. **Clone the repository**
2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

   _(Note: `numpy` is recommended for high-quality audio generation, but the game works without it.)_

## How to Run

Launch the game using Python:

```bash
python main.py
```

## Controls

- **Mouse Left Click**: Place pieces, select menu options, and adjust sliders.
- **Settings Menu**:
  - **Rows/Cols**: Adjust board size (3-20).
  - **Win (K)**: Set number of connected pieces needed to win.
  - **AI Levels**: Set strength for P1 and P2 separately (Level 0 = Human).
  - **Time/Increment**: Configure match timing.
  - **Music**: Adjust volume.

## Credits

Developed by **Vaibhav**.
Co-developed with **Antigravity**, an advanced AI coding assistant by Google Deepmind.
