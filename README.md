# MNK Game Deluxe

> ⚠️ **Disclaimer:** There are a lot of bugs in the game right now. They will be fixed later.

A highly customizable, advanced version of the m,n,k-game (generalized Tic Tac Toe) built with Python and Pygame. Challenge yourself against sophisticated AI opponents or play with friends.

## Features

### 🎮 Gameplay Customization

- **Flexible Board Rules**: Customize board dimensions ($M \times N$) and win condition ($K$-in-a-row).
- **Match Configurations**: Adjustable time limits and **time increments** (Fischer timing).
- **Game Modes**: Support for Human vs. Human, Human vs. AI, and AI vs. AI battles.

### 🤖 Advanced AI

- **4 Difficulty Levels**:
    - **Level 1**: Elo ~600 (Beginner)
    - **Level 2**: Elo ~1200 (Intermediate)
    - **Level 3**: Elo ~1800 (Advanced)
    - **Level 4**: Max Strength (Deep Search)
- **Modern Algorithms**: Powered by Minimax with Alpha-Beta Pruning, Iterative Deepening, and Zobrist Hashing.
- **Transposition Table**: Efficiently caches board states to speed up decision-making.
- **Opening Book**: Includes optimized opening moves for common board sizes.

### 📜 Game History

- **Match Tracking**: Automatically records every game played.
- **Move Logs**: Detailed move-by-move history with Algebraic Notation (e.g., A1, B3).
- **Review**: View past results including winner, time played, and board size.

### 🎨 Modern Experience

- **Sleek UI**: Clean interface with interactive sliders, real-time timer updates, and dynamic visuals.
- **Procedural Audio**: Real-time sound generation for game events (moves, wins).
- **Robustness**: Optimized for performance.

## Installation

1. **Clone the repository**
2. **Install dependencies**:

   ```bash
   pip install pygame numpy
   ```

   _(Note: `numpy` is recommended for high-quality audio generation, but the game works without it.)_

## How to Run

Launch the game using Python:

```bash
python main.py
```

## Controls

- **Mouse Left Click**: Place pieces, select menu options, and adjust sliders.
- **Menu Options**:
  - **Play**: Start a single game.
  - **History**: View past game records.
  - **Settings**: Configure game rules.
- **Settings**:
  - **Rows/Cols**: Adjust board size (3-32).
  - **Win (K)**: Set number of connected pieces needed to win (up to 32).
  - **AI Levels**: Set strength for P1 and P2 separately (Level 0 = Human).
  - **Time/Increment**: Configure match timing.

## Credits

Developed by **Vaibhav**.
Co-developed with **Antigravity**, an advanced AI coding assistant by Google Deepmind.
