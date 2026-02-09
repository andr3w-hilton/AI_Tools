# Enhanced Battleship Game

A complete implementation of the classic Battleship game with AI opponent, score tracking, and customizable features.

## Features

### All TODO Items Completed

1. **Score Counter** - Tracks hits, misses, total guesses, and accuracy percentage
2. **Multiple Ships** - Fleet includes 3-5 ships of varying sizes depending on board size
3. **AI Opponent** - Smart AI that uses hunting and targeting strategies
4. **Hit/Miss Tracking** - Visual board shows all previous guesses and their results
5. **Input Validation** - Comprehensive validation for all user inputs
6. **Customizable Board** - User-defined board size (6x6 to 15x15)

### Additional Features

- Two game modes: Single Player (vs AI) or Practice Mode
- Intelligent AI that adapts its strategy when it gets a hit
- Clear visual board display with coordinates
- Real-time statistics tracking
- Ship sinking detection
- Duplicate guess prevention
- Screen clearing for better gameplay experience
- Play again option

## How to Play

### Starting the Game

```bash
python battle_ships_enhanced.py
```

### Game Setup

1. Choose board size (6-15, default is 10)
2. Select game mode:
   - Single Player: Play against the AI
   - Practice Mode: Play without AI opponent

### Gameplay

- Enter row and column coordinates to make a guess
- Legend:
  - `~` = Water (unexplored)
  - `O` = Miss
  - `X` = Hit
  - `S` = Ship (only visible on your own board)

### Controls

- Enter coordinates when prompted (e.g., row: 3, col: 5)
- Enter 'q' to quit the game
- Press Enter to continue between turns

## Ship Fleet

### Standard Board (9+)
- Patrol Boat: 2 cells
- Destroyer: 3 cells
- Submarine: 3 cells
- Battleship: 4 cells
- Carrier: 5 cells

### Small Board (6-8)
- Destroyer: 2 cells
- Submarine: 3 cells
- Battleship: 4 cells

## AI Strategy

The AI uses a two-mode strategy:

1. **Hunt Mode**: Randomly searches the board for ships
2. **Target Mode**: After getting a hit, systematically checks adjacent cells

This creates a challenging opponent that feels intelligent without being unbeatable.

## Game Statistics

The game tracks and displays:
- Total guesses made
- Number of hits
- Number of misses
- Accuracy percentage
- Ships remaining vs total ships

## Code Structure

### Classes

**Ship**
- Represents a single ship
- Tracks position, hits, and sinking status

**Board**
- Manages the game grid
- Handles ship placement and guess validation
- Tracks all game statistics

**AIPlayer**
- Implements intelligent guessing strategy
- Maintains hunt queue for targeting mode
- Processes results to improve future guesses

**BattleshipGame**
- Main game controller
- Manages game flow and turn-based play
- Handles user interface and display

## Improvements Over Original

### From Original (battle_ships.py)
- Single ship
- 5x5 fixed board
- 4 turns limit
- Basic functionality
- No AI opponent
- Minimal feedback

### Enhanced Version
- Multiple ships (3-5 depending on board size)
- Customizable board (6x6 to 15x15)
- No turn limit (play until win/lose)
- Full statistics tracking
- Smart AI opponent
- Rich visual feedback
- Input validation
- Professional code structure

## Example Gameplay

```
OPPONENT'S BOARD:
  0 1 2 3 4 5 6 7 8 9
0 ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
1 ~ ~ O ~ ~ ~ ~ ~ ~ ~
2 ~ ~ ~ X ~ ~ ~ ~ ~ ~
3 ~ ~ ~ X ~ ~ ~ ~ ~ ~
4 ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
5 ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
6 ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
7 ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
8 ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
9 ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

Your Stats: Guesses: 3 | Hits: 2 | Misses: 1 | Accuracy: 66.7%
Enemy Ships Remaining: 4/5
```

## Future Enhancements

Possible additions:
- Manual ship placement option
- Difficulty levels for AI
- Reinforcement learning AI (train model over many games)
- Save/load game state
- Multiplayer over network
- Sound effects
- GUI version with graphics
- Tournament mode (best of X games)
- Advanced AI strategies (probability heat maps)
- Different ship configurations

## Requirements

- Python 3.6+
- No external dependencies (uses only standard library)

## File Structure

- `battle_ships.py` - Original simple version
- `battle_ships_enhanced.py` - Complete enhanced version
- `battleship_README.md` - This documentation

## Tips for Winning

1. **Spread out your guesses** - Don't cluster guesses in one area
2. **Use a pattern** - Try checkerboard pattern to find ships faster
3. **Follow hits** - Once you hit, check all adjacent cells
4. **Track ship sizes** - Remember which ships you've sunk
5. **Corner to corner** - Ships can't be placed diagonally

## Known Limitations

- AI uses basic strategy (no probability analysis)
- Ships are placed randomly (not strategically)
- No multiplayer over network
- Console-based interface only

## License

Educational project - feel free to modify and extend!
