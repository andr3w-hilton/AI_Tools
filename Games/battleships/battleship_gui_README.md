# Battleship Game - GUI Version

A beautiful graphical implementation of Battleship using Python's Tkinter library.

## Features

### Graphical Interface
- **Clean, Modern Design** - Professional UI with color-coded elements
- **Clickable Grid** - Simply click cells to fire
- **Visual Feedback** - Hover effects and instant visual updates
- **Two Boards** - Enemy waters on left, your fleet on right
- **Real-time Stats** - Live tracking of shots, hits, misses, and accuracy
- **Status Updates** - Clear messages for hits, misses, and sunk ships

### Game Features
- **Customizable Board Size** - Choose from 6x6 to 15x15
- **Two Game Modes**:
  - **vs AI** - Play against intelligent computer opponent
  - **Practice** - Play solo to learn the game
- **Smart AI** - Uses hunt and target strategy
- **Ship Fleet** - Multiple ships of different sizes
- **Win Detection** - Automatic game over with statistics

## Visual Elements

### Color Scheme
- **Blue** - Water (unexplored cells)
- **Gray** - Your ships
- **Red** - Hits
- **White** - Misses
- **Light Blue** - Hover effect

### Layout
```
+--------------------------------------------------+
|  BATTLESHIP          Status: Your Turn    [New]  |
+--------------------------------------------------+
|                                                   |
|   ENEMY WATERS              YOUR FLEET           |
|   [10x10 Grid]              [10x10 Grid]         |
|                                                   |
|   Stats Display             Stats Display        |
|                                                   |
+--------------------------------------------------+
|   Legend: Water | Ship | Hit | Miss              |
+--------------------------------------------------+
```

## How to Run

### Requirements
- Python 3.6 or higher
- Tkinter (usually included with Python)

### Check if Tkinter is installed:
```bash
python -m tkinter
```
This should open a small test window. If not, you need to install Tkinter.

### Run the game:
```bash
python battle_ships_gui.py
```

Or with specific Python version:
```bash
py -3.12 battle_ships_gui.py
```

## How to Play

### Starting a Game

1. **Launch the game** - Run the Python script
2. **Configure settings**:
   - Select board size (6-15)
   - Choose game mode (vs AI or Practice)
3. **Click START GAME**

### Gameplay

1. **Click on enemy grid** (left side) to fire at that location
2. **Watch for results**:
   - Red cell = Hit
   - White cell = Miss
   - Status message shows what happened
3. **Your ships** are visible on the right (gray cells)
4. **AI automatically fires** after your turn (in vs AI mode)
5. **Win by sinking** all enemy ships before AI sinks yours

### During Game

- **Status bar** shows whose turn it is and last action
- **Stats** update in real-time below each board
- **Hover over cells** to see targeting highlight
- **Click "New Game"** to return to menu

## Game Modes

### vs AI Mode
- Play against intelligent computer opponent
- AI uses hunting strategy to find and sink your ships
- Both boards are active
- Take turns with AI

### Practice Mode
- Play without AI opponent
- Focus on finding enemy ships
- Learn ship placement patterns
- No pressure gameplay

## Ship Fleet

### Standard Board (9+)
- Patrol Boat: 2 cells (Red)
- Destroyer: 3 cells (Brown)
- Submarine: 3 cells (Purple)
- Battleship: 4 cells (Green)
- Carrier: 5 cells (Gold)

### Small Board (6-8)
- Destroyer: 2 cells
- Submarine: 3 cells
- Battleship: 4 cells

## Statistics Tracked

For each board, the game tracks:
- **Total Shots** - Number of guesses made
- **Hits** - Successful shots on ships
- **Misses** - Shots that hit water
- **Accuracy** - Hit percentage
- **Ships Sunk** - Number of ships destroyed vs total

## Tips for Playing

1. **Don't cluster shots** - Spread out initial guesses
2. **Follow up hits** - When you hit, check adjacent cells
3. **Watch patterns** - AI might reveal patterns in its strategy
4. **Count ships** - Track which ships you've sunk
5. **Use corners** - Ships often spawn near edges

## Advantages Over Terminal Version

| Feature | Terminal | GUI |
|---------|----------|-----|
| Visual Appeal | Basic text | Colorful graphics |
| Input Method | Type coordinates | Click cells |
| Board View | Text grid | Visual grid |
| Feedback | Text messages | Colors + messages |
| Ease of Use | Requires typing | Point and click |
| Learning Curve | Steeper | Gentle |
| Mobile Friendly | No | No (desktop app) |
| Mistakes | Easy to mistype | Hard to misclick |

## Keyboard Shortcuts

- **Click** - Fire at cell
- **New Game** button - Return to menu
- **X** (close window) - Exit game

## Technical Details

### Architecture

**BattleshipGUI Class** - Main application controller
- Manages game state
- Handles UI creation and updates
- Processes player and AI turns

**Board Class** - Game logic
- Ship placement
- Guess validation
- Hit detection
- Statistics tracking

**AIPlayer Class** - Computer opponent
- Hunt mode (random search)
- Target mode (systematic attack)
- Result processing

**Ship Class** - Ship management
- Position tracking
- Hit tracking
- Sunk detection

### Performance

- **Lightweight** - Uses only Python standard library
- **Responsive** - Instant click feedback
- **Smooth** - No lag or delays
- **Scalable** - Handles board sizes up to 15x15

## Troubleshooting

### Tkinter Not Found
**Error**: `ImportError: No module named tkinter`

**Solution**: Install tkinter
- **Ubuntu/Debian**: `sudo apt-get install python3-tk`
- **Fedora**: `sudo dnf install python3-tkinter`
- **macOS**: Included with Python
- **Windows**: Included with Python

### Window Too Large
If the game window doesn't fit your screen:
- Use smaller board size (6-8)
- The cell size automatically adjusts

### Slow Performance
- Close other applications
- Use smaller board size
- Ensure Python is up to date

## Future Enhancements

Possible additions:
- **Manual ship placement** - Drag and drop ships before game
- **Animations** - Explosions, water splashes
- **Sound effects** - Shots, hits, explosions
- **Themes** - Different color schemes
- **Difficulty levels** - Easy, Medium, Hard AI
- **Network multiplayer** - Play against friends online
- **Save/Load** - Resume games later
- **High scores** - Track best accuracy/fewest shots
- **Ship placement preview** - See ships before finalizing
- **Touch screen support** - Better mobile/tablet experience

## File Comparison

### Three Versions Available:

1. **battle_ships.py** - Original simple version
   - Single ship
   - Terminal based
   - 4 turns
   - Basic features

2. **battle_ships_enhanced.py** - Complete terminal version
   - Multiple ships
   - Full features
   - AI opponent
   - Statistics

3. **battle_ships_gui.py** - Graphical version (this file)
   - All enhanced features
   - Beautiful GUI
   - Point and click
   - Visual feedback

## Requirements

- Python 3.6+
- Tkinter (standard library)
- No pip installs needed!

## Running on Different Systems

### Windows
```bash
python battle_ships_gui.py
```

### macOS/Linux
```bash
python3 battle_ships_gui.py
```

### If you have multiple Python versions
```bash
py -3.12 battle_ships_gui.py
```

## License

Educational project - free to use and modify!

## Credits

Built with:
- Python 3
- Tkinter GUI framework
- Love for classic games

Enjoy playing Battleship! 🚢💥
