"""
Battleship Game - GUI Version
A graphical implementation using Tkinter
"""

import tkinter as tk
from tkinter import messagebox, ttk
from random import randint, choice
import copy


class Ship:
    """Represents a ship on the board"""
    def __init__(self, name, size, color):
        self.name = name
        self.size = size
        self.color = color
        self.positions = []
        self.hits = []

    def is_sunk(self):
        """Check if all parts of the ship have been hit"""
        return len(self.hits) == self.size

    def add_hit(self, position):
        """Record a hit on this ship"""
        if position in self.positions and position not in self.hits:
            self.hits.append(position)
            return True
        return False


class Board:
    """Represents a game board"""
    def __init__(self, size=10):
        self.size = size
        self.grid = [["~"] * size for _ in range(size)]
        self.ships = []
        self.guesses = []
        self.hits = []
        self.misses = []

    def is_valid_position(self, row, col):
        """Check if position is within board bounds"""
        return 0 <= row < self.size and 0 <= col < self.size

    def can_place_ship(self, ship, start_row, start_col, orientation):
        """Check if a ship can be placed at the given position"""
        positions = []

        for i in range(ship.size):
            if orientation == "horizontal":
                row, col = start_row, start_col + i
            else:  # vertical
                row, col = start_row + i, start_col

            if not self.is_valid_position(row, col):
                return False, []

            if self.grid[row][col] == "S":
                return False, []

            positions.append((row, col))

        return True, positions

    def place_ship(self, ship, start_row, start_col, orientation):
        """Place a ship on the board"""
        can_place, positions = self.can_place_ship(ship, start_row, start_col, orientation)

        if can_place:
            ship.positions = positions
            for row, col in positions:
                self.grid[row][col] = "S"
            self.ships.append(ship)
            return True
        return False

    def place_ships_randomly(self, ships):
        """Randomly place all ships on the board"""
        for ship in ships:
            placed = False
            attempts = 0
            max_attempts = 100

            while not placed and attempts < max_attempts:
                start_row = randint(0, self.size - 1)
                start_col = randint(0, self.size - 1)
                orientation = choice(["horizontal", "vertical"])

                placed = self.place_ship(ship, start_row, start_col, orientation)
                attempts += 1

    def make_guess(self, row, col):
        """Make a guess at the given position"""
        if not self.is_valid_position(row, col):
            return "invalid", None

        if (row, col) in self.guesses:
            return "duplicate", None

        self.guesses.append((row, col))

        for ship in self.ships:
            if (row, col) in ship.positions:
                ship.add_hit((row, col))
                self.hits.append((row, col))
                self.grid[row][col] = "X"

                if ship.is_sunk():
                    return "sunk", ship
                return "hit", ship

        self.misses.append((row, col))
        self.grid[row][col] = "O"
        return "miss", None

    def all_ships_sunk(self):
        """Check if all ships have been sunk"""
        return all(ship.is_sunk() for ship in self.ships)


class AIPlayer:
    """AI opponent with hunting strategy"""
    def __init__(self, board_size):
        self.board_size = board_size
        self.hunt_queue = []
        self.mode = "hunt"

    def make_guess(self, opponent_board):
        """Make an intelligent guess"""
        if self.hunt_queue:
            self.mode = "target"
            row, col = self.hunt_queue.pop(0)
            if (row, col) not in opponent_board.guesses:
                return row, col

        self.mode = "hunt"
        while True:
            row = randint(0, self.board_size - 1)
            col = randint(0, self.board_size - 1)

            if (row, col) not in opponent_board.guesses:
                return row, col

    def process_result(self, result, row, col, ship=None):
        """Process the result and update strategy"""
        if result == "hit":
            adjacent = [
                (row - 1, col), (row + 1, col),
                (row, col - 1), (row, col + 1),
            ]

            for adj_row, adj_col in adjacent:
                if (0 <= adj_row < self.board_size and
                    0 <= adj_col < self.board_size):
                    if (adj_row, adj_col) not in self.hunt_queue:
                        self.hunt_queue.append((adj_row, adj_col))

        elif result == "sunk":
            self.hunt_queue = []
            self.mode = "hunt"


class BattleshipGUI:
    """Main GUI application"""
    def __init__(self, root):
        self.root = root
        self.root.title("Battleship Game")
        self.root.geometry("1200x700")
        self.root.resizable(False, False)

        # Game state
        self.board_size = 10
        self.player_board = None
        self.ai_board = None
        self.ai_player = None
        self.game_active = False
        self.player_turn = True

        # Colors
        self.colors = {
            'water': '#1E90FF',
            'ship': '#808080',
            'hit': '#FF4500',
            'miss': '#FFFFFF',
            'hover': '#87CEEB',
            'grid': '#000080'
        }

        # GUI elements
        self.ai_buttons = []
        self.player_buttons = []

        self.setup_menu_screen()

    def setup_menu_screen(self):
        """Setup the main menu"""
        # Clear root
        for widget in self.root.winfo_children():
            widget.destroy()

        menu_frame = tk.Frame(self.root, bg='#2C3E50')
        menu_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        title = tk.Label(menu_frame, text="BATTLESHIP",
                        font=('Arial', 48, 'bold'),
                        bg='#2C3E50', fg='white')
        title.pack(pady=50)

        # Board size selection
        size_frame = tk.Frame(menu_frame, bg='#2C3E50')
        size_frame.pack(pady=20)

        tk.Label(size_frame, text="Board Size:",
                font=('Arial', 16), bg='#2C3E50', fg='white').pack(side=tk.LEFT, padx=10)

        self.size_var = tk.IntVar(value=10)
        size_spinbox = tk.Spinbox(size_frame, from_=6, to=15,
                                  textvariable=self.size_var,
                                  font=('Arial', 14), width=5)
        size_spinbox.pack(side=tk.LEFT)

        # Game mode selection
        mode_frame = tk.Frame(menu_frame, bg='#2C3E50')
        mode_frame.pack(pady=20)

        tk.Label(mode_frame, text="Game Mode:",
                font=('Arial', 16), bg='#2C3E50', fg='white').pack(side=tk.LEFT, padx=10)

        self.mode_var = tk.StringVar(value="ai")
        tk.Radiobutton(mode_frame, text="vs AI", variable=self.mode_var,
                      value="ai", font=('Arial', 14),
                      bg='#2C3E50', fg='white', selectcolor='#34495E').pack(side=tk.LEFT, padx=10)
        tk.Radiobutton(mode_frame, text="Practice", variable=self.mode_var,
                      value="practice", font=('Arial', 14),
                      bg='#2C3E50', fg='white', selectcolor='#34495E').pack(side=tk.LEFT, padx=10)

        # Start button
        start_btn = tk.Button(menu_frame, text="START GAME",
                             command=self.start_game,
                             font=('Arial', 20, 'bold'),
                             bg='#27AE60', fg='white',
                             padx=40, pady=15,
                             cursor='hand2')
        start_btn.pack(pady=50)

        # Instructions
        instructions = """
        HOW TO PLAY:
        • Click on the enemy grid to fire
        • Blue = Water, White = Miss, Red = Hit
        • Sink all enemy ships to win!
        """
        tk.Label(menu_frame, text=instructions,
                font=('Arial', 12), bg='#2C3E50', fg='white',
                justify=tk.LEFT).pack(pady=20)

    def create_ships(self, board_size):
        """Create ship fleet"""
        if board_size <= 8:
            return [
                Ship("Destroyer", 2, "#8B4513"),
                Ship("Submarine", 3, "#4B0082"),
                Ship("Battleship", 4, "#006400"),
            ]
        else:
            return [
                Ship("Patrol Boat", 2, "#FF6347"),
                Ship("Destroyer", 3, "#8B4513"),
                Ship("Submarine", 3, "#4B0082"),
                Ship("Battleship", 4, "#006400"),
                Ship("Carrier", 5, "#B8860B"),
            ]

    def start_game(self):
        """Initialize and start the game"""
        self.board_size = self.size_var.get()
        self.game_mode = self.mode_var.get()

        # Create boards
        self.player_board = Board(self.board_size)
        self.ai_board = Board(self.board_size)
        self.ai_player = AIPlayer(self.board_size)

        # Place ships
        player_ships = self.create_ships(self.board_size)
        ai_ships = self.create_ships(self.board_size)

        self.player_board.place_ships_randomly(player_ships)
        self.ai_board.place_ships_randomly(ai_ships)

        self.game_active = True
        self.player_turn = True

        self.setup_game_screen()

    def setup_game_screen(self):
        """Setup the main game screen"""
        # Clear root
        for widget in self.root.winfo_children():
            widget.destroy()

        # Main container
        main_frame = tk.Frame(self.root, bg='#34495E')
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Top bar
        top_bar = tk.Frame(main_frame, bg='#2C3E50', height=60)
        top_bar.pack(fill=tk.X)
        top_bar.pack_propagate(False)

        tk.Label(top_bar, text="BATTLESHIP",
                font=('Arial', 24, 'bold'),
                bg='#2C3E50', fg='white').pack(side=tk.LEFT, padx=20)

        # Status label
        self.status_label = tk.Label(top_bar, text="Your Turn - Fire!",
                                     font=('Arial', 16),
                                     bg='#2C3E50', fg='#27AE60')
        self.status_label.pack(side=tk.LEFT, padx=20)

        # New game button
        tk.Button(top_bar, text="New Game", command=self.setup_menu_screen,
                 font=('Arial', 12), bg='#E74C3C', fg='white',
                 cursor='hand2').pack(side=tk.RIGHT, padx=20)

        # Boards container
        boards_frame = tk.Frame(main_frame, bg='#34495E')
        boards_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Enemy board (left)
        enemy_frame = tk.Frame(boards_frame, bg='#34495E')
        enemy_frame.pack(side=tk.LEFT, padx=10, fill=tk.BOTH, expand=True)

        tk.Label(enemy_frame, text="ENEMY WATERS",
                font=('Arial', 18, 'bold'),
                bg='#34495E', fg='white').pack(pady=10)

        self.ai_grid_frame = tk.Frame(enemy_frame, bg='#34495E')
        self.ai_grid_frame.pack()

        self.create_board_grid(self.ai_grid_frame, "ai")

        # Stats for enemy
        self.ai_stats_label = tk.Label(enemy_frame, text="",
                                       font=('Arial', 12),
                                       bg='#34495E', fg='white')
        self.ai_stats_label.pack(pady=10)

        # Player board (right)
        player_frame = tk.Frame(boards_frame, bg='#34495E')
        player_frame.pack(side=tk.RIGHT, padx=10, fill=tk.BOTH, expand=True)

        tk.Label(player_frame, text="YOUR FLEET",
                font=('Arial', 18, 'bold'),
                bg='#34495E', fg='white').pack(pady=10)

        self.player_grid_frame = tk.Frame(player_frame, bg='#34495E')
        self.player_grid_frame.pack()

        self.create_board_grid(self.player_grid_frame, "player")

        # Stats for player
        self.player_stats_label = tk.Label(player_frame, text="",
                                           font=('Arial', 12),
                                           bg='#34495E', fg='white')
        self.player_stats_label.pack(pady=10)

        # Legend
        legend_frame = tk.Frame(main_frame, bg='#2C3E50')
        legend_frame.pack(fill=tk.X, pady=10)

        legend_items = [
            ("Water", self.colors['water']),
            ("Ship", self.colors['ship']),
            ("Hit", self.colors['hit']),
            ("Miss", self.colors['miss'])
        ]

        for text, color in legend_items:
            item_frame = tk.Frame(legend_frame, bg='#2C3E50')
            item_frame.pack(side=tk.LEFT, padx=20)

            canvas = tk.Canvas(item_frame, width=20, height=20,
                             bg=color, highlightthickness=1,
                             highlightbackground='white')
            canvas.pack(side=tk.LEFT, padx=5)

            tk.Label(item_frame, text=text,
                    font=('Arial', 10),
                    bg='#2C3E50', fg='white').pack(side=tk.LEFT)

        self.update_stats()

    def create_board_grid(self, parent, board_type):
        """Create a clickable grid"""
        cell_size = min(40, 500 // self.board_size)

        # Column headers
        header_frame = tk.Frame(parent, bg='#34495E')
        header_frame.grid(row=0, column=0, columnspan=self.board_size + 1)

        tk.Label(header_frame, text="", width=2,
                bg='#34495E').grid(row=0, column=0)

        for col in range(self.board_size):
            tk.Label(header_frame, text=str(col),
                    font=('Arial', 10, 'bold'),
                    bg='#34495E', fg='white', width=2).grid(row=0, column=col + 1)

        # Create grid
        buttons = []
        for row in range(self.board_size):
            # Row header
            tk.Label(parent, text=str(row),
                    font=('Arial', 10, 'bold'),
                    bg='#34495E', fg='white', width=2).grid(row=row + 1, column=0)

            button_row = []
            for col in range(self.board_size):
                btn = tk.Button(parent, width=cell_size // 10, height=cell_size // 20,
                              bg=self.colors['water'],
                              relief=tk.RAISED, bd=2)

                if board_type == "ai":
                    btn.config(command=lambda r=row, c=col: self.on_ai_cell_click(r, c))
                    btn.bind('<Enter>', lambda e, b=btn: self.on_hover(e, b, board_type))
                    btn.bind('<Leave>', lambda e, b=btn, r=row, c=col: self.on_leave(e, b, r, c, board_type))

                btn.grid(row=row + 1, column=col + 1, padx=1, pady=1)
                button_row.append(btn)

            buttons.append(button_row)

        if board_type == "ai":
            self.ai_buttons = buttons
        else:
            self.player_buttons = buttons
            self.update_player_board()

    def update_player_board(self):
        """Update player board to show ships"""
        for row in range(self.board_size):
            for col in range(self.board_size):
                if self.player_board.grid[row][col] == "S":
                    self.player_buttons[row][col].config(bg=self.colors['ship'])

    def on_hover(self, event, button, board_type):
        """Handle mouse hover"""
        if board_type == "ai" and self.game_active and self.player_turn:
            current_bg = button.cget('bg')
            if current_bg == self.colors['water']:
                button.config(bg=self.colors['hover'])

    def on_leave(self, event, button, row, col, board_type):
        """Handle mouse leave"""
        if board_type == "ai":
            if (row, col) not in self.ai_board.guesses:
                button.config(bg=self.colors['water'])

    def on_ai_cell_click(self, row, col):
        """Handle click on AI board"""
        if not self.game_active or not self.player_turn:
            return

        # Make guess
        result, ship = self.ai_board.make_guess(row, col)

        if result == "invalid" or result == "duplicate":
            return

        # Update button
        if result == "hit" or result == "sunk":
            self.ai_buttons[row][col].config(bg=self.colors['hit'])
            if result == "sunk":
                self.status_label.config(text=f"You sunk the {ship.name}!", fg='#F39C12')
                self.root.after(1500, self.check_game_over)
            else:
                self.status_label.config(text="Hit!", fg='#27AE60')
        else:
            self.ai_buttons[row][col].config(bg=self.colors['miss'])
            self.status_label.config(text="Miss!", fg='#E74C3C')

        self.update_stats()

        # Check win
        if self.ai_board.all_ships_sunk():
            self.root.after(1500, self.player_wins)
            return

        # AI turn
        if self.game_mode == "ai":
            self.player_turn = False
            self.root.after(1500, self.ai_turn)

    def ai_turn(self):
        """Execute AI turn"""
        if not self.game_active:
            return

        self.status_label.config(text="AI is thinking...", fg='#F39C12')
        self.root.update()

        # Make AI guess
        ai_row, ai_col = self.ai_player.make_guess(self.player_board)
        ai_result, ai_ship = self.player_board.make_guess(ai_row, ai_col)

        self.ai_player.process_result(ai_result, ai_row, ai_col, ai_ship)

        # Update player board
        if ai_result == "hit" or ai_result == "sunk":
            self.player_buttons[ai_row][ai_col].config(bg=self.colors['hit'])
            if ai_result == "sunk":
                self.status_label.config(text=f"AI sunk your {ai_ship.name}!", fg='#E74C3C')
            else:
                self.status_label.config(text="AI hit your ship!", fg='#E74C3C')
        else:
            self.player_buttons[ai_row][ai_col].config(bg=self.colors['miss'])
            self.status_label.config(text="AI missed!", fg='#27AE60')

        self.update_stats()

        # Check if AI won
        if self.player_board.all_ships_sunk():
            self.root.after(1500, self.ai_wins)
            return

        # Back to player turn
        self.root.after(1500, self.reset_turn)

    def reset_turn(self):
        """Reset to player turn"""
        self.player_turn = True
        self.status_label.config(text="Your Turn - Fire!", fg='#27AE60')

    def check_game_over(self):
        """Check if game is over"""
        if self.ai_board.all_ships_sunk():
            self.player_wins()

    def player_wins(self):
        """Handle player victory"""
        self.game_active = False
        stats = self.ai_board
        accuracy = (len(stats.hits) / len(stats.guesses) * 100) if stats.guesses else 0

        message = f"""
VICTORY!

You sunk all enemy ships!

Final Stats:
Total Shots: {len(stats.guesses)}
Hits: {len(stats.hits)}
Misses: {len(stats.misses)}
Accuracy: {accuracy:.1f}%
        """

        messagebox.showinfo("You Win!", message)
        self.status_label.config(text="YOU WIN!", fg='#27AE60')

    def ai_wins(self):
        """Handle AI victory"""
        self.game_active = False
        message = "Game Over!\n\nThe AI sunk all your ships!"
        messagebox.showinfo("Game Over", message)
        self.status_label.config(text="AI WINS!", fg='#E74C3C')

    def update_stats(self):
        """Update statistics display"""
        # AI board stats (your performance)
        ai_stats = self.ai_board
        ai_accuracy = (len(ai_stats.hits) / len(ai_stats.guesses) * 100) if ai_stats.guesses else 0
        ships_sunk = sum(1 for ship in ai_stats.ships if ship.is_sunk())

        ai_text = f"""
Shots: {len(ai_stats.guesses)} | Hits: {len(ai_stats.hits)} | Misses: {len(ai_stats.misses)}
Accuracy: {ai_accuracy:.1f}%
Enemy Ships Sunk: {ships_sunk}/{len(ai_stats.ships)}
        """
        self.ai_stats_label.config(text=ai_text)

        # Player board stats (AI performance)
        if self.game_mode == "ai":
            player_stats = self.player_board
            player_accuracy = (len(player_stats.hits) / len(player_stats.guesses) * 100) if player_stats.guesses else 0
            player_ships_sunk = sum(1 for ship in player_stats.ships if ship.is_sunk())

            player_text = f"""
AI Shots: {len(player_stats.guesses)} | Hits: {len(player_stats.hits)} | Misses: {len(player_stats.misses)}
AI Accuracy: {player_accuracy:.1f}%
Your Ships Sunk: {player_ships_sunk}/{len(player_stats.ships)}
            """
            self.player_stats_label.config(text=player_text)


def main():
    """Main entry point"""
    root = tk.Tk()
    app = BattleshipGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
