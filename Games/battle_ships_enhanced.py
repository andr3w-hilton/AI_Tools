"""
Enhanced Battleships Game
Features:
- Score tracking
- Multiple ships of different sizes
- AI opponent with basic strategy
- Customizable board size
- Input validation
- Hit/miss tracking
"""

from random import randint, choice
import os
import sys


class Ship:
    """Represents a ship on the board"""
    def __init__(self, name, size):
        self.name = name
        self.size = size
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

    def print_board(self, hide_ships=False):
        """Print the board with coordinates"""
        # Print column numbers
        print("  ", end="")
        for i in range(self.size):
            print(f"{i} ", end="")
        print()

        # Print rows with row numbers
        for i, row in enumerate(self.grid):
            print(f"{i} ", end="")
            for cell in row:
                if hide_ships and cell == "S":
                    print("~ ", end="")  # Hide ships from opponent
                else:
                    print(f"{cell} ", end="")
            print()

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

            # Check if position is valid
            if not self.is_valid_position(row, col):
                return False, []

            # Check if position is already occupied
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

            if not placed:
                print(f"Warning: Could not place {ship.name}")

    def make_guess(self, row, col):
        """Make a guess at the given position"""
        if not self.is_valid_position(row, col):
            return "invalid", None

        if (row, col) in self.guesses:
            return "duplicate", None

        self.guesses.append((row, col))

        # Check if hit
        for ship in self.ships:
            if (row, col) in ship.positions:
                ship.add_hit((row, col))
                self.hits.append((row, col))
                self.grid[row][col] = "X"

                if ship.is_sunk():
                    return "sunk", ship
                return "hit", ship

        # Miss
        self.misses.append((row, col))
        self.grid[row][col] = "O"
        return "miss", None

    def all_ships_sunk(self):
        """Check if all ships have been sunk"""
        return all(ship.is_sunk() for ship in self.ships)

    def get_stats(self):
        """Get game statistics"""
        total_ship_cells = sum(ship.size for ship in self.ships)
        accuracy = (len(self.hits) / len(self.guesses) * 100) if self.guesses else 0

        return {
            "total_guesses": len(self.guesses),
            "hits": len(self.hits),
            "misses": len(self.misses),
            "accuracy": accuracy,
            "ships_remaining": sum(1 for ship in self.ships if not ship.is_sunk()),
            "total_ships": len(self.ships)
        }


class AIPlayer:
    """AI opponent with basic hunting strategy"""
    def __init__(self, board_size):
        self.board_size = board_size
        self.last_hit = None
        self.hunt_queue = []  # Positions to check after a hit
        self.mode = "hunt"  # "hunt" or "target"

    def make_guess(self, opponent_board):
        """Make an intelligent guess"""
        # If we have positions to target, use those first
        if self.hunt_queue:
            self.mode = "target"
            row, col = self.hunt_queue.pop(0)
            return row, col

        # Otherwise, make a random guess
        self.mode = "hunt"
        while True:
            row = randint(0, self.board_size - 1)
            col = randint(0, self.board_size - 1)

            if (row, col) not in opponent_board.guesses:
                return row, col

    def process_result(self, result, row, col, ship=None):
        """Process the result of the last guess and update strategy"""
        if result == "hit":
            self.last_hit = (row, col)
            # Add adjacent cells to hunt queue
            adjacent = [
                (row - 1, col),  # up
                (row + 1, col),  # down
                (row, col - 1),  # left
                (row, col + 1),  # right
            ]

            for adj_row, adj_col in adjacent:
                if (0 <= adj_row < self.board_size and
                    0 <= adj_col < self.board_size and
                    (adj_row, adj_col) not in self.hunt_queue):
                    self.hunt_queue.append((adj_row, adj_col))

        elif result == "sunk":
            # Clear hunt queue when ship is sunk
            self.hunt_queue = []
            self.last_hit = None
            self.mode = "hunt"


class BattleshipGame:
    """Main game controller"""
    def __init__(self):
        self.board_size = 10
        self.player_board = None
        self.ai_board = None
        self.ai_player = None
        self.game_mode = None

    def clear_screen(self):
        """Clear the console screen"""
        os.system('cls' if os.name == 'nt' else 'clear')

    def get_board_size(self):
        """Get board size from user"""
        while True:
            try:
                size = input("Enter board size (6-15, default 10): ").strip()
                if size == "":
                    return 10

                size = int(size)
                if 6 <= size <= 15:
                    return size
                else:
                    print("Please enter a size between 6 and 15")
            except ValueError:
                print("Please enter a valid number")

    def create_ships(self, board_size):
        """Create ship fleet based on board size"""
        if board_size <= 8:
            # Smaller board = fewer/smaller ships
            return [
                Ship("Destroyer", 2),
                Ship("Submarine", 3),
                Ship("Battleship", 4),
            ]
        else:
            # Standard fleet
            return [
                Ship("Patrol Boat", 2),
                Ship("Destroyer", 3),
                Ship("Submarine", 3),
                Ship("Battleship", 4),
                Ship("Carrier", 5),
            ]

    def setup_game(self):
        """Setup the game"""
        print("="*50)
        print("BATTLESHIP GAME")
        print("="*50)
        print()

        # Get board size
        self.board_size = self.get_board_size()

        # Get game mode
        print("\nGame Modes:")
        print("1. Single Player (vs AI)")
        print("2. Practice Mode (no AI)")

        while True:
            mode = input("Select mode (1 or 2): ").strip()
            if mode in ["1", "2"]:
                self.game_mode = "ai" if mode == "1" else "practice"
                break
            print("Please enter 1 or 2")

        # Create boards
        self.player_board = Board(self.board_size)
        self.ai_board = Board(self.board_size)
        self.ai_player = AIPlayer(self.board_size)

        # Place ships
        print("\nPlacing your ships...")
        player_ships = self.create_ships(self.board_size)
        self.player_board.place_ships_randomly(player_ships)

        print("Placing AI ships...")
        ai_ships = self.create_ships(self.board_size)
        self.ai_board.place_ships_randomly(ai_ships)

        print("\nGame setup complete!")
        input("Press Enter to start...")

    def get_player_guess(self):
        """Get and validate player guess"""
        while True:
            try:
                row = input(f"Enter row (0-{self.board_size-1}): ").strip()
                if row.lower() == 'q':
                    return None, None

                col = input(f"Enter column (0-{self.board_size-1}): ").strip()
                if col.lower() == 'q':
                    return None, None

                row = int(row)
                col = int(col)

                if 0 <= row < self.board_size and 0 <= col < self.board_size:
                    return row, col
                else:
                    print(f"Please enter values between 0 and {self.board_size-1}")

            except ValueError:
                print("Please enter valid numbers")

    def print_game_state(self):
        """Print current game state"""
        self.clear_screen()
        print("="*50)
        print("BATTLESHIP GAME")
        print("="*50)
        print()

        # Print AI board (with ships hidden)
        print("OPPONENT'S BOARD:")
        self.ai_board.print_board(hide_ships=True)
        print()

        # Print player stats
        ai_stats = self.ai_board.get_stats()
        print(f"Your Stats: Guesses: {ai_stats['total_guesses']} | "
              f"Hits: {ai_stats['hits']} | Misses: {ai_stats['misses']} | "
              f"Accuracy: {ai_stats['accuracy']:.1f}%")
        print(f"Enemy Ships Remaining: {ai_stats['ships_remaining']}/{ai_stats['total_ships']}")
        print()

        if self.game_mode == "ai":
            # Print player board
            print("YOUR BOARD:")
            self.player_board.print_board(hide_ships=False)
            print()

            # Print AI stats
            player_stats = self.player_board.get_stats()
            print(f"AI Stats: Guesses: {player_stats['total_guesses']} | "
                  f"Hits: {player_stats['hits']} | Misses: {player_stats['misses']} | "
                  f"Accuracy: {player_stats['accuracy']:.1f}%")
            print(f"Your Ships Remaining: {player_stats['ships_remaining']}/{player_stats['total_ships']}")
            print()

        print("Legend: ~ = Water | O = Miss | X = Hit | S = Ship")
        print("Enter 'q' to quit")
        print()

    def play_turn(self):
        """Play one turn (player and AI if applicable)"""
        self.print_game_state()

        # Player turn
        print("YOUR TURN")
        row, col = self.get_player_guess()

        if row is None:  # Player wants to quit
            return False

        result, ship = self.ai_board.make_guess(row, col)

        if result == "invalid":
            print("Invalid position!")
            input("Press Enter to continue...")
        elif result == "duplicate":
            print("You already guessed that position!")
            input("Press Enter to continue...")
        elif result == "hit":
            print(f"HIT! You hit the {ship.name}!")
            input("Press Enter to continue...")
        elif result == "sunk":
            print(f"HIT! You sunk the {ship.name}!")
            input("Press Enter to continue...")
        elif result == "miss":
            print("MISS!")
            input("Press Enter to continue...")

        # Check if player won
        if self.ai_board.all_ships_sunk():
            return "player_wins"

        # AI turn (if in AI mode)
        if self.game_mode == "ai":
            print("\nAI'S TURN...")
            ai_row, ai_col = self.ai_player.make_guess(self.player_board)
            ai_result, ai_ship = self.player_board.make_guess(ai_row, ai_col)

            self.ai_player.process_result(ai_result, ai_row, ai_col, ai_ship)

            if ai_result == "hit":
                print(f"AI HIT your {ai_ship.name} at ({ai_row}, {ai_col})!")
            elif ai_result == "sunk":
                print(f"AI SUNK your {ai_ship.name} at ({ai_row}, {ai_col})!")
            elif ai_result == "miss":
                print(f"AI missed at ({ai_row}, {ai_col})")

            input("Press Enter to continue...")

            # Check if AI won
            if self.player_board.all_ships_sunk():
                return "ai_wins"

        return True

    def play_game(self):
        """Main game loop"""
        self.setup_game()

        # Game loop
        while True:
            result = self.play_turn()

            if result == "player_wins":
                self.print_game_state()
                print("="*50)
                print("CONGRATULATIONS! YOU WIN!")
                print("="*50)
                stats = self.ai_board.get_stats()
                print(f"\nFinal Stats:")
                print(f"Total Guesses: {stats['total_guesses']}")
                print(f"Hits: {stats['hits']}")
                print(f"Misses: {stats['misses']}")
                print(f"Accuracy: {stats['accuracy']:.1f}%")
                break

            elif result == "ai_wins":
                self.print_game_state()
                print("="*50)
                print("GAME OVER - AI WINS!")
                print("="*50)
                stats = self.ai_board.get_stats()
                print(f"\nYour Stats:")
                print(f"Total Guesses: {stats['total_guesses']}")
                print(f"Hits: {stats['hits']}")
                print(f"Misses: {stats['misses']}")
                print(f"Accuracy: {stats['accuracy']:.1f}%")
                break

            elif result == False:  # Player quit
                print("\nThanks for playing!")
                break


def main():
    """Main entry point"""
    while True:
        game = BattleshipGame()
        game.play_game()

        play_again = input("\nPlay again? (y/n): ").strip().lower()
        if play_again != 'y':
            print("\nThanks for playing Battleship!")
            break


if __name__ == "__main__":
    main()
