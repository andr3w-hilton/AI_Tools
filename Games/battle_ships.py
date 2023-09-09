# This game was made in a single prompt using Claude
# This is a simple game of battleships. The user has 3 turns to guess the location of the ship.

TODO = """
    1. Add a score counter
    2. Add multiple ships
    3. Add a 2 player mode that is AI based (Reinforcement Learning type Model)
    4. Keep track of hits and misses
    5. Implement basic input validation
    6. Design a better board maybe make the size user defined

"""

from random import randint

board = []

for x in range(5):
    board.append(["O"] * 5)


def print_board(board):
    for row in board:
        print(" ".join(row))


print("Let's play Battleship!")
print_board(board)


def random_row(board):
    return randint(0, len(board) - 1)


def random_col(board):
    return randint(0, len(board[0]) - 1)


ship_row = random_row(board)
ship_col = random_col(board)

for turn in range(4):
    guess_row = int(input("Guess Row:"))
    guess_col = int(input("Guess Col:"))

    if guess_row == ship_row and guess_col == ship_col:
        print("Congratulations! You sunk my battleship!")
        break
    else:
        if (guess_row < 0 or guess_row > 4) or (guess_col < 0 or guess_col > 4):
            print("Oops, that's not even in the ocean.")
        elif board[guess_row][guess_col] == "X":
            print("You guessed that one already.")
        else:
            print("You missed my battleship!")
            board[guess_row][guess_col] = "X"
        print_board(board)
        if turn == 3:
            print("Game Over")
