'''
Algorithm
1. Start.
2. Create an 8 x 8 chessboard initialized with zeros.
3. Read the first Queen's row and column position.
4. Place the first Queen at the given position.
5. Check whether a position is safe:
   - No Queen should be present in the same column.
   - No Queen should be present in the left diagonal.
   - No Queen should be present in the right diagonal.
6. If the position is safe, place the Queen.
7. Recursively place the remaining Queens in the next rows.
8. If no safe position is available, remove the previously placed Queen and try another position (Backtracking).
9. Repeat until all eight Queens are placed.
10. Display the final chessboard.
11. Stop. '''



N = 8

def is_safe(board, row, col):

    # Check same column
    for i in range(row):
        if board[i][col] == 1:
            return False

    # Check upper-left diagonal
    i, j = row - 1, col - 1
    while i >= 0 and j >= 0:
        if board[i][j] == 1:
            return False
        i -= 1
        j -= 1

    # Check upper-right diagonal
    i, j = row - 1, col + 1
    while i >= 0 and j < N:
        if board[i][j] == 1:
            return False
        i -= 1
        j += 1

    return True


def solve(board, row):

    # All 8 queens are placed
    if row == N:
        return True

    # Try each column
    for col in range(N):

        if is_safe(board, row, col):

            board[row][col] = 1

            if solve(board, row + 1):
                return True

            # Backtrack
            board[row][col] = 0

    return False


# Main program
board = [[0 for _ in range(N)] for _ in range(N)]

r = int(input("Enter first queen row (1-8): ")) - 1
c = int(input("Enter first queen column (1-8): ")) - 1

if not (0 <= r < N and 0 <= c < N):
    print("Invalid position")

else:
    board[r][c] = 1

    if solve(board, r + 1):
        print("\n8-Queens Solution:")
        for row in board:
            print(*row)
    else:
        print("No solution exists")


'''
Explanation
- An 8 x 8 chessboard is created.
- The first Queen is placed at position (1, 1).
- The algorithm checks all possible safe positions for the remaining seven Queens.
- If a Queen cannot be placed safely, the algorithm backtracks and tries another position.
- Finally, all eight Queens are placed without attacking each other.
Complexity Analysis
- Time Complexity: (O(N!)) (upper-bound for the standard backtracking approach).
- Space Complexity: (O(N^2)) for the chessboard and (O(N)) for the recursion stack.  '''