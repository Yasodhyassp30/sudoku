import math

def find_empty_place(grid_size, grid):
    for i in range(grid_size):
        for j in range(grid_size):
            if grid[i][j] == 0:
                return i, j
    return None

def is_present_in_row(grid_size, row, num, grid):
    return num in grid[row]

def is_present_in_col(grid_size, col, num, grid):
    return num in [grid[i][col] for i in range(grid_size)]

def is_present_in_box(grid_size, box_start_row, box_start_col, num, grid):
    return any(num == grid[row + box_start_row][col + box_start_col] for row in range(int(math.sqrt(grid_size)))
               for col in range(int(math.sqrt(grid_size))))


def print_sudoku(grid):
    for row in grid:
        print(' '.join(map(str, row)))

def is_valid_place(grid_size, row, col, num, grid):
    return not (is_present_in_row(grid_size, row, num, grid) or
                is_present_in_col(grid_size, col, num, grid) or
                is_present_in_box(grid_size, row - row % int(math.sqrt(grid_size)), col - col % int(math.sqrt(grid_size)), num, grid))

def solve_sudoku(grid_size, grid):
    empty_place = find_empty_place(grid_size, grid)
    if not empty_place:
        return True

    row, col = empty_place

    # Get possible values for the current cell
    possible_values = [num for num in range(1, grid_size + 1) if is_valid_place(grid_size, row, col, num, grid)]

    # Try possibilities with the fewest options first
    possible_values.sort()

    for num in possible_values:
        grid[row][col] = num
        if solve_sudoku(grid_size, grid):
            return True
        grid[row][col] = 0

    return False

def read_input(filename):
    with open(filename, 'r') as input_file:
        lines = input_file.readlines()

        # Determine the grid size from the length of the first row
        grid_size = len(lines[0].split())

        # Parse the grid values
        grid = [list(map(int, line.split())) for line in lines]

    return grid_size, grid

if __name__ == "__main__":
    import sys
    import time

    if len(sys.argv) != 2:
        print("Usage:", sys.argv[0], "<input_filename>")
        sys.exit(1)

    input_filename = sys.argv[1]
    grid_size, grid = read_input(input_filename)

    start_time = time.time()
    if solve_sudoku(grid_size, grid):
        end_time = time.time()
        solve_time = end_time - start_time
        print(f"Sudoku solved in {solve_time:.6f} seconds.")
        print_sudoku( grid)
    else:
        print("No solution exists.")
