import math
import sys
import time
import numpy as np
from itertools import product

def print_board(board):
    for row in board:
        print(" ".join(map(str, row)))

def is_valid(row, col, num, possibilities):
    return num in possibilities[row, col]

def find_empty_location_with_fewest_possibilities(possibilities):
    empty_locations = [(i, j) for i, j in product(range(len(possibilities)), repeat=2) if possibilities[i, j]]
    
    if not empty_locations:
        return None
    
    return min(empty_locations, key=lambda pos: len(possibilities[pos[0], pos[1]]))

def update_possibilities(row, col, num, possibilities):
    possibilities[row, col] = set()
    possibilities[row, :] -= {num}
    possibilities[:, col] -= {num}
    box_size = int(np.sqrt(len(possibilities)))
    possibilities[row//box_size*box_size:(row//box_size+1)*box_size, col//box_size*box_size:(col//box_size+1)*box_size] -= {num}

def read_sudoku_from_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    puzzle = [[int(cell) for cell in line.strip().split()] for line in lines]
    return puzzle

def write_sudoku_to_file(file_path, puzzle):
    output_file_path = file_path.replace('.txt', '_output.txt')
    with open(output_file_path, 'w') as file:
        for row in puzzle:
            file.write(' '.join(map(str, row)) + '\n')

def solve_sudoku(board, possibilities):
    empty_location = find_empty_location_with_fewest_possibilities(possibilities)

    if not empty_location:
        return True

    row, col = empty_location

    for num in possibilities[row, col]:
        if is_valid(row, col, num, possibilities):
            board[row][col] = num
            update_possibilities(row, col, num, possibilities)

            if solve_sudoku(board, possibilities):
                return True

            board[row][col] = 0
            possibilities[row, col].add(num)
            possibilities[row, :] |= {num}
            possibilities[:, col] |= {num}
            box_size = int(np.sqrt(len(possibilities)))
            possibilities[row//box_size*box_size:(row//box_size+1)*box_size, col//box_size*box_size:(col//box_size+1)*box_size] |= {num}

    return False

def solve_and_measure_time(sudoku_board, puzzle_size):
    possibilities = [[] for _ in range(puzzle_size)]
    sub_size = int(math.sqrt(puzzle_size))

    for i in range(puzzle_size):
        for j in range(puzzle_size):
            if sudoku_board[i][j] == 0:
                possibilities[i].append(
                    set(range(1, puzzle_size+1))-set(sudoku_board[i])
                    -set([sudoku_board[k][j] for k in range(puzzle_size)])
                    -set([sudoku_board[k][l] for k in range(i//sub_size*sub_size,(i//sub_size+1)*sub_size) for l in range(j//sub_size*sub_size,(j//sub_size+1)*sub_size)])
                    -set([sudoku_board[k][l] for k in range(puzzle_size) for l in range(puzzle_size) if (k==i or l==j)]))
            else:
                possibilities[i].append(set())
    possibilities = np.array(possibilities)

    total_time = 0

    for _ in range(1):  
        start_time = time.time()
        if solve_sudoku(sudoku_board, possibilities):
            print(f"\nSolved {puzzle_size}x{puzzle_size} Sudoku:")
            elapsed_time = time.time() - start_time
            total_time += elapsed_time
            print(f"\nTime taken: {elapsed_time:.6f} seconds")
            print_board(sudoku_board)
            write_sudoku_to_file(file_path, sudoku_board)
        else:
            print("\nNo solution exists.")

    print(f"\nTotal time taken: {total_time:.6f} seconds")




if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python sudoku_solver.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]

    sudoku_board= read_sudoku_from_file(file_path)
    solve_and_measure_time(sudoku_board, puzzle_size=len(sudoku_board[0]))



