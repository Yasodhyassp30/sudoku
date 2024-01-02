import math
from typing import List, Tuple
import time

SIZE_9X9 = 9
SIZE_16X16 = 16


def print_sudoku(grid):
    for row in grid:
        print(" ".join(map(str, row)))
    print()


def is_valid_place(grid, row, col, num, subgrid_row_start, subgrid_col_start):
    for i in range(len(grid)):
        if (
            grid[row][i] == num
            or grid[i][col] == num
            or grid[subgrid_row_start + i // int(math.sqrt(len(grid)))][subgrid_col_start + i % int(math.sqrt(len(grid)))] == num
        ):
            return False
    return True


def find_min_possible_empty(grid, all_possible_values):
    min_possibilities = len(grid) + 1
    min_row, min_col, current_values = 0, 0, []
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == 0:
                possibilities = 0
                values = []
                subgrid_row_start = (i // int(math.sqrt(len(grid)))) * int(math.sqrt(len(grid)))
                subgrid_col_start = (j // int(math.sqrt(len(grid)))) * int(math.sqrt(len(grid)))
                for element in all_possible_values[i][j]:
                    if is_valid_place(grid, i, j, element, subgrid_row_start, subgrid_col_start):
                        possibilities += 1
                        values.append(element)
                if possibilities < min_possibilities:
                    min_possibilities = possibilities
                    min_row, min_col, current_values = i, j, values
                    if min_possibilities == 1:
                        return min_row, min_col, current_values, True
    return min_row, min_col, current_values, min_possibilities != len(grid) + 1


def get_possibilities(grid):
    all_possibilities = []
    for i in range(len(grid)):
        row = []
        for j in range(len(grid[0])):
            possible_values = []
            if grid[i][j] == 0:
                subgrid_row_start = (i // int(math.sqrt(len(grid)))) * int(math.sqrt(len(grid)))
                subgrid_col_start = (j // int(math.sqrt(len(grid)))) * int(math.sqrt(len(grid)))
                for num in range(1, len(grid) + 1):
                    if is_valid_place(grid, i, j, num, subgrid_row_start, subgrid_col_start):
                        possible_values.append(num)
                row.append(possible_values)
            else:
                row.append([])
        all_possibilities.append(row)
    return all_possibilities


def solve_sudoku(grid, all_possible_combinations):
    row, col, possible_values, no_solution = find_min_possible_empty(grid, all_possible_combinations)
    if not no_solution:
        return True

    for num in possible_values:
        grid[row][col] = num
        if solve_sudoku(grid, all_possible_combinations):
            return True
        grid[row][col] = 0

    return False


def read_input(filename):
    with open(filename, "r") as input_file:
        return [[int(num) for num in line.split()] for line in input_file]


def write_output(filename, grid):
    with open(filename, "w") as output_file:
        for row in grid:
            output_file.write(" ".join(map(str, row)) + "\n")


def obtain_size(filename):
    with open(filename, "r") as input_file:
        return sum(1 for line in input_file.readline().split())

def readPuzzleVaildate(grid):
    rowMask = [0] * len(grid)
    colMask = [0] * len(grid)
    subgridMask = [[0] * int(math.sqrt(len(grid))) for _ in range(int(math.sqrt(len(grid))))]

    for i in range(len(grid)):
        subgridRow = i // int(math.sqrt(len(grid)))
        for j in range(len(grid[0])):
            subgridCol = j // int(math.sqrt(len(grid)))
            if grid[i][j] == 0:
                continue
            digit = grid[i][j] - 1

            if rowMask[i] & (1 << digit) != 0:
                return False
            rowMask[i] |= 1 << digit

            if colMask[j] & (1 << digit) != 0:
                return False
            colMask[j] |= 1 << digit

            if subgridMask[subgridRow][subgridCol] & (1 << digit) != 0:
                return False
            subgridMask[subgridRow][subgridCol] |= 1 << digit

    return True

def solve_16x16(grid):
    if readPuzzleVaildate(grid) == False:
        return False
    all_possible_combinations = get_possibilities(grid)
    if solve_sudoku(grid, all_possible_combinations):
        return True
    return False

def solve_9x9(grid):
    if readPuzzleVaildate(grid) == False:
        return False
    all_possible_combinations = get_possibilities(grid)
    if solve_sudoku(grid, all_possible_combinations):
        return True
    return False

