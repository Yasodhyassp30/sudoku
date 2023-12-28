#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <sstream>
#include <cmath>


using namespace std;
using namespace std::chrono;

const int SIZE_9X9 = 9;
const int SIZE_16X16 = 16;


template <int SIZE>
const int SUBGRID_SIZE = static_cast<int>(sqrt(SIZE));

template <int SIZE>
const int SUBGRID_SIZE_VALUE = SIZE / SUBGRID_SIZE<SIZE>;

template <int SIZE>
void SudokuGrid(int grid[SIZE][SIZE])
{
    for (int row = 0; row < SIZE; row++)
    {
        for (int col = 0; col < SIZE; col++)
        {
            cout << grid[row][col] << " ";
        }
        cout << endl;
    }
}

template <int SIZE>
bool isValidPlace(int grid[SIZE][SIZE], int row, int col, int num,int subgridRowStart,int subgridColStart)
{
    for (int i = 0; i < SIZE; ++i)
    {
        if (grid[row][i] == num || grid[i][col] == num || grid[subgridRowStart + i / SUBGRID_SIZE<SIZE>][subgridColStart + i % SUBGRID_SIZE<SIZE>] == num)
        {
            return false;
        }
    }

    return true;
}

template <int SIZE>
bool findMinPossibleEmpty(int grid[SIZE][SIZE], int &row, int &col, vector<int> &currentValues, vector<vector<vector<int>>> &allPossibleValues)
{
    int minPossibilities = SIZE + 1;
    for (int i = 0; i < SIZE; ++i)
    {
        for (int j = 0; j < SIZE; ++j)
        {
            if (grid[i][j] == 0)
            {
                int possibilities = 0;
                vector<int> Values;
                int subgridRowStart =SUBGRID_SIZE_VALUE<SIZE> * (i / SUBGRID_SIZE_VALUE<SIZE>);
                int subgridColStart = SUBGRID_SIZE_VALUE<SIZE> * (j / SUBGRID_SIZE_VALUE<SIZE>);
                for (const int &element : allPossibleValues[i][j])
                {
                    if (isValidPlace(grid, i, j, element,subgridRowStart,subgridColStart))
                    {
                        possibilities++;
                        Values.emplace_back(element);
                    }
                }
                if (possibilities < minPossibilities)
                {
                    minPossibilities = possibilities;
                    row = i;
                    col = j;
                    currentValues = Values;
                    if (minPossibilities == 1)
                    {
                        return true;
                    }
                }
            }
        }
    }
    return minPossibilities != SIZE + 1;
}

template <int SIZE>
vector<vector<vector<int>>> getPosibilities(int grid[SIZE][SIZE])
{
    vector<vector<vector<int>>> allPosibilities;
    for (int i = 0; i < SIZE; i++)
    {
        vector<vector<int>> row;
        for (int j = 0; j < SIZE; j++)
        {
            vector<int> possibleValues;
            if (grid[i][j] == 0)
            {
                int subgridRowStart = SUBGRID_SIZE_VALUE<SIZE> * (i / SUBGRID_SIZE_VALUE<SIZE>);
                int subgridColStart = SUBGRID_SIZE_VALUE<SIZE> * (j / SUBGRID_SIZE_VALUE<SIZE>);
                for (int num = 1; num <= SIZE; num++)
                {
                    if (isValidPlace(grid, i, j, num,subgridRowStart,subgridColStart))
                    {
                        possibleValues.emplace_back(num);
                    }
                }
            }
            row.emplace_back(possibleValues);
        }
        allPosibilities.emplace_back(row);
    }
    return allPosibilities;
}

template <int SIZE>
bool solveSudoku(int (&grid)[SIZE][SIZE], vector<vector<vector<int>>> &allPossiblecombinations)
{
    int row, col;
    vector<int> possibleValues;
    if (!findMinPossibleEmpty(grid, row, col, possibleValues, allPossiblecombinations))
        return true;

    for (const int &num : possibleValues)
    {
        grid[row][col] = num;
        if (solveSudoku(grid, allPossiblecombinations))
            return true;
        grid[row][col] = 0;
    }
    return false;
}

template <int SIZE>
void readInput(string filename, int (&grid)[SIZE][SIZE])
{
    ifstream inputFile(filename);
    if (!inputFile.is_open())
    {
        cerr << "Error: Unable to open input file." << endl;
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < SIZE; i++)
        for (int j = 0; j < SIZE; j++)
            inputFile >> grid[i][j];

    inputFile.close();
}

template <int SIZE>
void writeOutput(string filename, const int (&grid)[SIZE][SIZE])
{
    ofstream outputFile(filename);
    if (!outputFile.is_open())
    {
        cerr << "Error: Unable to open output file." << endl;
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < SIZE; i++)
    {
        for (int j = 0; j < SIZE; j++)
        {
            outputFile << grid[i][j] << " ";
        }
        outputFile << endl;
    }

    outputFile.close();
}

int obtainSize(string filename)
{
    ifstream inputFile(filename);
    if (!inputFile.is_open())
    {
        cerr << "Error: Unable to open input file." << endl;
        exit(EXIT_FAILURE);
    }

    int size = 0;
    std::string line;
    while (std::getline(inputFile, line))
    {
        std::istringstream iss(line);
        int value;
        while (iss >> value)
        {
            size++;
        }
        break;
    }

    inputFile.close();
    return size;
}

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        cerr << "provide file path" << endl;
        return EXIT_FAILURE;
    }

    string inputFileName = argv[1];

    int puzzleSize = obtainSize(inputFileName);

    if (puzzleSize == SIZE_9X9)
    {
        
        cout << "Solving 9x9 Sudoku puzzle..." << endl;
        int sudoku9x9[SIZE_9X9][SIZE_9X9];
        readInput(inputFileName, sudoku9x9);
        vector<vector<vector<int>>> allPossiblecombinations = getPosibilities(sudoku9x9);
        auto startTime = high_resolution_clock::now();

        if (solveSudoku(sudoku9x9, allPossiblecombinations))
        {
            auto endTime = high_resolution_clock::now();
            auto solveTime = duration_cast<duration<double>>(endTime - startTime);
            cout << "Sudoku solved in " << solveTime.count() << " seconds." << endl;
            SudokuGrid(sudoku9x9);
            writeOutput(inputFileName.substr(0, inputFileName.length() - 4) + "_output.txt", sudoku9x9);
        }
        else
        {
            cout << "No solution exists." << endl;
        }
    }
    else if (puzzleSize == SIZE_16X16)
    {
        cout << "Solving 16x16 Sudoku puzzle..." << endl;
        int sudoku16x16[SIZE_16X16][SIZE_16X16];
        readInput(inputFileName, sudoku16x16);
        vector<vector<vector<int>>> allPossiblecombinations = getPosibilities(sudoku16x16);
        auto startTime = high_resolution_clock::now();

        if (solveSudoku(sudoku16x16, allPossiblecombinations))
        {
            auto endTime = high_resolution_clock::now();
            auto solveTime = duration_cast<duration<double>>(endTime - startTime);
            cout << "Sudoku solved in " << solveTime.count() << " seconds." << endl;
            SudokuGrid(sudoku16x16);
            writeOutput(inputFileName.substr(0, inputFileName.length() - 4), sudoku16x16);
        }
        else
        {
            cout << "No solution exists." << endl;
        }
    }

    return 0;
}
