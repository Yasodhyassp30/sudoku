//19/ENG/075
//A.K.Y.S.S.PERERA


#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <sstream>
#include <cmath>
#include <iomanip>

using namespace std;
using namespace std::chrono;

const int SIZE_9X9 = 9;
const int SIZE_16X16 = 16;


//reduce redundant calculations for subgrid size
template <int SIZE>
const int SUBGRID_SIZE = static_cast<int>(sqrt(SIZE));

template <int SIZE>
const int SUBGRID_SIZE_VALUE = SIZE / SUBGRID_SIZE<SIZE>;



template <int SIZE>
void SudokuGrid(int grid[SIZE][SIZE]) //print sudoku puzzle
{
    for (int row = 0; row < SIZE; row++)
    {
        for (int col = 0; col < SIZE; col++)
        {
            cout << setw(2 + (SIZE == SIZE_16X16 ? 1 : 0)) << grid[row][col] << " ";
            if ((col + 1) % SUBGRID_SIZE_VALUE<SIZE> == 0 && col + 1 < SIZE)
            {
                cout << "|";
            }
        }

        cout << endl;

        if ((row + 1) % SUBGRID_SIZE_VALUE<SIZE> == 0 && row + 1 < SIZE)
        {
            for (int i = 0; i < SIZE * 2 - 1; i++)
            {
                cout << "__";
            }
            cout << endl;
        }
    }
}




template <int SIZE>
//check the value is valid for the given row, column and subgrid
bool isValidPlace(const int (&grid)[SIZE][SIZE], int row, int col, int num, int subgridRowStart, int subgridColStart)
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
//find the empty space with minimum possible values and get the row and colmn with newly updated possible values
bool findMinPossibleEmptySpaces(const int (&grid)[SIZE][SIZE], int &row, int &col, vector<int> &currentValues, vector<vector<vector<int>>> &allPossibleValues,vector<pair<int, int>> &emptySpaces)
{
    int minPossibilities = SIZE + 1;
    for (const pair<int, int> &emptySpace : emptySpaces)
    {
        int i = emptySpace.first;
        int j = emptySpace.second;
        if (grid[i][j] == 0){
                int possibilities = 0;
                vector<int> Values;
                int subgridRowStart = SUBGRID_SIZE_VALUE<SIZE> * (i / SUBGRID_SIZE_VALUE<SIZE>);
                int subgridColStart = SUBGRID_SIZE_VALUE<SIZE> * (j / SUBGRID_SIZE_VALUE<SIZE>);
                for (const int &element : allPossibleValues[i][j])
                {
                    if (isValidPlace(grid, i, j, element, subgridRowStart, subgridColStart))
                    {
                        possibilities++;
                        Values.emplace_back(element);//creating new vector for storing possible values
                    }
                    
                }
                if (possibilities < minPossibilities)
                {
                    minPossibilities = possibilities;
                    row = i;
                    col = j;
                    currentValues = Values;
                    if (minPossibilities == 1)//early stopping for when found possiblity of 1 
                    {
                        return true;
                    }
                }
        }
    }
    return minPossibilities != SIZE + 1;
}


template <int SIZE>
//get all the empty spaces in the grid
vector<pair<int, int>> getEmptySpaces(const int (&grid)[SIZE][SIZE])
{
    vector<pair<int, int>> emptySpaces;
    for (int i = 0; i < SIZE; ++i)
    {
        for (int j = 0; j < SIZE; j++)
        {
            if (grid[i][j] == 0)
            {
                emptySpaces.emplace_back(make_pair(i, j));
            }
        }
    }
    return emptySpaces;
}


template <int SIZE>
//get all the possible values for each empty space
vector<vector<vector<int>>> getPosibilities(const int (&grid)[SIZE][SIZE])
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
                    if (isValidPlace(grid, i, j, num, subgridRowStart, subgridColStart))
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
bool solveSudoku(int (&grid)[SIZE][SIZE], vector<vector<vector<int>>> &allPossiblecombinations,vector<pair<int, int>> &emptySpaces)
{
    int row, col;
    vector<int> possibleValues;
    //check if there is any empty space
    if (!findMinPossibleEmptySpaces(grid, row, col, possibleValues, allPossiblecombinations,emptySpaces))
        return true;

    //loop for all possible values
    for (const int &num : possibleValues)
    {
        grid[row][col] = num;
        if (solveSudoku(grid, allPossiblecombinations,emptySpaces))
            return true;
        grid[row][col] = 0;
    }
    return false;
}

template <int SIZE>
//validte the puzzle before solving
bool readPuzzleVaildate(const int (&grid)[SIZE][SIZE])
{
    int rowMask[SIZE] = {0};
    int colMask[SIZE] = {0};
    int subgridMask[SUBGRID_SIZE<SIZE>][SUBGRID_SIZE<SIZE>] = {{0}};

    for (int i = 0; i < SIZE; i++)
    {
        int subgridRow = i / SUBGRID_SIZE<SIZE>;
        for (int j = 0; j < SIZE; j++)
        {
            int subgridCol = j / SUBGRID_SIZE<SIZE>;
            if (grid[i][j] == 0)
                continue;
            int digit = grid[i][j];

            if ((rowMask[i] & (1 << digit)) != 0)
                return false;
            rowMask[i] |= (1 << digit);

            if ((colMask[j] & (1 << digit)) != 0)
                return false;
            colMask[j] |= (1 << digit);

            if ((subgridMask[subgridRow][subgridCol] & (1 << digit)) != 0)
                return false;
            subgridMask[subgridRow][subgridCol] |= (1 << digit);
        }
    }

    return true;
}
template <int SIZE>
//validte the puzzle after solving
bool solvedPuzzleVaildate(const int (&grid)[SIZE][SIZE])
{
    int rowMask[SIZE] = {0};
    int colMask[SIZE] = {0};
    int subgridMask[SUBGRID_SIZE<SIZE>][SUBGRID_SIZE<SIZE>] = {{0}};

    for (int i = 0; i < SIZE; i++)
    {
        int subgridRow = i / SUBGRID_SIZE<SIZE>;
        for (int j = 0; j < SIZE; j++)
        {
            int subgridCol = j / SUBGRID_SIZE<SIZE>;
            int digit = grid[i][j];
            if (digit == 0)
                return false;

            if ((rowMask[i] & (1 << digit)) != 0)
                return false;
            rowMask[i] |= (1 << digit);

            if ((colMask[j] & (1 << digit)) != 0)
                return false;
            colMask[j] |= (1 << digit);

            if ((subgridMask[subgridRow][subgridCol] & (1 << digit)) != 0)
                return false;
            subgridMask[subgridRow][subgridCol] |= (1 << digit);
        }
    }

    return true;
}

template <int SIZE>
void readInput(string filename, int (&grid)[SIZE][SIZE], vector<pair<int, int>> &emptySpaces)
{
    ifstream inputFile(filename);
    if (!inputFile.is_open())
    {
        cerr << "Error: Unable to open input file." << endl;
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < SIZE; i++){
        for (int j = 0; j < SIZE; j++){
            inputFile >> grid[i][j];
            if (grid[i][j] == 0)
            {
                emptySpaces.emplace_back(make_pair(i, j));
            }
        }
    }
    inputFile.close();

    if (!readPuzzleVaildate(grid))
    {
        cerr << "Error: Invalid Sudoku puzzle." << endl;
        exit(EXIT_FAILURE);
    }
}

template <int SIZE>
//write the solved puzzle to the output file
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

//obtain the puzzle size by reading first line of the provided file
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

    if (puzzleSize == SIZE_9X9)// for 9x9 puzzle
    {

        cout << "Solving 9x9 Sudoku puzzle..." << endl;
        int sudoku9x9[SIZE_9X9][SIZE_9X9];
        vector<pair<int, int>> emptySpaces;
        readInput(inputFileName, sudoku9x9,emptySpaces);
        auto startTime = high_resolution_clock::now();
        vector<vector<vector<int>>> allPossiblecombinations = getPosibilities(sudoku9x9);
        
        
        if (solveSudoku(sudoku9x9, allPossiblecombinations,emptySpaces))
        {
            auto endTime = high_resolution_clock::now();
            auto solveTime = duration_cast<duration<double>>(endTime - startTime);
            cout << "Sudoku solved in " << solveTime.count() << " seconds." << endl;
            SudokuGrid(sudoku9x9);
            writeOutput(inputFileName.substr(0, inputFileName.length() - 4) + "_output.txt", sudoku9x9);
            cout<<endl;
            if(solvedPuzzleVaildate(sudoku9x9)){
                cout << "Solved puzzle is valid." << endl;
            }
            else{
                cout << "Solved puzzle is invalid." << endl;
            }
        }
        else
        {
            cout << "No solution exists." << endl;
        }
    }
    else if (puzzleSize == SIZE_16X16)//for 16x16 puzzle
    {
        cout << "Solving 16x16 Sudoku puzzle..." << endl;

        int sudoku16x16[SIZE_16X16][SIZE_16X16];
        vector<pair<int, int>> emptySpaces;
        readInput(inputFileName, sudoku16x16,emptySpaces);
        
        auto startTime = high_resolution_clock::now();
        vector<vector<vector<int>>> allPossiblecombinations = getPosibilities(sudoku16x16);
        if (solveSudoku(sudoku16x16, allPossiblecombinations,emptySpaces))
        {
            auto endTime = high_resolution_clock::now();
            auto solveTime = duration_cast<duration<double>>(endTime - startTime);
            cout << "Sudoku solved in " << solveTime.count() << " seconds." << endl;
            SudokuGrid(sudoku16x16);
            writeOutput(inputFileName.substr(0, inputFileName.length() - 4) + "_output.txt", sudoku16x16);
            cout<<endl;
            if(solvedPuzzleVaildate(sudoku16x16)){
                cout << "Solved puzzle is valid." << endl;
            }
            else{
                cout << "Solved puzzle is invalid." << endl;
            }
        }
        else
        {
            cout << "No solution exists." << endl;
        }
    }

    return 0;
}
