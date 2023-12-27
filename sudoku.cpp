#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <ctime>
#include <sstream>
#include <cmath>


using namespace std;


bool isPresentInRow(vector<vector<int>>&grid,int row, int num,int size)
{
    for (int col = 0; col < size; col++)
        if (grid[row][col] == num)
            return true;
    return false;
}

bool isPresentInCol(vector<vector<int>>grid,int col, int num,int size)
{
    for (int row = 0; row < size; row++)
        if (grid[row][col] == num)
            return true;
    return false;
}

bool isPresentInBox(vector<vector<int>>grid,int boxStartRow, int boxStartCol, int num, int subgrid)
{
    for (int row = 0; row < subgrid; row++)
        for (int col = 0; col < subgrid; col++)
            if (grid[row + boxStartRow][col + boxStartCol] == num)
                return true;
    return false;
}

bool findEmptyPlace(vector<vector<int>>grid,int &row, int &col, int size, int subgrid)
{
    int minPossibilities = size + 1; // Initialize to a value greater than the maximum possibilities
    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++)
            if (grid[i][j] == 0)
            {
                int possibilities = 0;
                for (int num = 1; num <= size; num++)
                    if (!isPresentInRow(grid, i, num,size) && !isPresentInCol(grid,j, num,size) && !isPresentInBox(grid,i - i % subgrid, j - j % subgrid, num,subgrid))
                        possibilities++;

                if (possibilities < minPossibilities)
                {
                    minPossibilities = possibilities;
                    row = i;
                    col = j;
                }
            }

    return minPossibilities != size + 1;
}


void SudokuGrid(vector<vector<int>>grid,int size)
{
    for (int row = 0; row < size; row++)
    {
        for (int col = 0; col < size; col++)
        {
            cout << grid[row][col] << " ";
        }
        cout << endl;
    }
}

bool isValidPlace(vector<vector<int>>grid,int row, int col, int num,int size,int subgrid)
{
    return !isPresentInRow(grid,row, num, size) && !isPresentInCol(grid,col, num,size) &&
           !isPresentInBox(grid,row - row % subgrid, col - col % subgrid, num,subgrid);
}

vector <vector<vector<int>>> getPosibilites(vector<vector<int>>grid,int size, int subgrid){
    vector <vector<vector<int>>> allPosibilites;
    for (int i = 0; i < size; i++){
        vector<vector<int>> row;
        for (int j = 0; j < size; j++){
            vector<int> possibleValues;
            for (int num = 1; num <= size; num++)
            {
                if (isValidPlace(grid,i, j, num,size,subgrid))
                {
                    possibleValues.push_back(num);
                }
            }
            allPosibilites.push_back(row);
            row.push_back(possibleValues);
        }
    }
    return allPosibilites;
}

int countPossibilities(vector<vector<int>>grid,int row, int col, int num,int size,int subgrid)
{
    int possibilities = 0;
    for (int i = 0; i < size; i++)
        if (!isPresentInRow(grid,row, num,size) && !isPresentInCol(grid,col, num,size) && !isPresentInBox(grid,row - row % subgrid, col - col % subgrid, num,subgrid))
            possibilities++;

    return possibilities;
}
bool solveSudoku(vector<vector<int>>&grid,int size, int subgrid)
{
    int row, col;
    if (!findEmptyPlace(grid,row, col, size, subgrid))
        return true;
    vector<int> possibleValues;
    for (int num = 1; num <= size; num++)
    {
        if (isValidPlace(grid,row, col, num,size,subgrid))
        {
            possibleValues.push_back(num);
        }
    }
    sort(possibleValues.begin(), possibleValues.end(), [&](int a, int b) {
        return countPossibilities(grid,row, col,size,subgrid, a) < countPossibilities(grid,row, col,size,subgrid, b);
    });

    for (int num : possibleValues)
    {
        grid[row][col] = num;
        if (solveSudoku(grid,size,subgrid))
            return true;
        grid[row][col] = 0;
    }
    return false;
}



vector<vector<int>> readInput(string filename)
{
    vector<vector<int>> grid;
    ifstream inputFile(filename);
    if (!inputFile.is_open())
    {
        cerr << "Error: Unable to open input file." << endl;
        exit(EXIT_FAILURE);
    }
    std::string line;
    while (std::getline(inputFile, line)) {
        std::istringstream iss(line);
        vector <int> row;
        int value;
        while (iss >> value) {
            row.push_back(value);
        }
        grid.push_back(row);
    }
    
    inputFile.close();
    return grid;
}

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        cerr << "Usage: " << argv[0] << " <input_filename>" << endl;
        return EXIT_FAILURE;
    }

    string inputFileName = argv[1];
    vector<vector<int>> grid = readInput(inputFileName);

    int length = grid.size();
    int subgrid = sqrt(length);

    clock_t startTime = clock();
    if (solveSudoku(grid,length, subgrid))
    {
        clock_t endTime = clock();
        double solveTime = double(endTime - startTime) / CLOCKS_PER_SEC;
        cout << "Sudoku solved in " << solveTime << " seconds." << endl;
        SudokuGrid(grid,length);
    }
    else
    {
        cout << "No solution exists." << endl;
    }

    return 0;
}
