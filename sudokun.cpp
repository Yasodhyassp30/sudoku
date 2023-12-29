#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <ctime>

#define N 16

using namespace std;

bool isPresentInRow(int row, int num);
bool isPresentInCol(int col, int num);
bool isPresentInBox(int boxStartRow, int boxStartCol, int num);
int countPossibilities(int row, int col, int num);

int grid[N][N];

bool findEmptyPlace(int &row, int &col)
{
    int minPossibilities = N + 1; // Initialize to a value greater than the maximum possibilities
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            if (grid[i][j] == 0)
            {
                int possibilities = 0;
                for (int num = 1; num <= N; num++)
                    if (!isPresentInRow(i, num) && !isPresentInCol(j, num) && !isPresentInBox(i - i % 4, j - j % 4, num))
                        possibilities++;

                if (possibilities < minPossibilities)
                {
                    minPossibilities = possibilities;
                    row = i;
                    col = j;
                }
            }

    return minPossibilities != N + 1;
}

bool isPresentInRow(int row, int num)
{
    for (int col = 0; col < N; col++)
        if (grid[row][col] == num)
            return true;
    return false;
}

bool isPresentInCol(int col, int num)
{
    for (int row = 0; row < N; row++)
        if (grid[row][col] == num)
            return true;
    return false;
}

bool isPresentInBox(int boxStartRow, int boxStartCol, int num)
{
    for (int row = 0; row < 4; row++)
        for (int col = 0; col < 4; col++)
            if (grid[row + boxStartRow][col + boxStartCol] == num)
                return true;
    return false;
}

void hexadokuGrid()
{
    for (int row = 0; row < N; row++)
    {
        for (int col = 0; col < N; col++)
        {
            if (col == 4 || col == 8 || col == 12)
                cout << " | ";
            cout << grid[row][col] << " ";
        }
        if (row == 3 || row == 7 || row == 11)
        {
            cout << endl;
            for (int i = 0; i < N; i++)
                cout << "---";
        }
        cout << endl;
    }
}

bool isValidPlace(int row, int col, int num)
{
    return !isPresentInRow(row, num) && !isPresentInCol(col, num) &&
           !isPresentInBox(row - row % 4, col - col % 4, num);
}

bool solveHexadoku()
{
    int row, col;
    if (!findEmptyPlace(row, col))
        return true;

    // Get possible values for the current cell
    vector<int> possibleValues;
    for (int num = 1; num <= N; num++)
    {
        if (isValidPlace(row, col, num))
        {
            possibleValues.push_back(num);
        }
    }

    for (int num : possibleValues)
    {
        grid[row][col] = num;
        if (solveHexadoku())
            return true;
        grid[row][col] = 0;
    }
    return false;
}

int countPossibilities(int row, int col, int num)
{
    int possibilities = 0;
    for (int i = 0; i < N; i++)
        if (!isPresentInRow(row, num) && !isPresentInCol(col, num) && !isPresentInBox(row - row % 4, col - col % 4, num))
            possibilities++;

    return possibilities;
}

void readHexadokuInput(string filename)
{
    ifstream inputFile(filename);
    if (!inputFile.is_open())
    {
        cerr << "Error: Unable to open input file." << endl;
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            inputFile >> grid[i][j];

    inputFile.close();
}

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        cerr << "Usage: " << argv[0] << " <input_filename>" << endl;
        return EXIT_FAILURE;
    }

    string inputFileName = argv[1];
    readHexadokuInput(inputFileName);

    clock_t startTime = clock();
    if (solveHexadoku())
    {
        clock_t endTime = clock();
        double solveTime = double(endTime - startTime) / CLOCKS_PER_SEC;
        cout << "Hexadoku solved in " << solveTime << " seconds." << endl;
        hexadokuGrid();
    }
    else
    {
        cout << "No solution exists." << endl;
    }

    return 0;
}
