// CPP program to implement Strassen’s Matrix
// Multiplication Algorithm
#include <bits/stdc++.h>
#include <omp.h>

using namespace std;

typedef long long lld;
int MAX_DEPTH = 4;
int threshold = 2;
/* Strassen's Algorithm for matrix multiplication
Complexity: O(n^2.808) */


int findEven(int x)
{
    if ((x%2) == 0)
        return -1;
    return x + 1;
}


inline lld** padding (lld ** a, int &R, int &C) 
{
    int newR = findEven(R);
    int newC = newR;
    if (newR == -1)
        return a;

    lld ** padded = new lld * [newR];
    for (int i = 0; i < newR; i++) {
        padded[i] = new lld [newC];
        for (int j = 0; j < newC; j++) 
        {
            if (i < R && j < C) 
            {
                padded[i][j] = a[i][j];
            } 
            else 
            {
                padded[i][j] = 0;
            }
        }
    }

    R = newR;
    C = newC;
    return padded;
}


inline lld** MatrixMultiply(lld** a, lld** b, int n,  int l, int m)
{
    lld** c = new lld*[n];
    for (int i = 0; i < n; i++)
        c[i] = new lld[m];

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            c[i][j] = 0;
            for (int k = 0; k < l; k++) {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    return c;
}

inline lld** Strassen(lld** a, lld** b, int R, int C, int depth)
{
    if (R == threshold || C == threshold)
        return MatrixMultiply(a, b, R, C, C);

    int size;
    if (R%2 !=0) size = R/2 +1;
    else size = R/2;

    lld **** As = new lld***[2];
    for (int x = 0; x < 2; x++) 
    {
        As[x] = new lld ** [2];
        for (int y = 0; y<2; y++)
        {
            As[x][y] = new lld *[size];
            for (int i = 0; i < size; i++)
            {
                As[x][y][i] = new lld [size];
                for (int j = 0; j < size; j++)
                {
                    int I = i + (x & 1) * (size);
                    int J = j + (y & 1) * (size);
                    As[x][y][i][j] = (I < R && J < C) ? a[I][J] : 0;
                }
            }
        }
    }

    lld **** Bs = new lld***[2];
    for (int x = 0; x < 2; x++) 
    {
        Bs[x] = new lld ** [2];
        for (int y = 0; y<2; y++)
        {
            Bs[x][y] = new lld *[size];
            for (int i = 0; i < size; i++)
            {
                Bs[x][y][i] = new lld [size];
                for (int j = 0; j < size; j++)
                {
                    int I = i + (x & 1) * (size);
                    int J = j + (y & 1) * (size);
                    Bs[x][y][i][j] = (I < R && J < C) ? b[I][J] : 0;
                }
            }
        }
    }


    ////// Test split //////
    // for (int x =0; x<2; x++) 
    // {
    //     for (int y =0; y<2; y++)
    //     {
    //         for (int i = 0; i < R/2; i++)
    //         {
    //             for (int j = 0; j < C/2; j++)
    //             {
    //                 cout << As[x][y][i][j] << " ";
    //             }
    //             cout << endl;
    //         }
    //     }
    // }


    //////////// A11 + A22 ////////////
    lld *** s = new lld**[10];
    s[0] = new lld*[size];
    for (int j = 0; j < size; j++) {
        s[0][j] = new lld[size];
        for (int k = 0; k < size; k++) {
            s[0][j][k] = As[0][0][j][k] + As[1][1][j][k];
        }
    }

    //////////// B11 + B22 ////////////
    s[1] = new lld*[size];
    for (int j = 0; j < size; j++) {
        s[1][j] = new lld[size];
        for (int k = 0; k < size; k++) {
            s[1][j][k] = Bs[0][0][j][k] + Bs[1][1][j][k];
        }
    }

    //////////// A21 + A22 ////////////
    s[2] = new lld*[size];
    for (int j = 0; j < size; j++) {
        s[2][j] = new lld[size];
        for (int k = 0; k < size; k++) {
            s[2][j][k] = As[1][0][j][k] + As[1][1][j][k];
        }
    }
    
    //////////// B12 - B22 ////////////
    s[3] = new lld*[size];
    for (int j = 0; j < size; j++) {
        s[3][j] = new lld[size];
        for (int k = 0; k < size; k++) {
            s[3][j][k] = Bs[0][1][j][k] - Bs[1][1][j][k];
        }
    }

    //////////// B21 - B11 ////////////
    s[4] = new lld*[size];
    for (int j = 0; j < size; j++) {
        s[4][j] = new lld[size];
        for (int k = 0; k < size; k++) {
            s[4][j][k] = Bs[1][0][j][k] - Bs[0][0][j][k];
        }
    }

    //////////// A11 + A12 ////////////
    s[5] = new lld*[size];
    for (int j = 0; j < size; j++) {
        s[5][j] = new lld[size];
        for (int k = 0; k < size; k++) {
            s[5][j][k] = As[0][0][j][k] + As[0][1][j][k];
        }
    }

    //////////// A21 - A11 ////////////
    s[6] = new lld*[size];
    for (int j = 0; j < size; j++) {
        s[6][j] = new lld[size];
        for (int k = 0; k < size; k++) {
            s[6][j][k] = As[1][0][j][k] - As[0][0][j][k];
        }
    }

    //////////// B11 + B12 ////////////
    s[7] = new lld*[size];
    for (int j = 0; j < size; j++) {
        s[7][j] = new lld[size];
        for (int k = 0; k < size; k++) {
            s[7][j][k] = Bs[0][0][j][k] + Bs[0][1][j][k];
        }
    }

    //////////// A12 - A22 ////////////
    s[8] = new lld*[size];
    for (int j = 0; j < size; j++) {
        s[8][j] = new lld[size];
        for (int k = 0; k < size; k++) {
            s[8][j][k] = As[0][1][j][k] - As[1][1][j][k];
        }
    }

    //////////// B21 + B22 ////////////
    s[9] = new lld*[size];
    for (int j = 0; j < size; j++) {
        s[9][j] = new lld[size];
        for (int k = 0; k < size; k++) {
            s[9][j][k] = Bs[1][0][j][k] + Bs[1][1][j][k];
        }
    }

    lld*** M = new lld**[7];
    #pragma omp taskgroup
    {
        #pragma omp task firstprivate(depth) shared(M) if (depth < MAX_DEPTH)
        {
            M[0] = Strassen(s[0], s[1], size, size, depth +1);
        }
        #pragma omp task firstprivate(depth) shared(M) if (depth < MAX_DEPTH)
        {
            M[1] = Strassen(s[2], Bs[0][0], size, size, depth +1);
        }
        #pragma omp task firstprivate(depth) shared(M) if (depth < MAX_DEPTH)
        {
            M[2] = Strassen(As[0][0], s[3], size, size, depth +1);
        }
        #pragma omp task firstprivate(depth) shared(M) if (depth < MAX_DEPTH)
        {
            M[3] = Strassen(As[1][1], s[4], size, size, depth +1);
        }
        #pragma omp task firstprivate(depth) shared(M) if (depth < MAX_DEPTH)
        {
            M[4] = Strassen(s[5], Bs[1][1], size, size, depth +1);
        }
        #pragma omp task firstprivate(depth) shared(M) if (depth < MAX_DEPTH)
        {
            M[5] = Strassen(s[6], s[7], size, size, depth +1);
        }
        #pragma omp task firstprivate(depth) shared(M) if (depth < MAX_DEPTH)
        {
            M[6] = Strassen(s[8], s[9], size, size, depth +1);
        }
    }

    lld ** c = new lld*[R];
    for (int i = 0; i < R; i++)
        c[i] = new lld[C];
    

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < size; i++) 
    {
        for (int j = 0; j < size; j++) 
        {
            c[i][j] = M[0][i][j] + M[3][i][j] - M[4][i][j] + M[6][i][j];
            if (j + size < C)
                c[i][j + size] = M[2][i][j] + M[4][i][j];
            if (i + size < R)
                c[i + size][j] = M[1][i][j] + M[3][i][j];
            if (i + size < R && j + size < C)
                c[i + size][j + size] = M[0][i][j] - M[1][i][j] + M[2][i][j] + M[5][i][j];
        }
    }

    // #pragma omp taskgroup
    // {
    // // C11
    //     #pragma omp task shared(c,M)
    //     {
    //         #pragma omp parallel for collapse(2)
    //         for (int i = 0; i < size; ++i)
    //             for (int j = 0; j < size; ++j)
    //                 c[i][j] = M[0][i][j] + M[3][i][j] - M[4][i][j] + M[6][i][j];
    //     }

    //     // C12
    //     #pragma omp task shared(c,M)
    //     {
    //         #pragma omp parallel for collapse(2)
    //         for (int i = 0; i < size; ++i)
    //             for (int j = 0; (j + size) < C && j < size; ++j)
    //                 c[i][j + size] = M[2][i][j] + M[4][i][j];
    //     }

    //     // C21
    //     #pragma omp task shared(c,M)
    //     {
    //         #pragma omp parallel for collapse(2)
    //         for (int i = 0; i + size < R && i < size; ++i)
    //             for (int j = 0; j < size; ++j)
    //                 c[i + size][j] = M[1][i][j] + M[3][i][j];
    //     }   

    //     // C22
    //     #pragma omp task shared(c,M)
    //     {
    //         #pragma omp parallel for collapse(2)
    //         for (int i = 0; i + size < R && i < size; ++i)
    //             for (int j = 0; j + size < C && j < size; ++j)
    //                 c[i + size][j + size] = M[0][i][j] - M[1][i][j] + M[2][i][j] + M[5][i][j];
    //     }
    // }
    return c;
}


// ============================================================================