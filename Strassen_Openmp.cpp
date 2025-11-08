#include<bits/stdc++.h>
#include"Strassen_Openmp.h"
using namespace std;




int main(int argc, char** argv)
{
    int R = 1000;
    int C = 1000;
    lld** matA;
    matA = new lld*[R];
    lld** matB;
    matB = new lld*[R];
    for (int i = 0; i < R; i++)
        matA[i] = new lld[C];
    for (int i = 0; i < R; i++)
        matB[i] = new lld[C];

    srand(42);

    for (int i = 0; i < R; ++i) {
        for (int j = 0; j < C; ++j) {
            int val = rand() % 100;
            matA[i][j] = val;
            matB[i][j] = val;
        }
    }
    cout << matA[51][51] << endl;
    
    omp_set_num_threads(argc > 1 ? atoi(argv[1]) : 1);
    #pragma omp parallel
    {
        #pragma omp single
        {
            matA = Strassen(matA, matB, R, C, 0);
        }
    }

    cout << matA[51][51] << endl;

    return 0;
}