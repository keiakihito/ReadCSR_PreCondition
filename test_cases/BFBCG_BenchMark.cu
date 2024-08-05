/*
Personal Note to compile this program in NCSA delta.

1. Srun
srun --account=bchn-delta-gpu --partition=gpuA40x4-interactive --nodes=1 --gpus-per-node=1 --tasks-per-node=16 --cpus-per-task=1 --mem=20G --pty bash

2. Comile with this long long command
nvcc bfbcgTest.cu -o bfbcgTest -I/sw/spack/deltas11-2023-03/apps/linux-rhel8-zen3/gcc-11.4.0/cuda-11.8.0-vfixfmc/include -L/sw/spack/deltas11-2023-03/apps/linux-rhel8-zen3/gcc-11.4.0/cuda-11.8.0-vfixfmc/lib64 -lcudart -lcublas -lcusolver -lcusparse -I/sw/spack/deltas11-2023-03/apps/linux-rhel8-zen3/gcc-11.4.0/magma-2.8.0-mvkrj4y/include -L/sw/spack/deltas11-2023-03/apps/linux-rhel8-zen3/gcc-11.4.0/magma-2.8.0-mvkrj4y/lib -L/sw/spack/deltas11-2023-03/apps/linux-rhel8-zen3/gcc-11.4.0/openblas-0.3.25-5yvxjnl/lib -lmagma -lopenblas

3. Set path for magma
 export LD_LIBRARY_PATH=/sw/spack/deltas11-2023-03/apps/linux-rhel8-zen3/gcc-11.4.0/magma-2.8.0-mvkrj4y/lib:/sw/spack/deltas11-2023-03/apps/linux-rhel8-zen3/gcc-11.4.0/cuda-11.8.0-vfixfmc/lib64:$LD_LIBRARY_PATH

 4. 
 ./bfbcgTest
 */




// includes, system
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include <math.h>
#include <vector>

/*Using updated (v2) interfaces to cublas*/
#include<cublas_v2.h>
#include<cuda_runtime.h>
#include<cusparse.h>
#include <cusolverDn.h>
#include<sys/time.h>


//Utilities
// #include "../include/utils/checks.h"
// #include "../include/functions/helper.h"
// #include "../include/functions/cuBLAS_util.h"
// #include "../include/functions/cuSPARSE_util.h"
// #include "../include/functions/cuSOLVER_util.h"
#include "../include/struct/BenchMarkBFBCG.h"
#include "../include/functions/bfbcg.h"

// #include "../include/struct/CSRMatrix.h"







void bfbcgTest_Case(BenchMarkBFBCG& bmBFBCG, int numOfA, int numOfBlock);

int main(int arg, char** argv)
{
    
    printf("\n\n= = = =Benchmark bfbcg each operation in 1st iteration= = = = \n\n");
    const int NUM_OF_ITERATION = 51; //Eliminate 1st iteration result
    const int NUM_OF_BLOCK = 16; // Block size for 

    BenchMarkBFBCG bmBFBCG; // Store all the results
    std::vector<double> averages; // Averages time for each opeartion

    //Sparse size 2^5 and 2^15
    for(int i = 5; i < 16; i++){        
        for(int j = 0; j < NUM_OF_ITERATION; j++){
            bfbcgTest_Case(bmBFBCG, (int)pow(2, i), NUM_OF_BLOCK);        
        }
        eliminateFirst(bmBFBCG); // Get rid of the first result
        averages = getAverages(bmBFBCG); // Calculate each operatoin average time
        printf("\n\n~~ matrix A 2^%d average time~~\n", i);
        printResult(averages); //Display result

        averages.clear(); // Clear vectors for the next average calculation
        clearBenchMarkBFBCG(bmBFBCG); // Rest all the vectors in the struct for the next calculation
    } // end of for 


    printf("\n\n= = = = End of Benchmark bfbcg  = = = =\n\n");

    return 0;
} // end of main



void bfbcgTest_Case(BenchMarkBFBCG& bmBFBCG, int numOfA, int numOfBlock)
{
    cudaDeviceSynchronize();
    bool debug = false;

    const int M = numOfA; 
    const int K = numOfA;
    const int N = numOfBlock;
    

    CSRMatrix csrMtxA_h = generateSparseSPDMatrixCSR(M);
    

    double mtxB_h[K*N];
    initializeRandom(mtxB_h, K, N);
    

    //(1) Allocate memory
    double* mtxSolX_d = NULL;
    double* mtxB_d = NULL;

    CHECK(cudaMalloc((void**)&mtxSolX_d,  K * N * sizeof(double)));
    CHECK(cudaMalloc((void**)&mtxB_d,  M * N * sizeof(double)));

    //(2) Copy value from host to device
    CHECK(cudaMemcpy(mtxSolX_d, mtxB_h, K * N * sizeof(double), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(mtxB_d, mtxB_h, M * N * sizeof(double), cudaMemcpyHostToDevice));

    if(debug){
        printf("\n\n~~csrMtxA_h~~\n\n");
        print_CSRMtx(csrMtxA_h);
        printf("\n\n~~mtxSolX~~\n\n");
        print_mtx_clm_d(mtxSolX_d, K, N);
        printf("\n\n~~mtxB~~\n\n");
        print_mtx_clm_d(mtxB_d, M, N);
    }

    //Solve AX = B with bfbcg method
    bfbcg_BenchMark(csrMtxA_h, mtxSolX_d, mtxB_d, M, N, bmBFBCG);

    if(debug){
        printf("\n\n~~📝📝📝Approximate Solution Marix📝📝📝~~\n\n");
        print_mtx_clm_d(mtxSolX_d, K, N);
    }

    //Validate
    //R = B - AX
    if(debug){
        printf("\n\n\n\n🔍👀Validate Solution Matrix X 🔍👀");
        double twoNorms = validateBFBCG(csrMtxA_h, M, mtxSolX_d, N, mtxB_d);
        printf("\n\n~~~ mtxR = B - AX_sol, 1st Column Vector 2 norms in mtxR : %f ~~~\n\n", twoNorms);
    }
    
    //Free memeory
    freeCSRMatrix(csrMtxA_h);
    CHECK(cudaFree(mtxSolX_d));
    CHECK(cudaFree(mtxB_d));


} // end of tranposeTest_Case1






