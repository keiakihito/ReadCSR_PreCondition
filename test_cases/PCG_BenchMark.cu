// includes, system
#include<stdio.h>
#include<stdlib.h>
#include<string.h>

/*Using updated (v2) interfaces to cublas*/
#include<cublas_v2.h>
#include<cuda_runtime.h>
#include<cusparse.h>
#include<sys/time.h>


//Utilities


//After refactor header files
// #include "include/utils/checks.h"
// #include "include/functions/helper.h"
// #include "include/functions/cuBLAS_util.h"
// #include "include/functions/cuSPARSE_util.h"
// #include "include/functions/cuSOLVER_util.h"

// #include "include/CSRMatrix.h"


#include "../include/struct/BenchMarkPCG.h"
#include "../include/functions/pcg.h"


void pcgTest_Case(BenchMarkPCG& bmPCG, int numOfA);


int main(int argc, char** argv)
{   
    printf("\n\n= = = =Benchmark pcg each operation in 1st iteration= = = = \n\n");
    const int NUM_OF_ITERATION = 51; //Eliminate 1st iteration result


    BenchMarkPCG bmPCG; // Store all the results
    std::vector<double> averages; // Averages time for each opeartion

    //Sparse size 2^5 through 2^15
    for(int i = 5; i < 16; i++){        
        for(int j = 0; j < NUM_OF_ITERATION; j++){
            pcgTest_Case(bmPCG, (int)pow(2, i));        
        }
        eliminateFirst(bmPCG); // Get rid of the first result
        averages = getAverages(bmPCG); // Calculate each operatoin average time
        printf("\n\n~~ matrix A 2^%d average time~~\n", i);
        printResult(averages); //Display result

        averages.clear(); // Clear vectors for the next average calculation
        clearBenchMarkPCG(bmPCG); // Rest all the vectors in the struct for the next calculation
    } // end of for 


    printf("\n\n= = = = End of Benchmark PCG  = = = =\n\n");

    return 0;
} // end of main



void pcgTest_Case(BenchMarkPCG& bmPCG, int numOfA)
{
    cudaDeviceSynchronize();
    bool debug = false;
    
    const int N = numOfA;
    CSRMatrix csrMtxA_h = generateSparseSPDMatrixCSR(N);


    double mtxB_h[N];
    initializeRandom(mtxB_h, N, 1);

    //(1) Allocate memory
    double* mtxSolX_d = NULL;
    double* mtxB_d = NULL;

    CHECK(cudaMalloc((void**)&mtxSolX_d,  N * sizeof(double)));
    CHECK(cudaMalloc((void**)&mtxB_d,  N * sizeof(double)));

    //(2) Copy value from host to device
    CHECK(cudaMemcpy(mtxSolX_d, mtxB_h, N * sizeof(double), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(mtxB_d, mtxB_h, N * sizeof(double), cudaMemcpyHostToDevice));

    if(debug){
        printf("\n\n~~csrMtxA_h~~\n\n");
        print_CSRMtx(csrMtxA_h);
        printf("\n\n~~mtxSolX~~\n\n");
        print_vector(mtxSolX_d, N);
        printf("\n\n~~mtxB~~\n\n");
        print_vector(mtxB_d, N);
    }

    //Solve AX = B with PCG method
    pcg_BenchMark(csrMtxA_h, mtxSolX_d, mtxB_d, N, bmPCG);


    if(debug){
        printf("\n\n~~📝📝📝Approximate Solution Vector X📝📝📝~~\n\n");
        print_vector(mtxSolX_d, N);
            
        //Validate with r - b -Ax with 2 Norm
        printf("\n\n~~🔍👀Validate Solution vector X 🔍👀~~");
        double twoNorms = validateCG(csrMtxA_h, N, mtxSolX_d, mtxB_d);
        printf("\n\n~~Valicate : r = b - A * x_sol ~~ \n = =  vector r 2 norms: %f = =\n\n", twoNorms);
    }


    //()Free memeory
    freeCSRMatrix(csrMtxA_h);
    CHECK(cudaFree(mtxSolX_d));
    CHECK(cudaFree(mtxB_d));



} // end of tranposeTest_Case4
