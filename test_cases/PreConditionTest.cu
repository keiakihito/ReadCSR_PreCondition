// includes, system
#include<stdio.h>
#include<stdlib.h>
#include<string.h>

/*Using updated (v2) interfaces to cublas*/
#include<cublas_v2.h>
#include<cuda_runtime.h>
#include<cusparse.h>
#include <cusolverDn.h>
#include<sys/time.h>


//Utilities
// helper function CUDA error checking and initialization 
#include "../include/functions/orth_SVD.h"
#include "../include/struct/CSRMatrix.h"



void preCondition_test1();



int main(int argc, char** argv)
{
    printf("\n\n~~preCondition_Test()~~\n\n");

    printf("\n\n🔍🔍🔍Test case 1🔍🔍🔍\n");
    preCondition_test1();


    printf("\n= = = End of orth_test  = = = \n\n");


}// end of main


void preCondition_test1()
{   
    bool debug = true;
    cusparseHandle_t cusparseHandler = NULL;
    CHECK_CUSPARSE(cusparseCreate(&cusparseHandler));
    /*
    A = sparse([4 -1 0 -1 0 0;
                -1 4 -1 0 -1 0;
                0 -1 4 0 0 -1;
                -1 0 0 4 -1 0;
                0 -1 0 -1 4 -1;
                0 0 -1 0 -1 4]);
    */

    // Host CSR arrays
    int row_offsets[] = {0, 3, 7, 10, 13, 17, 20};
    int col_indices[] = {0, 1, 3, 0, 1, 2, 4, 1, 2, 5, 0, 3, 4, 1, 3, 4, 5, 2, 4, 5};
    double vals[] = {4, -1, -1, -1, 4, -1, -1, -1, 4, -1, -1, 4, -1, -1, -1, 4, -1, -1, 4, -1, 4};

    int N = 6; // Number of rows/columns
    int nnz = 20; // Number of non-zero elements

    CSRMatrix mtxA = constructCSRMatrix(N, N, nnz, row_offsets, col_indices, vals);
    if(debug){
        printf("\n\n~~mtxA~~\n\n");
        print_CSRMtx(mtxA);
    }

    CSRMatrix mtxM = constructPreConditionMatrixCSR(cusparseHandler, mtxA);
    if(debug){
        printf("\n\n~~mtxM~~\n\n");
        print_CSRMtx(mtxM);
    }


} // end of orth_QRtest1()
