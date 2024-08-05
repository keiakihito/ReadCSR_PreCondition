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
#include "../include/functions/orth_QR.h"
#include "../include/functions/orth_SVD.h"



void orth_QRtest1();
void orth_QRtest2();
void orth_QRtest3();
void orth_QRtest4();
void orth_QRtest5();

void orth_SVDtest1();
void orth_SVDtest2();
void orth_SVDtest3();
void orth_SVDtest4();
void orth_SVDtest5();

int main(int argc, char** argv)
{
    printf("\n\n~~orth_QR_Test()~~\n\n");

    printf("\n\n🔍🔍🔍Test case 1🔍🔍🔍\n");
    printf("\n\n = = 👀orth_QR Test👀 = = \n");
    orth_QRtest1();
    printf("\n\n = = 👀orth_SVD Test👀 = =\n");
    orth_SVDtest1();
    
    // printf("\n\n🔍🔍🔍Test case 2🔍🔍🔍\n");
    // printf("\n\n = = 👀orth_QR Test👀 = = \n");
    // orth_QRtest2();
    // printf("\n\n = = 👀orth_SVD Test👀 = =\n");
    // orth_SVDtest2();
    

    // printf("\n\n🔍🔍🔍Test case 3🔍🔍🔍\n");
    // printf("\n\n = = 👀orth_QR Test👀 = = \n");
    // orth_QRtest3();
    // printf("\n\n = = 👀orth_SVD Test👀 = =\n");
    // orth_SVDtest3();
    

    // printf("\n\n🔍🔍🔍Test case 4🔍🔍🔍\n");
    // printf("\n\n = = 👀orth_QR Test👀 = = \n");
    // orth_QRtest4();
    // printf("\n\n = = 👀orth_SVD Test👀 = =\n");
    // orth_SVDtest4();
    

    // printf("\n\n🔍🔍🔍Test case 5🔍🔍🔍\n");
    // printf("\n\n = = 👀orth_QR Test👀 = = \n");
    // orth_QRtest5();
    // printf("\n\n = = 👀orth_SVD Test👀 = =\n");
    // orth_SVDtest5();
    

    printf("\n= = = End of orth_test  = = = \n\n");


}// end of main


void orth_QRtest1()
{
       /*
    Z = | 1.0  5.0  9.0 |
        | 2.0  6.0  10.0|
        | 3.0  7.0  11.0|
        | 4.0  8.0  12.0| 
    */

    // Define the dense matrixB column major
    double mtxZ[] = {
    1.0, 2.0, 3.0, 4.0,
    5.0, 6.0, 7.0, 8.0,
    9.0, 10.0, 11.0, 12.0,
    };

    int numOfRow = 4;
    int numOfClm = 3;
    int crntRank = 3;

    double* mtxZ_d = NULL;
    double* mtxQ_trnc_d = NULL;
    double* mtxI_d = NULL;

    bool debug = true;

    cublasHandle_t cublasHandler = NULL;
	CHECK_CUBLAS(cublasCreate(&cublasHandler));


    CHECK(cudaMalloc((void**)&mtxZ_d, numOfRow * numOfClm * sizeof(double)));
    CHECK(cudaMemcpy(mtxZ_d, mtxZ, numOfRow * numOfClm * sizeof(double), cudaMemcpyHostToDevice));

    if(debug){
        printf("\n\n~~mtxZ~~\n\n");
        print_mtx_clm_d(mtxZ_d, numOfRow, numOfClm);
    }


    orth_QR(&mtxQ_trnc_d, mtxZ_d, numOfRow, numOfClm, crntRank);

    printf("\n\n~~Outside function: mtxQ_trunc_d~~\n\n");
    print_mtx_clm_d(mtxQ_trnc_d, numOfRow, crntRank);
    printf("\n\n~~Current Rarnk = %d~~\n\n", crntRank);

    printf("\n\n= = 🔍Check Orthogonality🔍 = = \n\n");
    CHECK(cudaMalloc((void**)&mtxI_d, crntRank * crntRank * sizeof(double)));
    multiply_Den_ClmM_mtxT_mtx(cublasHandler, mtxQ_trnc_d, mtxI_d, numOfRow, crntRank);
    print_mtx_clm_d(mtxI_d, crntRank, crntRank);



} // end of orth_QRtest1()

void orth_QRtest2()
{
 // Define the dense matrixB column major
    double mtxZ[] = {
    1.1, 0.8, 3.0, 2.2, 0.2, 0.7,
    2.2, 1.6, 4.1, 3.3, 0.3, 0.8,
    3.3, 2.4, 5.2, 4.4, 0.4, 1.1,
    4.4, 3.2, 6.3, 5.5, 0.5, 1.5,
    5.5, 2.3, 0.7, 1.7, 0.6, 3.2
    };

    int numOfRow = 6;
    int numOfClm = 5;
    int crntRank = 5;

    double* mtxZ_d = NULL;
    double* mtxQ_trnc_d = NULL;
    double* mtxI_d = NULL;

    bool debug = true;

    cublasHandle_t cublasHandler = NULL;
	CHECK_CUBLAS(cublasCreate(&cublasHandler));


    CHECK(cudaMalloc((void**)&mtxZ_d, numOfRow * numOfClm * sizeof(double)));
    CHECK(cudaMemcpy(mtxZ_d, mtxZ, numOfRow * numOfClm * sizeof(double), cudaMemcpyHostToDevice));

    if(debug){
        printf("\n\n~~mtxZ~~\n\n");
        print_mtx_clm_d(mtxZ_d, numOfRow, numOfClm);
    }


    orth_QR(&mtxQ_trnc_d, mtxZ_d, numOfRow, numOfClm, crntRank);

    printf("\n\n~~Outside function: mtxQ_trunc_d~~\n\n");
    print_mtx_clm_d(mtxQ_trnc_d, numOfRow, crntRank);
    printf("\n\n~~Current Rarnk = %d~~\n\n", crntRank);

    printf("\n\n= = 🔍Check Orthogonality🔍 = = \n\n");
    CHECK(cudaMalloc((void**)&mtxI_d, crntRank * crntRank * sizeof(double)));
    multiply_Den_ClmM_mtxT_mtx(cublasHandler, mtxQ_trnc_d, mtxI_d, numOfRow, crntRank);
    print_mtx_clm_d(mtxI_d, crntRank, crntRank);

} // end of orth_QRtest1()

void orth_QRtest3()
{
// Define the dense matrixB column major
    double mtxZ[] = {
        1.1, 0.8, 3.0, 2.2, 1.4, 0.6, 7.7,
        2.2, 1.6, 4.1, 3.3, 2.5, 1.5, 5.6,
        3.3, 2.4, 5.2, 4.4, 3.6, 2.4, 9.6,
        4.4, 3.2, 6.3, 5.5, 4.7, 3.3, 8.8,
        5.5, 4.0, 7.4, 6.6, 5.8, 4.2, 7.0
    };

    int numOfRow = 7;
    int numOfClm = 5;
    int crntRank = 5;

        double* mtxZ_d = NULL;
    double* mtxQ_trnc_d = NULL;
    double* mtxI_d = NULL;

    bool debug = true;

    cublasHandle_t cublasHandler = NULL;
	CHECK_CUBLAS(cublasCreate(&cublasHandler));


    CHECK(cudaMalloc((void**)&mtxZ_d, numOfRow * numOfClm * sizeof(double)));
    CHECK(cudaMemcpy(mtxZ_d, mtxZ, numOfRow * numOfClm * sizeof(double), cudaMemcpyHostToDevice));

    if(debug){
        printf("\n\n~~mtxZ~~\n\n");
        print_mtx_clm_d(mtxZ_d, numOfRow, numOfClm);
    }


    orth_QR(&mtxQ_trnc_d, mtxZ_d, numOfRow, numOfClm, crntRank);

    printf("\n\n~~Outside function: mtxQ_trunc_d~~\n\n");
    print_mtx_clm_d(mtxQ_trnc_d, numOfRow, crntRank);
    printf("\n\n~~Current Rarnk = %d~~\n\n", crntRank);

    printf("\n\n= = 🔍Check Orthogonality🔍 = = \n\n");
    CHECK(cudaMalloc((void**)&mtxI_d, crntRank * crntRank * sizeof(double)));
    multiply_Den_ClmM_mtxT_mtx(cublasHandler, mtxQ_trnc_d, mtxI_d, numOfRow, crntRank);
    print_mtx_clm_d(mtxI_d, crntRank, crntRank);


} // end of orth_QRtest1()

void orth_QRtest4()
{
 // Define the dense matrixB column major
    double mtxZ[] = {
        1.2, 0.9, 3.1, 2.3, 1.5, 0.7, 2.1, 3.3, 0.1, 0.2, 0.3, 
        2.3, 1.8, 4.2, 3.4, 2.6, 1.6, 3.0, 4.4, 0.5, 0.7, 0.2, 
        3.4, 2.7, 5.3, 4.5, 3.7, 2.5, 4.9, 5.5, 0.3, 0.4, 0.5, 
        4.5, 3.6, 6.4, 5.6, 4.8, 3.4, 5.8, 6.6, 1.1, 1.2, 1.3, 
        5.6, 4.5, 7.5, 6.7, 5.9, 4.3, 6.7, 7.7, 1.9, 1.5, 1.8, 
        6.7, 5.4, 8.6, 7.8, 6.0, 5.2, 7.6, 8.8, 2.2, 2.3, 2.5, 
        7.8, 6.3, 9.7, 8.9, 7.1, 6.1, 8.5, 9.9, 2.9, 3.1, 3.2
    };


    int numOfRow = 11;
    int numOfClm = 7;
    int crntRank = 7;

    double* mtxZ_d = NULL;
    double* mtxQ_trnc_d = NULL;
    double* mtxI_d = NULL;

    bool debug = true;

    cublasHandle_t cublasHandler = NULL;
	CHECK_CUBLAS(cublasCreate(&cublasHandler));


    CHECK(cudaMalloc((void**)&mtxZ_d, numOfRow * numOfClm * sizeof(double)));
    CHECK(cudaMemcpy(mtxZ_d, mtxZ, numOfRow * numOfClm * sizeof(double), cudaMemcpyHostToDevice));

    if(debug){
        printf("\n\n~~mtxZ~~\n\n");
        print_mtx_clm_d(mtxZ_d, numOfRow, numOfClm);
    }


    orth_QR(&mtxQ_trnc_d, mtxZ_d, numOfRow, numOfClm, crntRank);

    printf("\n\n~~Outside function: mtxQ_trunc_d~~\n\n");
    print_mtx_clm_d(mtxQ_trnc_d, numOfRow, crntRank);
    printf("\n\n~~Current Rarnk = %d~~\n\n", crntRank);

    printf("\n\n= = 🔍Check Orthogonality🔍 = = \n\n");
    CHECK(cudaMalloc((void**)&mtxI_d, crntRank * crntRank * sizeof(double)));
    multiply_Den_ClmM_mtxT_mtx(cublasHandler, mtxQ_trnc_d, mtxI_d, numOfRow, crntRank);
    print_mtx_clm_d(mtxI_d, crntRank, crntRank);



} // end of orth_QRtest1()

void orth_QRtest5()
{
    // Define the dense matrixB column major
    // Define the dense matrixB column major
    double mtxZ[] = {
        1.2, 0.9, 3.1, 2.3, 1.5, 0.7, 2.1, 3.3, 2.5, 1.9, 0.9, 1.0, 1.1, 1.2, 1.3, 7.7, 6.4, 8.6, 8.8, 6.0,
        2.3, 1.8, 4.2, 3.4, 2.6, 1.6, 3.0, 4.4, 3.6, 2.8, 1.8, 2.1, 2.4, 2.6, 2.8, 6.2, 7.6, 8.8, 7.0, 6.4,
        3.4, 2.7, 5.3, 4.5, 3.7, 2.5, 4.9, 5.5, 4.7, 3.7, 2.7, 3.2, 3.7, 4.0, 4.3, 5.4, 6.5, 7.6, 8.2, 8.8,
        4.5, 3.6, 6.4, 5.6, 4.8, 3.4, 5.8, 6.6, 5.8, 4.6, 3.6, 4.3, 5.0, 5.4, 5.8, 8.8, 7.3, 9.7, 9.9, 7.1,
        5.6, 4.5, 7.5, 6.7, 5.9, 4.3, 6.7, 7.7, 6.9, 5.5, 4.5, 5.4, 6.3, 6.8, 7.3, 7.1, 8.5, 9.9, 8.1, 7.3,
        6.7, 5.4, 8.6, 7.8, 6.0, 5.2, 7.6, 8.8, 7.0, 6.4, 5.4, 6.5, 7.6, 8.2, 8.8, 6.3, 7.6, 8.9, 9.6, 10.3,
        7.8, 6.3, 9.7, 8.9, 7.1, 6.1, 8.5, 9.9, 8.1, 7.3, 6.3, 7.6, 8.9, 9.6, 10.3, 9.9, 8.2, 0.8, 9.0, 8.2,
        8.9, 7.2, 0.8, 9.0, 8.2, 7.0, 9.4, 0.1, 9.2, 8.2, 7.2, 8.7, 0.2, 0.5, 1.8, 8.0, 9.4, 0.3, 9.2, 8.2,
        9.0, 8.1, 1.9, 1.1, 9.3, 8.9, 0.3, 1.1, 0.3, 9.1, 8.1, 9.8, 1.5, 1.4, 2.4, 7.2, 8.7, 0.2, 0.4, 1.8,
        1.1, 9.0, 2.0, 2.2, 0.4, 9.8, 1.2, 2.2, 1.4, 0.2, 9.0, 0.9, 2.8, 2.8, 3.0, 1.0, 9.1, 1.9, 2.2, 0.4,
        2.2, 1.9, 3.1, 3.3, 1.5, 1.7, 2.1, 3.3, 2.5, 1.9, 0.9, 1.0, 1.1, 1.2, 1.3, 9.9, 1.2, 2.2, 1.4, 0.7,
        3.3, 2.8, 4.2, 4.4, 2.6, 2.6, 3.0, 4.4, 3.6, 2.8, 1.8, 2.1, 2.4, 2.6, 2.8, 9.0, 0.9, 2.8, 2.8, 3.0,
        4.4, 3.7, 5.3, 5.5, 3.7, 3.5, 4.9, 5.5, 4.7, 3.7, 2.7, 3.2, 3.7, 4.0, 4.3, 2.1, 0.7, 2.0, 3.2, 1.5,
        5.5, 4.6, 6.4, 6.6, 4.8, 4.4, 5.8, 6.6, 5.8, 4.6, 3.6, 4.3, 5.0, 5.4, 5.8, 0.8, 2.1, 3.2, 2.3, 1.0,
        6.6, 5.5, 7.5, 7.7, 5.9, 5.3, 6.7, 7.7, 6.9, 5.5, 4.5, 5.4, 6.3, 6.8, 7.3, 0.9, 1.0, 1.1, 1.2, 1.3
    };
    int numOfRow = 20;
    int numOfClm = 15;
    int crntRank = 15;

        double* mtxZ_d = NULL;
    double* mtxQ_trnc_d = NULL;
    double* mtxI_d = NULL;

    bool debug = true;

    cublasHandle_t cublasHandler = NULL;
	CHECK_CUBLAS(cublasCreate(&cublasHandler));


    CHECK(cudaMalloc((void**)&mtxZ_d, numOfRow * numOfClm * sizeof(double)));
    CHECK(cudaMemcpy(mtxZ_d, mtxZ, numOfRow * numOfClm * sizeof(double), cudaMemcpyHostToDevice));

    if(debug){
        printf("\n\n~~mtxZ~~\n\n");
        print_mtx_clm_d(mtxZ_d, numOfRow, numOfClm);
    }


    orth_QR(&mtxQ_trnc_d, mtxZ_d, numOfRow, numOfClm, crntRank);

    printf("\n\n~~Outside function: mtxQ_trunc_d~~\n\n");
    print_mtx_clm_d(mtxQ_trnc_d, numOfRow, crntRank);
    printf("\n\n~~Current Rarnk = %d~~\n\n", crntRank);

    printf("\n\n= = 🔍Check Orthogonality🔍 = = \n\n");
    CHECK(cudaMalloc((void**)&mtxI_d, crntRank * crntRank * sizeof(double)));
    multiply_Den_ClmM_mtxT_mtx(cublasHandler, mtxQ_trnc_d, mtxI_d, numOfRow, crntRank);
    print_mtx_clm_d(mtxI_d, crntRank, crntRank);


} // end of orth_QRtest1()





void orth_SVDtest1()
{
       /*
    Z = | 1.0  5.0  9.0 |
        | 2.0  6.0  10.0|
        | 3.0  7.0  11.0|
        | 4.0  8.0  12.0| 
    */

    // Define the dense matrixB column major
    double mtxZ[] = {
    1.0, 2.0, 3.0, 4.0,
    5.0, 6.0, 7.0, 8.0,
    9.0, 10.0, 11.0, 12.0,
    };

    int numOfRow = 4;
    int numOfClm = 3;
    int crntRank = 3;

    double* mtxZ_d = NULL;
    double* mtxQ_trnc_d = NULL;
    double* mtxI_d = NULL;

    bool debug = true;

    cublasHandle_t cublasHandler = NULL;
	CHECK_CUBLAS(cublasCreate(&cublasHandler));


    CHECK(cudaMalloc((void**)&mtxZ_d, numOfRow * numOfClm * sizeof(double)));
    CHECK(cudaMemcpy(mtxZ_d, mtxZ, numOfRow * numOfClm * sizeof(double), cudaMemcpyHostToDevice));

    if(debug){
        printf("\n\n~~mtxZ~~\n\n");
        print_mtx_clm_d(mtxZ_d, numOfRow, numOfClm);
    }


    // orth_QR(&mtxQ_trnc_d, mtxZ_d, numOfRow, numOfClm, crntRank);
    orth_SVD(&mtxQ_trnc_d, mtxZ_d, numOfRow, numOfClm, crntRank);

    printf("\n\n~~Outside function: mtxQ_trunc_d~~\n\n");
    print_mtx_clm_d(mtxQ_trnc_d, numOfRow, crntRank);
    printf("\n\n~~Current Rarnk = %d~~\n\n", crntRank);

    printf("\n\n= = 🔍Check Orthogonality🔍 = = \n\n");
    CHECK(cudaMalloc((void**)&mtxI_d, crntRank * crntRank * sizeof(double)));
    multiply_Den_ClmM_mtxT_mtx(cublasHandler, mtxQ_trnc_d, mtxI_d, numOfRow, crntRank);
    print_mtx_clm_d(mtxI_d, crntRank, crntRank);



} // end of orth_QRtest1()

void orth_SVDtest2()
{
 // Define the dense matrixB column major
    double mtxZ[] = {
    1.1, 0.8, 3.0, 2.2, 0.2, 0.7,
    2.2, 1.6, 4.1, 3.3, 0.3, 0.8,
    3.3, 2.4, 5.2, 4.4, 0.4, 1.1,
    4.4, 3.2, 6.3, 5.5, 0.5, 1.5,
    5.5, 2.3, 0.7, 1.7, 0.6, 3.2
    };

    int numOfRow = 6;
    int numOfClm = 5;
    int crntRank = 5;

    double* mtxZ_d = NULL;
    double* mtxQ_trnc_d = NULL;
    double* mtxI_d = NULL;

    bool debug = true;

    cublasHandle_t cublasHandler = NULL;
	CHECK_CUBLAS(cublasCreate(&cublasHandler));


    CHECK(cudaMalloc((void**)&mtxZ_d, numOfRow * numOfClm * sizeof(double)));
    CHECK(cudaMemcpy(mtxZ_d, mtxZ, numOfRow * numOfClm * sizeof(double), cudaMemcpyHostToDevice));

    if(debug){
        printf("\n\n~~mtxZ~~\n\n");
        print_mtx_clm_d(mtxZ_d, numOfRow, numOfClm);
    }


    orth_SVD(&mtxQ_trnc_d, mtxZ_d, numOfRow, numOfClm, crntRank);

    printf("\n\n~~Outside function: mtxQ_trunc_d~~\n\n");
    print_mtx_clm_d(mtxQ_trnc_d, numOfRow, crntRank);
    printf("\n\n~~Current Rarnk = %d~~\n\n", crntRank);

    printf("\n\n= = 🔍Check Orthogonality🔍 = = \n\n");
    CHECK(cudaMalloc((void**)&mtxI_d, crntRank * crntRank * sizeof(double)));
    multiply_Den_ClmM_mtxT_mtx(cublasHandler, mtxQ_trnc_d, mtxI_d, numOfRow, crntRank);
    print_mtx_clm_d(mtxI_d, crntRank, crntRank);

} // end of orth_QRtest1()

void orth_SVDtest3()
{
// Define the dense matrixB column major
    double mtxZ[] = {
        1.1, 0.8, 3.0, 2.2, 1.4, 0.6, 7.7,
        2.2, 1.6, 4.1, 3.3, 2.5, 1.5, 5.6,
        3.3, 2.4, 5.2, 4.4, 3.6, 2.4, 9.6,
        4.4, 3.2, 6.3, 5.5, 4.7, 3.3, 8.8,
        5.5, 4.0, 7.4, 6.6, 5.8, 4.2, 7.0
    };

    int numOfRow = 7;
    int numOfClm = 5;
    int crntRank = 5;

        double* mtxZ_d = NULL;
    double* mtxQ_trnc_d = NULL;
    double* mtxI_d = NULL;

    bool debug = true;

    cublasHandle_t cublasHandler = NULL;
	CHECK_CUBLAS(cublasCreate(&cublasHandler));


    CHECK(cudaMalloc((void**)&mtxZ_d, numOfRow * numOfClm * sizeof(double)));
    CHECK(cudaMemcpy(mtxZ_d, mtxZ, numOfRow * numOfClm * sizeof(double), cudaMemcpyHostToDevice));

    if(debug){
        printf("\n\n~~mtxZ~~\n\n");
        print_mtx_clm_d(mtxZ_d, numOfRow, numOfClm);
    }


    orth_SVD(&mtxQ_trnc_d, mtxZ_d, numOfRow, numOfClm, crntRank);

    printf("\n\n~~Outside function: mtxQ_trunc_d~~\n\n");
    print_mtx_clm_d(mtxQ_trnc_d, numOfRow, crntRank);
    printf("\n\n~~Current Rarnk = %d~~\n\n", crntRank);

    printf("\n\n= = 🔍Check Orthogonality🔍 = = \n\n");
    CHECK(cudaMalloc((void**)&mtxI_d, crntRank * crntRank * sizeof(double)));
    multiply_Den_ClmM_mtxT_mtx(cublasHandler, mtxQ_trnc_d, mtxI_d, numOfRow, crntRank);
    print_mtx_clm_d(mtxI_d, crntRank, crntRank);


} // end of orth_QRtest1()

void orth_SVDtest4()
{
 // Define the dense matrixB column major
    double mtxZ[] = {
        1.2, 0.9, 3.1, 2.3, 1.5, 0.7, 2.1, 3.3, 0.1, 0.2, 0.3, 
        2.3, 1.8, 4.2, 3.4, 2.6, 1.6, 3.0, 4.4, 0.5, 0.7, 0.2, 
        3.4, 2.7, 5.3, 4.5, 3.7, 2.5, 4.9, 5.5, 0.3, 0.4, 0.5, 
        4.5, 3.6, 6.4, 5.6, 4.8, 3.4, 5.8, 6.6, 1.1, 1.2, 1.3, 
        5.6, 4.5, 7.5, 6.7, 5.9, 4.3, 6.7, 7.7, 1.9, 1.5, 1.8, 
        6.7, 5.4, 8.6, 7.8, 6.0, 5.2, 7.6, 8.8, 2.2, 2.3, 2.5, 
        7.8, 6.3, 9.7, 8.9, 7.1, 6.1, 8.5, 9.9, 2.9, 3.1, 3.2
    };


    int numOfRow = 11;
    int numOfClm = 7;
    int crntRank = 7;

    double* mtxZ_d = NULL;
    double* mtxQ_trnc_d = NULL;
    double* mtxI_d = NULL;

    bool debug = true;

    cublasHandle_t cublasHandler = NULL;
	CHECK_CUBLAS(cublasCreate(&cublasHandler));


    CHECK(cudaMalloc((void**)&mtxZ_d, numOfRow * numOfClm * sizeof(double)));
    CHECK(cudaMemcpy(mtxZ_d, mtxZ, numOfRow * numOfClm * sizeof(double), cudaMemcpyHostToDevice));

    if(debug){
        printf("\n\n~~mtxZ~~\n\n");
        print_mtx_clm_d(mtxZ_d, numOfRow, numOfClm);
    }


    orth_SVD(&mtxQ_trnc_d, mtxZ_d, numOfRow, numOfClm, crntRank);

    printf("\n\n~~Outside function: mtxQ_trunc_d~~\n\n");
    print_mtx_clm_d(mtxQ_trnc_d, numOfRow, crntRank);
    printf("\n\n~~Current Rarnk = %d~~\n\n", crntRank);

    printf("\n\n= = 🔍Check Orthogonality🔍 = = \n\n");
    CHECK(cudaMalloc((void**)&mtxI_d, crntRank * crntRank * sizeof(double)));
    multiply_Den_ClmM_mtxT_mtx(cublasHandler, mtxQ_trnc_d, mtxI_d, numOfRow, crntRank);
    print_mtx_clm_d(mtxI_d, crntRank, crntRank);



} // end of orth_QRtest1()

void orth_SVDtest5()
{
    // Define the dense matrixB column major
    // Define the dense matrixB column major
    double mtxZ[] = {
        1.2, 0.9, 3.1, 2.3, 1.5, 0.7, 2.1, 3.3, 2.5, 1.9, 0.9, 1.0, 1.1, 1.2, 1.3, 7.7, 6.4, 8.6, 8.8, 6.0,
        2.3, 1.8, 4.2, 3.4, 2.6, 1.6, 3.0, 4.4, 3.6, 2.8, 1.8, 2.1, 2.4, 2.6, 2.8, 6.2, 7.6, 8.8, 7.0, 6.4,
        3.4, 2.7, 5.3, 4.5, 3.7, 2.5, 4.9, 5.5, 4.7, 3.7, 2.7, 3.2, 3.7, 4.0, 4.3, 5.4, 6.5, 7.6, 8.2, 8.8,
        4.5, 3.6, 6.4, 5.6, 4.8, 3.4, 5.8, 6.6, 5.8, 4.6, 3.6, 4.3, 5.0, 5.4, 5.8, 8.8, 7.3, 9.7, 9.9, 7.1,
        5.6, 4.5, 7.5, 6.7, 5.9, 4.3, 6.7, 7.7, 6.9, 5.5, 4.5, 5.4, 6.3, 6.8, 7.3, 7.1, 8.5, 9.9, 8.1, 7.3,
        6.7, 5.4, 8.6, 7.8, 6.0, 5.2, 7.6, 8.8, 7.0, 6.4, 5.4, 6.5, 7.6, 8.2, 8.8, 6.3, 7.6, 8.9, 9.6, 10.3,
        7.8, 6.3, 9.7, 8.9, 7.1, 6.1, 8.5, 9.9, 8.1, 7.3, 6.3, 7.6, 8.9, 9.6, 10.3, 9.9, 8.2, 0.8, 9.0, 8.2,
        8.9, 7.2, 0.8, 9.0, 8.2, 7.0, 9.4, 0.1, 9.2, 8.2, 7.2, 8.7, 0.2, 0.5, 1.8, 8.0, 9.4, 0.3, 9.2, 8.2,
        9.0, 8.1, 1.9, 1.1, 9.3, 8.9, 0.3, 1.1, 0.3, 9.1, 8.1, 9.8, 1.5, 1.4, 2.4, 7.2, 8.7, 0.2, 0.4, 1.8,
        1.1, 9.0, 2.0, 2.2, 0.4, 9.8, 1.2, 2.2, 1.4, 0.2, 9.0, 0.9, 2.8, 2.8, 3.0, 1.0, 9.1, 1.9, 2.2, 0.4,
        2.2, 1.9, 3.1, 3.3, 1.5, 1.7, 2.1, 3.3, 2.5, 1.9, 0.9, 1.0, 1.1, 1.2, 1.3, 9.9, 1.2, 2.2, 1.4, 0.7,
        3.3, 2.8, 4.2, 4.4, 2.6, 2.6, 3.0, 4.4, 3.6, 2.8, 1.8, 2.1, 2.4, 2.6, 2.8, 9.0, 0.9, 2.8, 2.8, 3.0,
        4.4, 3.7, 5.3, 5.5, 3.7, 3.5, 4.9, 5.5, 4.7, 3.7, 2.7, 3.2, 3.7, 4.0, 4.3, 2.1, 0.7, 2.0, 3.2, 1.5,
        5.5, 4.6, 6.4, 6.6, 4.8, 4.4, 5.8, 6.6, 5.8, 4.6, 3.6, 4.3, 5.0, 5.4, 5.8, 0.8, 2.1, 3.2, 2.3, 1.0,
        6.6, 5.5, 7.5, 7.7, 5.9, 5.3, 6.7, 7.7, 6.9, 5.5, 4.5, 5.4, 6.3, 6.8, 7.3, 0.9, 1.0, 1.1, 1.2, 1.3
    };
    int numOfRow = 20;
    int numOfClm = 15;
    int crntRank = 15;

        double* mtxZ_d = NULL;
    double* mtxQ_trnc_d = NULL;
    double* mtxI_d = NULL;

    bool debug = true;

    cublasHandle_t cublasHandler = NULL;
	CHECK_CUBLAS(cublasCreate(&cublasHandler));


    CHECK(cudaMalloc((void**)&mtxZ_d, numOfRow * numOfClm * sizeof(double)));
    CHECK(cudaMemcpy(mtxZ_d, mtxZ, numOfRow * numOfClm * sizeof(double), cudaMemcpyHostToDevice));

    if(debug){
        printf("\n\n~~mtxZ~~\n\n");
        print_mtx_clm_d(mtxZ_d, numOfRow, numOfClm);
    }


    orth_SVD(&mtxQ_trnc_d, mtxZ_d, numOfRow, numOfClm, crntRank);

    printf("\n\n~~Outside function: mtxQ_trunc_d~~\n\n");
    print_mtx_clm_d(mtxQ_trnc_d, numOfRow, crntRank);
    printf("\n\n~~Current Rarnk = %d~~\n\n", crntRank);

    printf("\n\n= = 🔍Check Orthogonality🔍 = = \n\n");
    CHECK(cudaMalloc((void**)&mtxI_d, crntRank * crntRank * sizeof(double)));
    multiply_Den_ClmM_mtxT_mtx(cublasHandler, mtxQ_trnc_d, mtxI_d, numOfRow, crntRank);
    print_mtx_clm_d(mtxI_d, crntRank, crntRank);


} // end of orth_QRtest1()



/*
Sample Run
~~orth_QR_Test()~~



🔍🔍🔍Test case 1🔍🔍🔍


 = = 👀orth_QR Test👀 = = 


~~mtxZ~~

1.000000 5.000000 9.000000 
2.000000 6.000000 10.000000 
3.000000 7.000000 11.000000 
4.000000 8.000000 12.000000 


~~Outside function: mtxQ_trunc_d~~

-0.426162 -0.719990 
-0.473514 -0.275290 
-0.520865 0.169409 
-0.568216 0.614109 


~~Current Rarnk = 2~~



= = 🔍Check Orthogonality🔍 = = 

1.000000 0.000000 
0.000000 1.000000 


 = = 👀orth_SVD Test👀 = =


~~mtxZ~~

1.000000 5.000000 9.000000 
2.000000 6.000000 10.000000 
3.000000 7.000000 11.000000 
4.000000 8.000000 12.000000 


~~Outside function: mtxQ_trunc_d~~

-0.403618 -0.732866 
-0.464744 -0.289850 
-0.525871 0.153167 
-0.586997 0.596183 


~~Current Rarnk = 2~~



= = 🔍Check Orthogonality🔍 = = 

1.000000 0.000000 
0.000000 1.000000 

= = = End of orth_test  = = = 

 */