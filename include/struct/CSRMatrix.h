#ifndef CSRMatrix_H
#define CSRMatrix_H

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cusparse_v2.h>
#include <time.h>

#include "../functions/helper.h"
// #include "../functions/cuBLAS_util.h"
// #include "../functions/cuSPARSE_util.h"
// #include "../functions/cuSOLVER_util.h"

#define max(a,b)((a)>(b) ? (a):(b))

// CSR Matrix
struct CSRMatrix{
    int numOfRows;
    int numOfClms;
    int numOfnz;
    int *row_offsets;
    int *col_indices;
    double *vals;
};


CSRMatrix constructCSRMatrix(int numOfRow, int numOfClm, int nnz, int* row_offsets, int* col_indices, double* vals);
void genTridiag(int *I, int *J, double *val, int N, int nz);
CSRMatrix generateSparseSPDMatrixCSR(int N);
CSRMatrix generateSparseIdentityMatrixCSR(int N);
void freeCSRMatrix(CSRMatrix &csrMtx);
double* csrToDense(const CSRMatrix &csrMtx);
void print_CSRMtx(const CSRMatrix &csrMtx);

//Construct precondtion matrix M with incomplete cholesky factorization such that M ~ L_hat * L_hat'
CSRMatrix constructPreConditionMatrixCSR(cusparseHandle_t cusparseHandler, CSRMatrix mtxA);

//Constructor like function
CSRMatrix constructCSRMatrix(int numOfRow, int numOfClm, int nnz, int* row_offsets, int* col_indices, double* vals){
    CSRMatrix csrMtx;
    csrMtx.numOfRows = numOfRow;
    csrMtx.numOfClms = numOfClm;
    csrMtx.numOfnz = nnz;

    csrMtx.row_offsets = (int*)malloc((numOfRow + 1) * sizeof(int));
    csrMtx.col_indices = (int*)malloc(nnz * sizeof(int));
    csrMtx.vals = (double*)malloc(nnz * sizeof(double));

    if(!csrMtx.row_offsets || !csrMtx.col_indices || !csrMtx.vals){
        fprintf(stderr, "\n\n!!!ERROR!!! Fail to allocate memory for CSR marix.\n");
        exit(EXIT_FAILURE);
    }

    memcpy(csrMtx.row_offsets, row_offsets, (numOfRow + 1) * sizeof(int));
    memcpy(csrMtx.col_indices, col_indices, nnz * sizeof(int));
    memcpy(csrMtx.vals, vals, nnz * sizeof(double));
    
    return csrMtx;
}



// Generate a random tridiagonal symmetric matrix
//It comes from CUDA CG sample code to generate sparse tridiagobal matrix
void genTridiag(int *I, int *J, double *val, int N, int nz) {
    I[0] = 0;
    J[0] = 0;
    J[1] = 1;
    val[0] = (double)rand() / RAND_MAX + 10.0f;
    val[1] = (double)rand() / RAND_MAX;
    int start;

    for (int i = 1; i < N; i++) {
        if (i > 1) {
            I[i] = I[i - 1] + 3;
        } else {
            I[1] = 2;
        }

        start = (i - 1) * 3 + 2;
        J[start] = i - 1;
        J[start + 1] = i;

        if (i < N - 1) {
            J[start + 2] = i + 1;
        }

        val[start] = val[start - 1];
        val[start + 1] = (double)rand() / RAND_MAX + 10.0f;

        if (i < N - 1) {
            val[start + 2] = (double)rand() / RAND_MAX;
        }
    }

    I[N] = nz;
}

// Generate a sparse SPD matrix in CSR format
CSRMatrix generateSparseSPDMatrixCSR(int N) {
    int nzMax = 3 * N - 2; // Maximum non-zero elements for a tridiagonal matrix
    int *row_offsets = (int*)malloc((N + 1) * sizeof(int));
    int *col_indices = (int*)malloc(nzMax * sizeof(int));
    double *vals = (double*)malloc(nzMax * sizeof(double));

    genTridiag(row_offsets, col_indices, vals, N, nzMax);

    // Create CSRMatrix object with the result
    CSRMatrix csrMtx;
    csrMtx.numOfRows = N;
    csrMtx.numOfClms = N;
    csrMtx.numOfnz = nzMax;
    csrMtx.row_offsets = row_offsets;
    csrMtx.col_indices = col_indices;
    csrMtx.vals = vals;

    return csrMtx;
}

// Generate a sparse SPD matrix in CSR format
CSRMatrix generateSparseIdentityMatrixCSR(int N) {
    int *row_offsets = (int*)malloc((N + 1) * sizeof(int));
    int *col_indices = (int*)malloc(N * sizeof(int));
    double *vals = (double*)malloc(N * sizeof(double));

    if (!row_offsets || !col_indices || !vals) {
        fprintf(stderr, "\n\nFailed to allocate memory for CSR matrix. \n\n");
        exit(EXIT_FAILURE);
    }


    // Fill row_offsets, col_indices, and vals
    for (int wkr = 0; wkr < N; ++wkr) {
        row_offsets[wkr] = wkr;
        col_indices[wkr] = wkr;
        vals[wkr] = 1.0f;
    }

    // Last element of row_offsets should be the number of non-zero elements
    row_offsets[N] = N; 


    // Create CSRMatrix object with the result
    CSRMatrix csrMtx;
    csrMtx.numOfRows = N;
    csrMtx.numOfClms = N;
    csrMtx.numOfnz = N;
    csrMtx.row_offsets = row_offsets;
    csrMtx.col_indices = col_indices;
    csrMtx.vals = vals;

    return csrMtx;
}


void freeCSRMatrix(CSRMatrix &csrMtx){
    free(csrMtx.row_offsets);
    free(csrMtx.col_indices);
    free(csrMtx.vals);
} // end of freeCSRMatrix


double* csrToDense(const CSRMatrix &csrMtx)
{
    double *dnsMtx = (double*)calloc(csrMtx.numOfRows * csrMtx.numOfClms, sizeof(double));
    for(int otWkr = 0; otWkr < csrMtx.numOfRows; otWkr++){
        for(int inWkr = csrMtx.row_offsets[otWkr]; inWkr < csrMtx.row_offsets[otWkr+1]; inWkr++){
            dnsMtx[otWkr * csrMtx.numOfClms + csrMtx.col_indices[inWkr]] = csrMtx.vals[inWkr];
        }// end of inner loop
    } // end of outer loop

    return dnsMtx;

} // end of csrToDense


//Print out CSRMatrix object
void print_CSRMtx(const CSRMatrix &csrMtx)
{
    printf("\n\nnumOfRows: %d, numOfClms: %d , number of non zero: %d", csrMtx.numOfRows, csrMtx.numOfClms, csrMtx.numOfnz);

    printf("\n\nrow_offsets: ");
    for(int wkr = 0; wkr <= csrMtx.numOfRows; wkr++){
        if(wkr == 0){
            printf("\n[ ");
        }
        printf("%d ", csrMtx.row_offsets[wkr]);
        if(wkr == csrMtx.numOfRows){
            printf("]\n");
        }
    }

    printf("\n\ncol_indices: ");
    for(int wkr = 0; wkr < csrMtx.numOfnz; wkr++){
        if(wkr == 0){
            printf("\n[ ");
        }
        printf("%d ", csrMtx.col_indices[wkr]);
        if(wkr == csrMtx.numOfnz - 1){
            printf("]\n");
        }
    }

    printf("\n\nnon zero values: ");
    for(int wkr = 0; wkr < csrMtx.numOfnz; wkr++){
        if(wkr == 0){
            printf("\n[ ");
        }
        printf("%f ", csrMtx.vals[wkr]);
        if(wkr == csrMtx.numOfnz - 1){
            printf("]\n");
        }
    }

    printf("\n");

} // end of print_CSRMtx

//Construct precondtion matrix M with incomplete cholesky factorization such that M ~ L_hat * L_hat'
CSRMatrix constructPreConditionMatrixCSR(cusparseHandle_t cusparseHandler, CSRMatrix mtxA){

    
    //(0) Set up variables
    cusparseMatDescr_t descr_M = 0;
    cusparseMatDescr_t descr_L = 0;

    // we need one info for csric02 and two info's for csrsv2
    csric02Info_t info_M =0;
    csrsv2Info_t info_L = 0;
    csrsv2Info_t info_Lt =0;

    int structural_zero = 0;
    int numerical_zero = 0;

    //(1) Allocate memory and copy mtxA values to GPU
    int N = mtxA.numOfRows;
    int numOfnz = mtxA.numOfnz;
    int *row_offsets_d = NULL;
    int *col_indices_d = NULL;
    double * vals_d = NULL;

    cusparseStatus_t status;
    

    CHECK(cudaMalloc((void**)&row_offsets_d, (N+1) * sizeof(int)));
    CHECK(cudaMalloc((void**)&col_indices_d, numOfnz * sizeof(int)));
    CHECK(cudaMalloc((void**)&vals_d, numOfnz * sizeof(double)));

    CHECK(cudaMemcpy(row_offsets_d, mtxA.row_offsets, (N+1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(col_indices_d, mtxA.col_indices, (numOfnz) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(vals_d, mtxA.vals, (numOfnz) * sizeof(double), cudaMemcpyHostToDevice));

    //(2) Create descriptors
    // - matrix M is base-1
    // - matrix L is base-1
    // - matrix L is lower triangular
    // - matrix L has non-unit diagonal
    CHECK_CUSPARSE(cusparseCreateMatDescr(&descr_M));
    CHECK_CUSPARSE(cusparseSetMatIndexBase(descr_M,CUSPARSE_INDEX_BASE_ONE));
    CHECK_CUSPARSE(cusparseSetMatType(descr_M, CUSPARSE_MATRIX_TYPE_GENERAL));
    CHECK_CUSPARSE(cusparseCreateMatDescr(&descr_L));
    CHECK_CUSPARSE(cusparseSetMatIndexBase(descr_L, CUSPARSE_INDEX_BASE_ONE));
    CHECK_CUSPARSE(cusparseSetMatType(descr_L, CUSPARSE_MATRIX_TYPE_GENERAL));
    CHECK_CUSPARSE(cusparseSetMatFillMode(descr_L, CUSPARSE_FILL_MODE_LOWER));
    CHECK_CUSPARSE(cusparseSetMatDiagType(descr_L, CUSPARSE_DIAG_TYPE_NON_UNIT));
    CHECK_CUSPARSE(cusparseCreateCsric02Info(&info_M));
    CHECK_CUSPARSE(cusparseCreateCsrsv2Info(&info_L));
    CHECK_CUSPARSE(cusparseCreateCsrsv2Info(&info_Lt));

    //(2) Calculate buffer size and allocate space
    int pBufferSize_M = 0;
    int pBufferSize_L = 0;
    int pBufferSize_Lt = 0;
    int pBufferSize = 0;
    void *pBuffer = 0;
    CHECK_CUSPARSE(cusparseDcsric02_bufferSize(cusparseHandler, N, numOfnz, descr_M, vals_d, row_offsets_d, col_indices_d, info_M, &pBufferSize_M));
    CHECK_CUSPARSE(cusparseDcsrsv2_bufferSize(cusparseHandler, CUSPARSE_OPERATION_NON_TRANSPOSE, N, numOfnz, descr_L, vals_d, row_offsets_d, col_indices_d, info_L, &pBufferSize_L));
    CHECK_CUSPARSE(cusparseDcsrsv2_bufferSize(cusparseHandler, CUSPARSE_OPERATION_NON_TRANSPOSE, N, numOfnz, descr_L, vals_d, row_offsets_d, col_indices_d, info_Lt, &pBufferSize_Lt));
    
    // pBuffer returned by cudaMalloc is automatically aligned to 128 bytes.
    pBufferSize = max(pBufferSize_M, max(pBufferSize_L, pBufferSize_Lt));
    CHECK(cudaMalloc(&pBuffer, pBufferSize));

    
    // (4): perform analysis of incomplete Cholesky on M
    //      perform analysis of triangular solve on L
    //      perform analysis of triangular solve on L'
    // The lower triangular part of M has the same sparsity pattern as L, so
    // we can do analysis of csric02 and csrsv2 simultaneously.
    CHECK_CUSPARSE(cusparseDcsric02_analysis(cusparseHandler, N, numOfnz, descr_M, vals_d, row_offsets_d, col_indices_d, info_M, CUSPARSE_SOLVE_POLICY_NO_LEVEL, pBuffer));
    status = cusparseXcsric02_zeroPivot(cusparseHandler, info_M, &structural_zero);
    if(CUSPARSE_STATUS_ZERO_PIVOT == status){
        printf("A(%d,%d) is missing\n", structural_zero, structural_zero);
    }

    CHECK_CUSPARSE(cusparseDcsrsv2_analysis(cusparseHandler, CUSPARSE_OPERATION_NON_TRANSPOSE, N, numOfnz, descr_L, vals_d, row_offsets_d, col_indices_d, info_L, CUSPARSE_SOLVE_POLICY_NO_LEVEL, pBuffer));
    CHECK_CUSPARSE(cusparseDcsrsv2_analysis(cusparseHandler, CUSPARSE_OPERATION_TRANSPOSE, N, numOfnz, descr_L, vals_d, row_offsets_d, col_indices_d, info_Lt, CUSPARSE_SOLVE_POLICY_USE_LEVEL, pBuffer));

    //(5) Perform Incomplete Cholesky Factorization
    CHECK_CUSPARSE(cusparseDcsric02(cusparseHandler, N, numOfnz, descr_M, vals_d, row_offsets_d, col_indices_d, info_M, CUSPARSE_SOLVE_POLICY_NO_LEVEL, pBuffer));
    status = cusparseXcsric02_zeroPivot(cusparseHandler, info_M, &numerical_zero);
    if(CUSPARSE_STATUS_ZERO_PIVOT == status){
        printf("L(%d, %d) is zero\n", numerical_zero, numerical_zero);
    }

    //(6) Extract the lower triangular matrix L_hat from GPU to CPU
    int *row_offsets_h = (int*)malloc((N+1) * sizeof(int));
    int *col_indices_h = (int*)malloc((numOfnz) * sizeof(int));
    double *vals_h = (double*)malloc(numOfnz * sizeof(double));
    CHECK(cudaMemcpy(row_offsets_h, row_offsets_d, (N + 1) * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(col_indices_h, col_indices_d, numOfnz * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(vals_h, vals_d, numOfnz * sizeof(double), cudaMemcpyDeviceToHost));

    //(7) Create precondition CSRMatrix object
    CSRMatrix csrMtxM;
    csrMtxM.numOfRows = N;
    csrMtxM.numOfClms = N;
    csrMtxM.numOfnz = numOfnz;
    csrMtxM.row_offsets = row_offsets_h;
    csrMtxM.col_indices = col_indices_h;
    csrMtxM.vals = vals_h;

    //(8) Free memeory
    CHECK(cudaFree(pBuffer));
    CHECK_CUSPARSE(cusparseDestroyMatDescr(descr_M));
    CHECK_CUSPARSE(cusparseDestroyMatDescr(descr_L));
    CHECK_CUSPARSE(cusparseDestroyCsric02Info(info_M));
    CHECK_CUSPARSE(cusparseDestroyCsrsv2Info(info_L));
    CHECK_CUSPARSE(cusparseDestroyCsrsv2Info(info_Lt));

    return csrMtxM;
}


#endif // CSRMatix_h