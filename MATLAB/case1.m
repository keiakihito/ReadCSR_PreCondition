% Create the same sparse matrix A used in your CUDA function
A = sparse([4 -1 0 -1 0 0;
                     -1 4 -1 0 -1 0;
                     0 -1 4 0 0 -1;
                     -1 0 0 4 -1 0;
                     0 -1 0 -1 4 -1;
                     0 0 -1 0 -1 4]);

% Compute the incomplete Cholesky factorization
L = ichol(A);
mtxM = L * L';
disp(mtxM);

