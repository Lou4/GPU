/*
 * rand_gen_mqdb.cu
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "matrix.h"

/*
 * rand_gen_mqdb: return a random generated mqdb
 */
mqdb rand_gen_mqdb(int n, int k, int seed) {

	mqdb M;
	srand(seed);

	if (n < 0 || k > n) {
		printf("error: n must be positive and greater than k!\n");
		exit(-1);
	}

	// random generation of block sizes
	M.num_blks = k;
	M.blk_dims = (int *) malloc(k * sizeof(int));
	int sum = 0;
	int lim = n - (k - 1);
	for (int i = 0; i < k - 1; i++) {
		M.blk_dims[i] = rand() % lim/2 + 1;
		sum += M.blk_dims[i];
		lim = n - sum - (k-i-2);
	}
	M.blk_dims[k - 1] = n - sum;

	// random permute blk dims
	for (int i = k - 1; i >= 0; --i) {
		int j = rand() % (i + 1);
		int temp = M.blk_dims[i];
		M.blk_dims[i] = M.blk_dims[j];
		M.blk_dims[j] = temp;
	}
	
	//print blk_dims
   for(int i = 0; i < k; i++){
     printf("blk %d size: %d\n", i, M.blk_dims[i]);
   }

	// random fill mat entries
	M.elems = (float *) calloc(n * n, sizeof(float));
	printf("Elems: %d\n", n*n);
	int offset = 0;
	for (int i = 0; i < k; i++) {
		for (int j = 0; j < M.blk_dims[i]; j++) {
			for (int k = 0; k < M.blk_dims[i]; k++) {
				M.elems[offset * n + j * n + k + offset] = 1;
				//(float) rand() / (float) RAND_MAX;
			}
		}
		offset += M.blk_dims[i];
	}

	// set description
	sprintf(M.desc, "Random mqdb:  mat. size = %d, num. blocks = %d\n", n, k);

	return M;
}

/*
 * print_mqdb: print matrix of type mqdb
 */
void print_mqdb(mqdb A) {
	int n = 0;
	for (int j = 0; j < A.num_blks; j++)
		n += A.blk_dims[j];
	printf("%s", A.desc);
	for (int j = 0; j < n * n; j++) {
		if (A.elems[j] == 0)
			printf("------");
		else
			printf("%5.2f ", A.elems[j]);
		if ((j + 1) % n == 0)
			printf("\n");
	}
	printf("\n");
}
