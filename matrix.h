/*
 * matrix.h
 *
 * Matrices are stored in row-major order:
 * M(i,j) corresponds to *(M.elem + i * M.cols + j)
 */

typedef struct {
	char desc[100];  // description
	int num_blks;    // num. of blocks
	int *blk_dims;   // block dimensions
	float* elems;    // elements in row-major order
} mqdb;

mqdb rand_gen_mqdb(int, int, int);
void matrix_prod_H(mqdb, mqdb, mqdb);
void checkResult(mqdb, mqdb);
void print_mqdb(mqdb);
