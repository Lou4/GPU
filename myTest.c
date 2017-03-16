#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "matrix.h"

#define SEED 5
#define BLK_NUM 3
#define MTX_SIZE 20


void matrix_product(mqdb *A, mqdb *B, mqdb *RES){
	int lenA = 0;
	int lenB = 0;
	int index;
	int val;

	//So che le matrici sono sicuramente quadrate, scopriamo se sono moltiplicabili
	for(int i = 0; i<A->num_blks; i++){
		lenA += A->blk_dims[i];
	} 
	printf("A is a %dx%d matrix\n", lenA, lenA);
	
	for(int i = 0; i<B->num_blks; i++){
		lenB += B->blk_dims[i];
	}
	printf("B is a %dx%d matrix\n", lenB, lenB);
	
	if(lenA != lenB){
		printf("Error, can't multiply a %dx%d matrix with a %dx%d matrix\n", lenA, lenA, lenB, lenB);
		exit(EXIT_FAILURE);
	}

	//Moltiplichiamo
	RES->elems = calloc(lenA * lenA, sizeof(int));
	RES->num_blks = A->num_blks;
	RES->blk_dims = calloc(A->num_blks, sizeof(int));
	memcpy(RES->blk_dims, A->blk_dims, A->num_blks*sizeof(int));	

	//In ogni punto in cui ho usato lenA, potevo anche usare lenB tanto le due matrici hanno la stessa dimenzione N x N
	for(int c = 0; c < lenA; c++){
		for(int r = 0; r < lenA; r++){
			val = 0;
			//Calcoliamo il valore che andrà messo nella r-esima riga c-esima colonna della matrice risultato
			for(int i = 0, indexA = c * lenA, indexB = r; i < lenA; i++, indexA++, indexB+=lenA){
				//printf("[%d] indexA: %d, indexB: %d\n",i , indexA, indexB);
				val += A->elems[indexA] * B->elems[indexB];
			}			
			index = r + (c*lenA);
			//printf("index: %d\n", index);
			RES->elems[index] = val;	
		}
	}

	printf("Multiply is over...\n");				
}

int main(void){
	mqdb A, B, C, res;
	
	A = rand_gen_mqdb(MTX_SIZE, BLK_NUM, SEED);
	B = rand_gen_mqdb(MTX_SIZE, BLK_NUM, SEED);
	
	print_mqdb(A);
	printf("\n");
	print_mqdb(B);
	printf("\n");

	matrix_product(&A, &B, &res);
	print_mqdb(res);
}
