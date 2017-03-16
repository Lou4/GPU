#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/timeb.h>
#include <sys/time.h>

#define N 1000000000


__global__ void hello(int *a, int *b, int *res, int const BLOCKS){
	//printf("threadIdx.x: %d blockIdx.x: %d, blockDim.x: %d, blockDim.y: %d, blockDim.z: %d\n", threadIdx.x, blockIdx.x, blockDim.x, blockDim.y, blockDim.z);
	for(int i = BLOCKS; i < BLOCKS + blockDim.x; i++){
		res[i] = a[i] + b[i];
	}

}

void sequential_product(int *a, int *b, int *res){
	for(int i = 0; i < N; i++){
		res[i] = a[i] + b[i];
	}
}

int main(void){
	int BLOCKS = 10;
	int THREAD_PER_BLOCK = 100;
	int *a, *b, *res, *res_seq;
	int *dev_a, *dev_b, *dev_res;

	a = (int*)malloc(N * sizeof(int));
	b = (int*)malloc(N * sizeof(int));
	res = (int*)malloc(N * sizeof(int));
	res_seq = (int*)malloc(N * sizeof(int));

	printf("Start sequential\n");
	sequential_product(a, b, res);
	printf("Stop sequential\n");

	cudaMalloc((void**)&dev_a, N*sizeof(int));
	cudaMalloc((void**)&dev_b, N*sizeof(int));
	cudaMalloc((void**)&dev_res, N*sizeof(int));

	cudaMemcpy(dev_a, a, N*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, N*sizeof(int), cudaMemcpyHostToDevice);

	printf("Start parallel\n");
	hello<<<BLOCKS,THREAD_PER_BLOCK>>>(dev_a, dev_b, dev_res, BLOCKS);
	printf("Stop parallel\n");

	cudaMemcpy(res_seq, dev_res, N*sizeof(int), cudaMemcpyDeviceToHost);

	int isEqual = 1;
	for(int i = 0; i<N; i++){
		if(res[i] != res_seq[i]){
			isEqual = 0;
		}
	}

	if(isEqual) printf("The two sum is equals\n"); else printf("Error in sum\n");
	cudaDeviceReset();
	return 0;
}
