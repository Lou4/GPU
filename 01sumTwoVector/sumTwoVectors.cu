#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/timeb.h>
#include <sys/time.h>

#define N 300000000
#define LIMIT 4000
__global__ void hello(int *a, int *b, int *res, int const BLOCKS){
	printf("[%d-%d] start \n", blockIdx.x, threadIdx.x);
	int numEl = (N / BLOCKS / blockDim.x);
	int start = (blockDim.x * blockIdx.x * numEl) + (threadIdx.x * numEl);
/*
#ifdef DEBUG
	printf("threadIdx.x: %d blockIdx.x: %d, blockDim.x: %d, blockDim.y: %d, blockDim.z: %d\n", threadIdx.x, blockIdx.x, blockDim.x, blockDim.y, blockDim.z);
	printf("start: %d, end: %d\n", start, start + numEl);
	printf("N: %d, BLOCKS: %d, blockDim.x: %d, numEl: %d\n", N, BLOCKS, blockDim.x, numEl);
#endif
*/
	for(int i = start; i < start + numEl; i++){
		res[i] = (a[i] + b[i]) * 2;
	}

	printf("[%d-%d] end \n", blockIdx.x, threadIdx.x);
}

void array_init(int *a, int *b){
	printf("init array . . .\n");
	for(int i = 0; i<N; i++){
		a[i] = i % LIMIT;
		b[i] = (i * i) % LIMIT;
	}
	printf("end init array");
}

void sequential_product(int *a, int *b, int *res){
	printf("Start sequential\n");
	for(int i = 0; i < N; i++){
		res[i] = (a[i] + b[i]) * 2;
	}
	printf("End sequential\n");
}


int main(void){
	int BLOCKS = 5;
	int THREAD_PER_BLOCK = 10;
	int *a, *b, *res, *res_seq;  	//Host var
	int *dev_a, *dev_b, *dev_res;	//Device var
	struct timeb init, fin;
	int diff;

	float millisec = 0;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//Init array
	a = (int*)malloc(N * sizeof(int));
	b = (int*)malloc(N * sizeof(int));
	res = (int*)calloc(N, sizeof(int));
	res_seq = (int*)calloc(N, sizeof(int));

	ftime(&init);
	array_init(a, b);
	ftime(&fin);
	diff = (int) (1000.0 * (fin.time - init.time) + (fin.millitm - init.millitm));
	printf("Stop sequential, time: %d (millisec)\n\n\n\n", diff);

	cudaMalloc((void**)&dev_a, N*sizeof(int));
	cudaMalloc((void**)&dev_b, N*sizeof(int));
	cudaMalloc((void**)&dev_res, N*sizeof(int));

	cudaMemcpy(dev_a, a, N*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, N*sizeof(int), cudaMemcpyHostToDevice);

	//Sequential
	ftime(&init);
	sequential_product(a, b, res);
	ftime(&fin);
	diff = (int) (1000.0 * (fin.time - init.time) + (fin.millitm - init.millitm));
	printf("Stop sequential, time: %d (millisec)\n\n\n\n", diff);

	//Parallel
	printf("Start parallel\n");
	cudaEventRecord(start);
	hello<<<BLOCKS,THREAD_PER_BLOCK>>>(dev_a, dev_b, dev_res, BLOCKS);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&millisec, start, stop);
	printf("Stop parallel, time: %.5f (millisec)\n\n\n\n", millisec);


	cudaMemcpy(res_seq, dev_res, N*sizeof(int), cudaMemcpyDeviceToHost);


#ifdef DEBUG
	for(int i = 0; i<100; i++){
		printf("res[%d] = %d, res_seq[%d] = %d\n", i, res[i], i, res_seq[i]);
	}
#endif

	//Corectness checking . . .
	int isEqual = 1;
	for(int i = 0; i<N; i++){
		if(res[i] != res_seq[i]){
			printf("res[%d] = %d, res_seq[%d] = %d\n", i, res[i], i, res_seq[i]);
			isEqual = 0;
			break;
		}
	}


	if(isEqual)
		printf("The two sum is equals\n");
	else
		printf("Error in sum\n");

	cudaDeviceReset();
	return 0;
}
