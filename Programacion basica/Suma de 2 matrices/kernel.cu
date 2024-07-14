#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

// Tamaño de la matriz = N * N;
#define N 8192

#define ERROR_CHECK { cudaError_t err; \
  if ((err = cudaGetLastError()) != cudaSuccess) { \
    printf("CUDA error: %s, line %d\n", cudaGetErrorString(err), __LINE__);}}


// Función secuencial que suma los elementos de dos matrices posición a posición
void addSec(int* a, int* b, int* c)
{
	unsigned iPos;

	// Para cada fila de la matriz
	for (int iFila = 0; iFila < N; iFila++)
		// Para cada columna de la matriz
		for (int iCol = 0; iCol < N; iCol++) {
			iPos = iFila * N + iCol;
			c[iPos] = a[iPos] + b[iPos];
		}
}

__global__ void addPar(int* a, int* b, int* c)
{
	unsigned int iCol = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int iFila = blockDim.y * blockIdx.y + threadIdx.y;

	unsigned int iPos = iFila * N + iCol;
	c[iPos] = a[iPos] + b[iPos];
}


int main(void)
{
	int* host_a, * host_b, * host_cSec, * host_cPar;
	int* dev_a, * dev_b, * dev_c;

	// Se pide memoria para las variables del Host
	printf("\nPidiendo memoria en el Host");
	//////////////////////////////////////////////////////////////////////////////////////////////
	//										PASO 1												//
	//									  COMPLETAR												//
	// Pedir memoria para todas las variables en el Host (host_a, host_b, host_cSec, host_cPar) // 
	//////////////////////////////////////////////////////////////////////////////////////////////

	host_a = (int*)malloc(N * N * sizeof(int));
	host_b = (int*)malloc(N * N * sizeof(int));
	host_cSec = (int*)malloc(N * N * sizeof(int));
	host_cPar = (int*)malloc(N * N * sizeof(int));

	// Se rellenan las matrices con valores aleatorios
	printf("\nRellenando matrices");
	srand((unsigned)time(NULL));
	for (unsigned int i = 0; i < N; i++) {
		for (unsigned int j = 0; j < N; j++) {
			host_a[i * N + j] = floorf(100 * (rand() / (float)RAND_MAX));
			host_b[i * N + j] = floorf(100 * (rand() / (float)RAND_MAX));
		}
	}

	// Se reserva memoria en el Device
	printf("\nReservando memoria en el Device");
	//////////////////////////////////////////////////////////////////////////////////////////////
	//										PASO 2												//
	//									  COMPLETAR												//
	// Pedir memoria para todas las variables en el Device (dev_a, dev_b, dev_c)				// 
	//////////////////////////////////////////////////////////////////////////////////////////////

	cudaMalloc((void**)&dev_a, N * N * sizeof(int));
	cudaMalloc((void**)&dev_b, N * N * sizeof(int));
	cudaMalloc((void**)&dev_c, N * N * sizeof(int));

	ERROR_CHECK;

	// Se copian las matrices del Host al Device
	printf("\nCopiando matrices al Device");
	//////////////////////////////////////////////////////////////////////////////////////////////
	//										PASO 3												//
	//									  COMPLETAR												//
	// Copiar las matrices para hacer la suma del Host al Device    							// 
	//////////////////////////////////////////////////////////////////////////////////////////////

	cudaMemcpy(dev_a, host_a, N * N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, host_b, N * N * sizeof(int), cudaMemcpyHostToDevice);
	ERROR_CHECK;

	//////////////////////////////////////////////////////////////////////////////////////////////
	//										PASO 4												//
	//									  COMPLETAR												//
	// Preparar la llamada al kernel y llamarlo con los parámetros adecuados    				// 
	//////////////////////////////////////////////////////////////////////////////////////////////
	dim3 gridOfBlocks(256,256);
	dim3 blockOfThreads(32,32);

	printf("\nSuma paralela");
	addPar <<< gridOfBlocks, blockOfThreads>>> (dev_a,dev_b,dev_c);
	ERROR_CHECK;

	// Se copia la matriz resultado del Device al Host
	printf("\nCopiando matriz resultado al Host");
	//////////////////////////////////////////////////////////////////////////////////////////////
	//										PASO 5												//
	//									  COMPLETAR												//
	// Copiar la matriz resultado del Device al Host			    							// 
	//////////////////////////////////////////////////////////////////////////////////////////////

	cudaMemcpy(host_a, dev_a, N * N * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(host_b, dev_b, N * N * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(host_cPar, dev_c, N * N * sizeof(int), cudaMemcpyDeviceToHost);

	// Se llama a la suma de matrices secuencial
	printf("\nSuma secuencial");
	addSec(host_a, host_b, host_cSec);

	// Se compara si las matrices resultado paralela y secuencial son iguales
	printf("\nComprobando resultados");
	bool bError = false;
	unsigned int iPos;
	for (int iFila = 0; iFila < N; iFila++)
	{
		for (int iCol = 0; iCol < N; iCol++) {
			iPos = iFila * N + iCol;

			if (host_cSec[iPos] != host_cPar[iPos])
			{
				printf("\nValores diferentes en [%d][%d] => (%d,%d) => %d %d", iCol, iFila, host_a[iPos],
					host_b[iPos], host_cSec[iPos],
					host_cPar[iPos]);
				bError = true;
				break;
			}
		}
		if (bError)
			break;
	}
	if (!bError)
		printf("\nCORRECTO!\n");

	// Se libera la memoria del Host
	//////////////////////////////////////////////////////////////////////////////////////////////
	//										PASO 6												//
	//									  COMPLETAR												//
	// Liberar la memoria pedida en el Host			    										// 
	//////////////////////////////////////////////////////////////////////////////////////////////

	free(host_a);
	free(host_b);
	free(host_cPar);
	free(host_cSec);

	// Se libera la memoria del Device
	//////////////////////////////////////////////////////////////////////////////////////////////
	//										PASO 7												//
	//									  COMPLETAR												//
	// Liberar la memoria pedida en el Device		    										// 
	//////////////////////////////////////////////////////////////////////////////////////////////

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);


	return 0;
}