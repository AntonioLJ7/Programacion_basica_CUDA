
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define N       60000
#define THREADS  1024

// Realiza la convolución secuencial de los valores de los vectores
float convolucionSecuencial(float* vectorA, float* vectorB)
{
	int iPos;
	float fResultado = 0.0;

	// Se multiplican los dos vectores posición a posición
	for (iPos = 0; iPos < N; iPos++)
		vectorA[iPos] *= vectorB[iPos];

	// Se realiza la convolución
	for (iPos = 0; iPos < N; iPos++)
		fResultado += vectorA[iPos];

	return fResultado;
}

__global__ void multParalelaElementoAElemento(float* vectorA, float* vectorB)
{
	/* COMPLETAR */;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;	//Para saber en que bloque me encuentro

	if (i < N)
		vectorA[i] *= vectorB[i];
}


// Kernell CUDA para la suma de los valores del vector
__global__ void sumaParalela(float* vector, int nDatos)
{
	__shared__ float vectorCompartido[THREADS];

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	// Si el dato está fuera del vector
	// o si la hebra no tiene que procesar ningún dato
	if (i >= N || tid >= nDatos)
		vectorCompartido[tid] = 0.0;	// Se rellena con ceros
	else
		vectorCompartido[tid] = vector[i];	// Se copia el dato a la memoria compartida

	__syncthreads();

	for (unsigned int iPos = (blockDim.x >> 1); iPos >= 1; iPos = iPos >> 1)
	{
		if (tid < iPos)
			vectorCompartido[tid] += vectorCompartido[tid + iPos];

		__syncthreads();
	}

	if (tid == 0)
		vector[blockIdx.x] = vectorCompartido[0];
}



int main(void)
{
	float host_vA[N], host_vB[N];
	float fResultadoParalelo, fResultadoSecuencial;
	float* dev_vA, * dev_vB;
	unsigned int blocks;
	unsigned int nDatos;

	// Se llena de forma aleatoria el vector sobre el que se realiza la suma
	srand((unsigned)time(NULL));
	for (int i = 0; i < N; i++)
	{
		host_vA[i] = floorf(10 * (rand() / (float)RAND_MAX));
		host_vB[i] = floorf(10 * (rand() / (float)RAND_MAX));
	}

	// Pedir memoria en el Device para los vectores a sumar (dev_vA y dev_vB)
	/* COMPLETAR */

	cudaMalloc((void**)&dev_vA, N * sizeof(float));
	cudaMalloc((void**)&dev_vB, N * sizeof(float));

	// Transferir los vectores del Host al Device
	/* COMPLETAR */

	cudaMemcpy(dev_vA, host_vA, N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_vB, host_vB, N * sizeof(float), cudaMemcpyHostToDevice);


	blocks = ceil((float)N / (float)THREADS);

	// Llamada al kernel para hacer la multiplicación elemento a elemento
	/* COMPLETAR */
	multParalelaElementoAElemento <<<blocks, THREADS >>> (dev_vA, dev_vB);
	blocks = N;

	// Llamar al kernell CUDA
	do
	{
		// Se calcula el número de datos que se procesarán por cada bloque
		if (blocks >= THREADS)
			nDatos = THREADS;
		else
			nDatos = blocks % THREADS;

		// Se calcula el número de bloques necesarios para el número de hebras
		blocks = ceil((float)blocks / (float)THREADS);

		// Llamar al kernel para hacer la resucción
		/* COMPLETAR */
		sumaParalela <<< blocks, THREADS >>> (dev_vA, nDatos);
	} while (blocks > 1);

	// Copiar el resultado de la operación del Device al Host
	/* COMPLETAR */

	cudaMemcpy(&fResultadoParalelo, dev_vA, sizeof(float), cudaMemcpyDeviceToHost);

	// Se comprueba que el resultado es correcto y se muestra un mensaje
	fResultadoSecuencial = convolucionSecuencial(host_vA, host_vB);
	if (fResultadoParalelo == fResultadoSecuencial)
		printf("Operacion correcta\nDevice = %f\nHost   = %f\n", fResultadoParalelo, fResultadoSecuencial);
	else
		printf("Operacion INCORRECTA\nDevice = %f\nHost   = %f\n", fResultadoParalelo, fResultadoSecuencial);

	// Librerar la memoria solicitada en el Device
	/* COMPLETAR */

	cudaFree(dev_vA);
	cudaFree(dev_vB);

	return 0;
}