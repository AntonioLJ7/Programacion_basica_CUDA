#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define ERROR_CHECK { cudaError_t err; \
  if ((err = cudaGetLastError()) != cudaSuccess) { \
    printf("CUDA error: %s, line %d\n", cudaGetErrorString(err), __LINE__);}}


#define N 512

__constant__ float coefs[N];


// Realiza la convolución secuencial de los valores de los vectores
void FIRSecuencial(float* data, float* output, float* coef, int iPosInput)
{
	int iPosCoef;
	float fResultado;

	fResultado = 0.0;
	for (iPosCoef = 0; iPosCoef < N; iPosCoef++)
	{
		if (iPosInput + iPosCoef - N + 1 >= 0)
			fResultado += data[iPosInput + iPosCoef - N + 1] * coef[iPosCoef];
	}
	output[iPosInput] = fResultado;
}


// Kernell CUDA para la convolución de dos vectores
__global__ void FIRParalelo(float* data, float* output, int iPosInput)
{
	//////////////////////////////////////////////////////////////////////////////////////////////////////////
	//											Paso 1														//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////
	/*              COMPLETAR                     */
	__shared__ float vectorCompartido[N];
	//////////////////////////////////////////////////////////////////////////////////////////////////////////
	//											Paso 2														//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////

	// Primera y segunda posición de memoria dentro de los datos (data) de los que se encarga la hebra
	int iPosData1 = iPosInput - N + threadIdx.x + 1;/*              COMPLETAR                     */
	int iPosData2 = iPosData1 + (N / 2);	/*              COMPLETAR                     */

	//////////////////////////////////////////////////////////////////////////////////////////////////////////
	//											Paso 3														//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////

	// Primera y segunda posición de memoria dentro del vector compartido y del vector de coeficientes (coef)
	int iPosCompartidoYCoef1 = threadIdx.x;/*              COMPLETAR                     */
	int iPosCompartidoYCoef2 = threadIdx.x + (N / 2);/*         COMPLETAR                     */

	//////////////////////////////////////////////////////////////////////////////////////////////////////////
	//											Paso 4														//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////

	// Se multiplican los vectores posición a posición
	// Cada hebra se encarga de calcular dos valores y almacenarlos en memoria compartida
	// Hay que comprobar que las posiciones existen dentro de los límites de los vectores a multiplicar
	// Cálculo del valor de la primera posición coefs * data
	/*              COMPLETAR                     */

	if (iPosData1 >= 0)
	{
		vectorCompartido[iPosCompartidoYCoef1] = coefs[iPosCompartidoYCoef1] * data[iPosData1];
	}
	else
		vectorCompartido[iPosCompartidoYCoef1] = 0;


	// Cálculo del valor de la segunda posición coefs * data
	/*              COMPLETAR                     */

	if (iPosData2 >= 0)
	{
		vectorCompartido[iPosCompartidoYCoef2] = coefs[iPosCompartidoYCoef2] * data[iPosData2];
	}
	else
		vectorCompartido[iPosCompartidoYCoef2] = 0;

	//////////////////////////////////////////////////////////////////////////////////////////////////////////
	//											Paso 5														//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////
	/*              COMPLETAR                     */
	__syncthreads();
	//////////////////////////////////////////////////////////////////////////////////////////////////////////
	//											Paso 6														//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////

	// Se realiza la suma por reducción de los valores del vector en memoria compartida
	/*              COMPLETAR                     */
	for (unsigned int iPos = N >> 1; iPos >= 1; iPos = iPos >> 1)
	{
		if (threadIdx.x < iPos)
		{
			vectorCompartido[threadIdx.x]+= vectorCompartido[threadIdx.x + iPos];
		}
		__syncthreads();
	}
	//////////////////////////////////////////////////////////////////////////////////////////////////////////
	//											Paso 7														//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////

	// La primera hebra almacena el resultado total en el vector de salida
	/*              COMPLETAR                     */
	if (threadIdx.x == 0)
		output[iPosInput] = vectorCompartido[0];
}

int main(int argc, char** argv)
{
	int iPos, iDataCount;
	char buffer[32];
	short bufferData;
	float host_coef[N], * host_input, * host_output, * sec_output;
	float* dev_input, * dev_output;

	if (argc != 5)
	{
		printf("USO:\nFIR <ficheroCoeficientes> <ficheroDatos> <ficheroSalidaParalelo> <ficheroSalidaSecuencial>\n");
		return -1;
	}

	//////////////////////////////////////////////////////////////////////////////////////////////////////////
	//											Paso 1														//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////

	// Se abre el fichero de coeficientes
	FILE* fCoef = fopen(argv[1], "rt");
	if (fCoef == NULL)
	{
		printf("Error abriendo el fichero de coeficientes\n");
		return -1;
	}

	// Se cargan los coeficientes en memoria del Host
	for (iPos = 0; iPos < N; iPos++)
	{
		fgets(buffer, 32, fCoef);
		host_coef[iPos] = (float)atof(buffer);
		fgets(buffer, 32, fCoef);	// Truco para que funcione bien con el formato que tiene el fichero
	}
	fclose(fCoef);

	//////////////////////////////////////////////////////////////////////////////////////////////////////////
	//											Paso 2														//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////

	// Se abre el fichero de datos
	FILE* fData = fopen(argv[2], "rb"); 
	if (fData == NULL)
	{
		printf("Error abriendo el fichero de datos\n");
		return -1;
	}

	// Se averigua el tamaño en bytes del fichero de datos
	fseek(fData, 0, SEEK_END);
	iDataCount = ftell(fData);
	fseek(fData, 0, SEEK_SET);

	// Como los datos son de 2 bytes, se calcula el número de datos (muestras) del fichero
	iDataCount /= 2;

	//////////////////////////////////////////////////////////////////////////////////////////////////////////
	//											Paso 3														//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////

	// Se pide memoria para las variables en el host	
	/*              COMPLETAR                     */

	host_input = (float*)malloc(iDataCount * sizeof(float));
	host_output = (float*)malloc(iDataCount * sizeof(float));
	sec_output = (float*)malloc(iDataCount * sizeof(float));

	//////////////////////////////////////////////////////////////////////////////////////////////////////////
	//											Paso 4														//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////

	// Se cargan los datos en memoria del Host
	for (iPos = 0; iPos < iDataCount; iPos++)
	{
		fread((void*)&bufferData, sizeof(short), 1, fData);
		host_input[iPos] = (float)bufferData;
	}
	fclose(fData);

	//////////////////////////////////////////////////////////////////////////////////////////////////////////
	//											Paso 5														//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////

	// Pedir memoria en el Device
	/*              COMPLETAR                     */

	cudaMalloc((void**)&dev_input, iDataCount * sizeof(float));
	cudaMalloc((void**)&dev_output, iDataCount * sizeof(float));
	ERROR_CHECK;

	//////////////////////////////////////////////////////////////////////////////////////////////////////////
	//											Paso 6														//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////

	// Transferir los coeficientes del Host al Device
	/*              COMPLETAR                     */

	cudaMemcpyToSymbol(coefs, host_coef, N * sizeof(float));

	// Transferir los datos del Host al Device
	/*              COMPLETAR                     */

	cudaMemcpy(dev_input, host_input, iDataCount * sizeof(float), cudaMemcpyHostToDevice);
	ERROR_CHECK;

	//////////////////////////////////////////////////////////////////////////////////////////////////////////
	//											Paso 7														//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////

	int threads = (N / 2) + N % 2;

	// Llamar al kernell CUDA para cada elemento de los datos
	printf("\nLlamada al FIR paralelo");
	for (iPos = 0; iPos < iDataCount; iPos++)
		/*              COMPLETAR                     */
	FIRParalelo << <2, threads >> > (dev_input, dev_output, iPos);
	printf("\nFin FIR paralelo\n");

	//////////////////////////////////////////////////////////////////////////////////////////////////////////
	//											Paso 8														//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////

	// Copiar el resultado de la operación del Device al Host
	/*              COMPLETAR                     */
	cudaMemcpy(host_output, dev_output, iDataCount * sizeof(float), cudaMemcpyDeviceToHost);
	ERROR_CHECK;

	//////////////////////////////////////////////////////////////////////////////////////////////////////////
	//											Paso 9														//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////

	// Llamada al filtro secuencial
	printf("\nLlamada al FIR secuencial\n");
	for (iPos = 0; iPos < iDataCount; iPos++)
		FIRSecuencial(host_input, sec_output, host_coef, iPos);
	printf("Fin FIR secuencial\n");

	//////////////////////////////////////////////////////////////////////////////////////////////////////////
	//											Paso 10														//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////

	// Se abre el fichero de salida paralela
	FILE* fDataOut = fopen(argv[3], "wb");
	if (fDataOut == NULL)
	{
		printf("Error abriendo el fichero de salida\n");
		return -1;
	}

	// Se escriben los datos del filtrado paralelo
	short data;
	for (iPos = 0; iPos < iDataCount; iPos++)
	{
		data = (short)floor(host_output[iPos]);
		fwrite((void*)&data, sizeof(short), 1, fDataOut);
	}
	fclose(fDataOut);

	//////////////////////////////////////////////////////////////////////////////////////////////////////////
	//											Paso 11 													//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////

	// Se abre el fichero de salida secuencial
	fDataOut = fopen(argv[4], "wb");
	if (fDataOut == NULL)
	{
		printf("Error abriendo el fichero de salida\n");
		return -1;
	}

	// Se escriben los datos del filtrado secuencial
	for (iPos = 0; iPos < iDataCount; iPos++)
	{
		data = (short)floor(sec_output[iPos]);
		fwrite((void*)&data, sizeof(short), 1, fDataOut);
	}
	fclose(fDataOut);

	//////////////////////////////////////////////////////////////////////////////////////////////////////////
	//											Paso 12 													//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////

	// Comparar la salida de la ejecución secuencial y paralela
	for (iPos = 0; iPos < iDataCount; iPos++)
	{
		if (floor(sec_output[iPos]) != floor(host_output[iPos]))
		{
			printf("\nLas salidas no son iguales en posición - %d: %d - %d", iPos, (int)floor(sec_output[iPos]), (int)floor(host_output[iPos]));
		}
	}

	//////////////////////////////////////////////////////////////////////////////////////////////////////////
	//											Paso 13 													//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////

	// Liberar la memoria del Host
	/*              COMPLETAR                     */

	free(host_input);
	free(host_output);
	free(sec_output);
	//////////////////////////////////////////////////////////////////////////////////////////////////////////
	//											Paso 14 													//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////

	// Librerar la memoria solicitada en el Device
	/*              COMPLETAR                     */

	cudaFree(dev_input);
	cudaFree(dev_output);

	return 0;
}
