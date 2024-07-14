#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <assert.h>
#include <stdlib.h>

#define ERROR_CHECK { cudaError_t err; \
  if ((err = cudaGetLastError()) != cudaSuccess) { \
    printf("CUDA error: %s, line %d\n", cudaGetErrorString(err), __LINE__);}}

// Hebras por dimensión de cada bloque
#define THREADS 32


// Cabecera de fichero BMP sin la signature
struct BMPHEADER
{
	unsigned int iFileSize;
	unsigned int iReserved;
	unsigned int iFileOffsetToPixelArray;

	unsigned int iDIBHeaderSize;
	unsigned int iWidth;
	unsigned int iHeigth;
	unsigned short iColorPlanes;
	unsigned short iBitsPerPixel;
	unsigned int iCompression;
	unsigned int iSizeOfDataWithPadding;
	unsigned int iPrintResolutionHorizontal;
	unsigned int iPrintResolutionVertical;
	unsigned int iColorsInPalete;
	unsigned int iImportantColors;
};

// Estructura para un pixel (B G R) (1 byte x 1 byte x 1 byte)
struct PIXEL
{
	unsigned char B;
	unsigned char G;
	unsigned char R;
};



// Negativo paralelo
__global__ void negativoParalelo(PIXEL* dev_input, PIXEL* dev_output, unsigned iWidth)
{
	// COMPLETAR
	unsigned int iCol = blockIdx.x * blockDim.y + threadIdx.x; //Calculamos la columna
	unsigned int iFila = blockIdx.y * blockDim.y + threadIdx.y; //Calculamos la fila

	dev_output[iFila * iWidth + iCol].R = 255 - dev_input[iFila * iWidth + iCol].R;
	dev_output[iFila * iWidth + iCol].G = 255 - dev_input[iFila * iWidth + iCol].G;
	dev_output[iFila * iWidth + iCol].B = 255 - dev_input[iFila * iWidth + iCol].B;
}


// Negativo secuencial
void negativoSecuencial(PIXEL* host_input, PIXEL* sec_output, unsigned iWidth, unsigned int iHeigth)
{
	// COMPLETAR
	unsigned int ejeX, ejeY;

	//Realizamos inversion pixel a pixel
	for (ejeX = 0; ejeX < iWidth; ejeX++) //Ancho
	{
		for (ejeY = 0; ejeY < iHeigth; ejeY++) //Alto
		{
			sec_output[ejeX * iHeigth + ejeY].R = 255 - host_input[ejeX * iHeigth + ejeY].R; //Componente Rojo a negativo
			sec_output[ejeX * iHeigth + ejeY].G = 255 - host_input[ejeX * iHeigth + ejeY].G; //Componente Verde a negativo
			sec_output[ejeX * iHeigth + ejeY].B = 255 - host_input[ejeX * iHeigth + ejeY].B; //Componente Azul a negativo 
		}
	}
}


int main(int argc, char** argv)
{
	struct BMPHEADER BMPheader;
	char signature[2];
	size_t dataToRead;
	struct PIXEL pixel;

	// Variables para las imagenes en el HOST
	struct PIXEL* host_input, * host_output, * sec_output;

	// Variables para las imagenes en el DEVICE
	struct PIXEL* dev_input, * dev_output;

	if (argc != 4)
	{
		printf("USO:\nFiltro2D <fichero imagen> <fichero salida paralelo> <fichero salida secuencial\n");
		return -1;
	}

	// Se abre el fichero de la imagen
	FILE* fData = fopen(argv[1], "rb");
	if (fData == NULL)
	{
		printf("Error abriendo el fichero de datos\n");
		return -1;
	}

	// Se lee la signature y se comprueba que realmente es un fichero BMP
	fread((void*)signature, sizeof(char), 2, fData);
	if (signature[0] != 'B' || signature[1] != 'M')
	{
		printf("El fichero de la imagen no es BMP\n");
		fclose(fData);
		return -1;
	}

	// Se lee la cabecera del fichero BMP
	fread((void*)&BMPheader, sizeof(BMPHEADER), 1, fData);

	// Los datos a leer serán el alto x ancho
	dataToRead = BMPheader.iWidth * BMPheader.iHeigth;

	// Se pide memoria para las variables en el host
	// COMPLETAR
	//
	host_input = (struct PIXEL*)malloc(dataToRead * sizeof(struct PIXEL));
	host_output = (struct PIXEL*)malloc(dataToRead * sizeof(struct PIXEL));
	sec_output = (struct PIXEL*)malloc(dataToRead * sizeof(struct PIXEL));

	// Se cargan los datos de cada pixel en memoria del Host
	for (int iPos = 0; iPos < (int)dataToRead; iPos++)
	{
		fread((void*)&pixel, sizeof(PIXEL), 1, fData);
		host_input[iPos] = pixel;
	}
	fclose(fData);


	// Pedir memoria en el Device
	// COMPLETAR
	cudaMalloc((void**)&dev_input, dataToRead * sizeof(struct PIXEL));
	cudaMalloc((void**)&dev_output, dataToRead * sizeof(struct PIXEL));
	ERROR_CHECK;

	// Transferir los datos del Host al Device
	// COMPLETAR
	cudaMemcpy(dev_input, host_input, dataToRead * sizeof(struct PIXEL), cudaMemcpyHostToDevice);
	ERROR_CHECK;

	
	dim3 grid(BMPheader.iWidth/ THREADS, BMPheader.iHeigth/ THREADS);/* COMPLETAR */
	dim3 block(THREADS, THREADS);/* COMPLETAR */

	// Se llama al filtrado paralelo CUDA
	// COMPLETAR
	negativoParalelo <<<grid, block >>> (dev_input, dev_output, BMPheader.iWidth);
	ERROR_CHECK;

	// Copiar el resultado final del Device al Host
	// COMPLETAR
	cudaMemcpy(host_output, dev_output, dataToRead * sizeof(struct PIXEL), cudaMemcpyDeviceToHost);
	ERROR_CHECK;

	// Llamada al negativo secuencial
	negativoSecuencial(host_input, sec_output, BMPheader.iWidth, BMPheader.iHeigth);

	// Se abre el fichero de salida paralela
	FILE* fDataOut = fopen(argv[2], "wb");
	if (fDataOut == NULL)
	{
		printf("Error abriendo el fichero de salida\n");
		return -1;
	}

	// Se escribe la signature
	fwrite((void*)&signature, sizeof(char), 2, fDataOut);
	// Se escribe la cabecera
	fwrite((void*)&BMPheader, sizeof(BMPHEADER), 1, fDataOut);
	// Se escriben los datos del negativo paralelo
	for (int iPos = 0; iPos < (int)dataToRead; iPos++)
	{
		pixel.R = host_output[iPos].R;
		pixel.G = host_output[iPos].G;
		pixel.B = host_output[iPos].B;
		fwrite((void*)&pixel, sizeof(PIXEL), 1, fDataOut);
	}
	fclose(fDataOut);

	// Se abre el fichero de salida secuencial
	fDataOut = fopen(argv[3], "wb");
	if (fDataOut == NULL)
	{
		printf("Error abriendo el fichero de salida\n");
		return -1;
	}

	// Se escribe la signature
	fwrite((void*)&signature, sizeof(char), 2, fDataOut);
	// Se escribe la cabecera
	fwrite((void*)&BMPheader, sizeof(BMPHEADER), 1, fDataOut);
	// Se escriben los datos del negativo secuencial
	for (int iPos = 0; iPos < (int)dataToRead; iPos++)
	{
		pixel.R = sec_output[iPos].R;
		pixel.G = sec_output[iPos].G;
		pixel.B = sec_output[iPos].B;
		fwrite((void*)&pixel, sizeof(PIXEL), 1, fDataOut);
	}
	fclose(fDataOut);

	// Liberar la memoria del Host
	// COMPLETAR
	cudaFree(host_input);
	cudaFree(host_output);
	cudaFree(sec_output);

	// Librerar la memoria solicitada en el Device
	// COMPLETAR
	cudaFree(dev_input);
	cudaFree(dev_output);
	cudaDeviceReset();

	return 0;
}