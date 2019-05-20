#include <iostream>
#include <fstream>
#include <locale>
#include <string.h>
#include <sys/time.h>
#include "Graph.h"

#define VERBOSE 0
#define REDUCE 1

// #define blocksize 64	// 8x8
// #define blocksize 128	// 12x12 (frag.)
// #define blocksize 256	// 16x16
// #define blocksize 512	// 23x23 (frag.)
#define blocksize 1024	// 32x32

using namespace std;

//**************************************************************************//

double cpuSecond() {
	struct timeval tp;
	gettimeofday(&tp, NULL);
	return((double)tp.tv_sec + (double)tp.tv_usec*1e-6);
}

//**************************************************************************//

void floydCPU(uint niters, uint nverts, int * A) {
	// BUCLE PPAL DEL ALGORITMO
	int inj, in, kn;
	for(int k = 0; k < niters; k++) {
    kn = k * nverts;
	  for(int i=0;i<nverts;i++) {
			in = i * nverts;
			for(int j = 0; j < nverts; j++)
   			if (i!=j && i!=k && j!=k){
          inj = in + j;
		 	    A[inj] = min(A[in+k] + A[kn+j], A[inj]);
        }
	  }
  }
}

//**************************************************************************//

__global__ void floyd1D(int * M, const uint nverts, const uint k) {
	int ij = threadIdx.x + blockDim.x * blockIdx.x;
  if (ij < nverts * nverts) {
		int Mij = M[ij];
    int i= ij / nverts;
    int j= ij - i * nverts;
    if (i != j && i != k && j != k) {
			int Mikj = M[i * nverts + k] + M[k * nverts + j];
    	Mij = (Mij > Mikj) ? Mikj : Mij;
    	M[ij] = Mij;
		}
  }
}

//**************************************************************************//

__global__ void floyd2D(int * M, const uint nverts, const uint k) {
	uint i = blockIdx.y * blockDim.y + threadIdx.y; // fila de la matriz
  uint j = blockIdx.x * blockDim.x + threadIdx.x; // columna de la matriz

  if (i < nverts && j < nverts) {
     int ij = i*nverts+j; 	// indice de la matriz
     int Mij = M[ij];
     if (i != j && i != k && j != k){
			uint Mikj = M[i*nverts+k] + M[k*nverts+j];
			Mij = (Mij > Mikj) ? Mikj : Mij;
			M[ij] = Mij;
     }
	 }
}

//**************************************************************************//

void imprime(int * A, uint vertices) {
	int i,j;
	 for(i=0;i<vertices;i++) {
		 cout << "A["<<i << ",*]= ";

	  	for(j=0;j<vertices;j++) {
	      if (A[i*vertices+j]==INF)
	        cout << "INF";
	      else
	        cout << A[i*vertices+j];
	      if (j<vertices-1)
	        cout << ",";
	      else
	        cout << endl;
			}
	 }
}

//**************************************************************************//

// https://github.com/mark-poscablo/gpu-sum-reduction/blob/master/sum_reduction/reduce.cu
__global__ void reduce3(int *g_idata, int *g_odata, const uint size) {
	extern __shared__ int sdata[];

	uint tid = threadIdx.x;
	uint i = blockIdx.x * (blockDim.x*2) + threadIdx.x;

	// Cada hebra carga un elemento desde memoria global a memoria compartida
	sdata[tid] = 0;
	if (i < size)
		sdata[tid] = g_idata[i] + g_idata[i+blockIdx.x];

	__syncthreads();

	// Reducir en memoria compartida
	for(uint s = blockDim.x*0.5; s > 0; s >>= 1) {
		if(tid < s)
			sdata[tid] += sdata[tid+s];

		// Esperar al resto de hebras para comenzar la nueva etapa
		__syncthreads();
	}

	// Escribir resultado de este bloque en memoria global
	if (tid == 0)
		g_odata[blockIdx.x] = sdata[0];
}

//**************************************************************************//

// https://github.com/mark-poscablo/gpu-sum-reduction/blob/master/sum_reduction/reduce.cu
__global__ void reduceMax(int *g_idata, int *g_odata, const uint size) {
	__shared__ int sdata[blocksize];

	uint tid = threadIdx.x;
	uint i = blockIdx.x * blockDim.x*2 + threadIdx.x;

	// Cada hebra carga un elemento desde memoria global a memoria compartida
	if (i < size)
		sdata[tid] = (g_idata[i] > g_idata[i+blockDim.x]) ? g_idata[i] : g_idata[i+blockDim.x];

	__syncthreads();

	// Reducir en memoria compartida
	// s = blockDim.x >> 1
	// s >>=1
	// Ambas dividen el tamaño de bloque por 2.
	for(uint s = blockDim.x>>1; s > 0; s >>= 1) {
		if(tid < s)
			if(sdata[tid] < sdata[tid+s])
				sdata[tid] = sdata[tid+s];

		// Esperar al resto de hebras para comenzar la nueva etapa
		__syncthreads();
	}

	// Escribir resultado de este bloque en memoria global
	if (tid == 0)
		g_odata[blockIdx.x] = sdata[0];
}

//**************************************************************************//

int main (int argc, char *argv[]) {

	// Usar coma como delimitador decimal
	locale mylocale("");
	cout.imbue(mylocale);

	if (argc < 2) {
		cerr << "Sintaxis: " << argv[0] << " <archivo de grafo>" << endl;
		return(-1);
	}

  //Get GPU information
  int devID;
  cudaDeviceProp props;
  cudaError_t err;
  err = cudaGetDevice(&devID);
  if(err != cudaSuccess) {
		cout << "CUDA GET DEVICE ERROR" << endl;
	}

  cudaGetDeviceProperties(&props, devID);
  printf("Device %d: \"%s\" with Compute %d.%d capability\n", devID, props.name, props.major, props.minor);

	Graph G;
	G.lee(argv[1]);// Read the Graph

	const int nverts = G.vertices;
	const int niters = nverts;
	const int nverts2 = nverts * nverts;

	cout << "\nVértices: " << nverts << " -> " << nverts2 << endl;

	int *host_M = new int[nverts2];
	int size = nverts2*sizeof(int);
	int *device_M = NULL;

	err = cudaMalloc((void **) &device_M, size);
	if (err != cudaSuccess) {
		cout << "[ERROR] RESERVA" << endl;
	}

	int *A = G.Get_Matrix();

	/**************************************************************************/
	/*	1D BLOCKS GPU PHASE
	/*
	/**************************************************************************/
	double  t1 = cpuSecond();

	err = cudaMemcpy(device_M, A, size, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		cout << "[ERROR] COPIA A GPU 1D" << endl;
	}

	int threadsPerBlock = blocksize;
	int blocksPerGrid = (nverts2 + threadsPerBlock - 1) / threadsPerBlock;
	cout << "NumBlocks: " << blocksPerGrid << endl;
	for(int k = 0; k < niters; k++) {
	  floyd1D<<<blocksPerGrid, threadsPerBlock>>>(device_M, nverts, k);
	  err = cudaGetLastError();

	  if (err != cudaSuccess) {
	  	fprintf(stderr, "Failed to launch kernel!\n");
	  	exit(EXIT_FAILURE);
		}
	}

	cudaMemcpy(host_M, device_M, size, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	double Tgpu_1D = cpuSecond()-t1;
	if(VERBOSE)
		cout << "Tiempo gastado GPU (1D)= " << Tgpu_1D << endl << endl;

	/**************************************************************************/
	/*	2D BLOCKS GPU PHASE
	/*
	/**************************************************************************/
	t1 = cpuSecond();

	err = cudaMemcpy(device_M, A, size, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		cout << "ERROR COPIA A GPU 2D" << endl;
	}

	uint threads = sqrt(blocksize);
	dim3 threadsPerBlock_2D (threads, threads);
	dim3 blocksPerGrid_2D (	ceil((float)nverts/threadsPerBlock_2D.x),
													ceil((float)nverts/threadsPerBlock_2D.y) 	);
	for(int k = 0; k < niters; k++) {
	  floyd2D <<<blocksPerGrid_2D, threadsPerBlock_2D>>> (device_M, nverts, k);
	  err = cudaGetLastError();

	  if (err != cudaSuccess) {
			fprintf(stderr, "Failed to launch kernel!");
	  	// fprintf(stderr, "Failed to launch kernel! : %s\n", cudaGetErrorString(err));
	  	exit(EXIT_FAILURE);
		}
	}

	cudaMemcpy(host_M, device_M, size, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	double Tgpu_2D = cpuSecond()-t1;
	if(VERBOSE)
		cout << "Tiempo gastado GPU (2D)= " << Tgpu_2D << endl << endl;

	/**************************************************************************/
	/*	CPU PHASE
	/*
	/**************************************************************************/
	t1 = cpuSecond();
	floydCPU(niters, nverts, A);
  double t2 = cpuSecond() - t1;
	if(VERBOSE) {
		cout << "Tiempo gastado CPU= " << t2 << endl << endl;
		cout << "Ganancia 1D= " << t2 / Tgpu_1D << endl;
		cout << "Ganancia 2D= " << t2 / Tgpu_2D << endl;
	}
	cout << t2 << "\t" << Tgpu_1D << "\t" << t2/Tgpu_1D << "\t" << Tgpu_2D;
	cout << "\t" << t2/Tgpu_2D << endl;

	// Comprobación de integridad de resultados
  for(int i = 0; i < nverts; i++) {
		for(int j = 0;j < nverts; j++)
			if (abs(host_M[i*nverts+j] - G.arista(i,j)) > 0) {
				cout << "Error (" << i << "," << j << ")   " << host_M[i*nverts+j];
				cout << "..." << G.arista(i,j) << endl;
			}
	}

	/**************************************************************************/
	/*	GPU REDUCE
	/*
	/**************************************************************************/
	if(REDUCE) {
		// Respaldar datos para el procesamiento en CPU
		int *host_M_reduce = new int[nverts2];
		memcpy(host_M_reduce, host_M, size);

		t1 = cpuSecond();

		// Llevar la matriz resultado al dispositivo
		err = cudaMemcpy(device_M, host_M, size, cudaMemcpyHostToDevice);
		if (err != cudaSuccess)
			cout << "[ERROR] COPIA A GPU REDUCTION" << endl;

		// Reserva de memoria en dispositivo para la nueva matriz resultado
		int *d_reduction = NULL;
		err = cudaMalloc((void **) &d_reduction, nverts*sizeof(int));
		if (err != cudaSuccess)
			cout << "[ERROR] RESERVA REDUCTION" << endl;

		// Reducción por filas
		reduceMax<<<nverts, nverts>>1 >>>(device_M, d_reduction, nverts2);

		// Recuperar resultado en memoria local
		err = cudaMemcpy(host_M, d_reduction, nverts*sizeof(int), cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
		if (err != cudaSuccess)
			cout << "[ERROR] COPIA A CPU REDUCTION" << endl;

		// Reserva de memoria en el dispositivo para última reducción
		int *d_max = NULL;
		err = cudaMalloc((void **) &d_max, sizeof(int));
		if (err != cudaSuccess)
			cout << "[ERROR] RESERVA REDUCTION MAX" << endl;

		// Reducción por columnas. Comparar los máximos de cada bloque
		reduceMax<<<1, nverts>>1 >>>(d_reduction, d_max, nverts);

		// Copia a memoria local del resultado final
		int *h_max = new int[1];
		err = cudaMemcpy(h_max, d_max, sizeof(int), cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
		if (err != cudaSuccess)
			cout << "[ERROR] COPIA A CPU REDUCTION MAX" << endl;

		t2 = cpuSecond() - t1;

		// uint m = 0;
		// for (int i = 0; i < niters; i++)
		// 	m = (m > host_M[i]) ? m : host_M[i];

		cout << "[GPU] Camino más largo: " << h_max[0] << endl;
		cout << "[GPU] Tiempo: " << t2 << endl << endl;

	/**************************************************************************/
	/*	CPU REDUCE
	/*
	/**************************************************************************/
		t1 = cpuSecond();

		uint m = 0;
		for (int i = 0; i < nverts2; i++)
			m = (m > host_M_reduce[i]) ? m : host_M_reduce[i];

		t2 = cpuSecond() - t1;
		cout << "[CPU] Camino más largo: " << m << endl;
		cout << "[CPU] Tiempo: " << t2 << endl << endl;

		// Liberar memoria
		delete(host_M_reduce);
		delete(h_max);
		cudaFree(d_max);
		cudaFree(d_reduction);
	}
	// Liberación de memoria y finalización
	delete(host_M);
	cudaFree(device_M);
	// cudaDeviceReset();
	return 0;
}
