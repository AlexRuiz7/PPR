#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <string>
#include <locale>

using namespace std;

/*****************************************************************************/

__global__ void calcularC (const float * A, const float * B, float * C, const int size) {
  uint block_start = blockIdx.x * blockDim.x;
  uint block_end = block_start + blockDim.x;
  uint i = block_start + threadIdx.x;

  if (i < size) {
    C[i]=0;

    for(int j=block_start; j<block_end; j++) {
      float a = A[j]*i;
      if((int) ceil(a) % 2 == 0)
        C[i] += a + B[j];
      else
        C[i] += a - B[j];
    }
  }
}

/*****************************************************************************/

__global__ void calcularC_shared (const float * A, const float * B, float * C, const int size) {
  extern __shared__ float sdata[];
  uint block_start = blockIdx.x * blockDim.x;
  uint i = block_start + threadIdx.x;
  uint tid = threadIdx.x;

  if (i < size) {
    C[i]=0;
    sdata[tid] = A[i];
    sdata[tid+blockDim.x] = B[i];
    __syncthreads();

    for(int j=0; j<blockDim.x; j++) {
      float a = sdata[j]*i;
      if((int) ceil(a) % 2 == 0)
        C[i] += a + sdata[j+blockDim.x];
      else
        C[i] += a - sdata[j+blockDim.x];
      // __syncthreads();
    }
  }
}

/*****************************************************************************/


// https://github.com/mark-poscablo/gpu-sum-reduction/blob/master/sum_reduction/reduce.cu
__global__ void calcularD(const float *C, float *D, const uint size) {
	extern __shared__ float sdata[];

	uint tid = threadIdx.x;
	uint i = blockIdx.x * (blockDim.x*2) + threadIdx.x;

	// Cada hebra carga un elemento desde memoria global a memoria compartida
	sdata[tid] = 0;
	if (i < size)
		sdata[tid] = C[i] + C[i+blockDim.x];

	__syncthreads();

	// Reducir en memoria compartida
	for(uint s = blockDim.x>>1; s > 0; s >>= 1) {
		if(tid < s)
			sdata[tid] += sdata[tid+s];

		// Esperar al resto de hebras para comenzar la nueva etapa
		__syncthreads();
	}

	// Escribir resultado de este bloque en memoria global
	if (tid == 0)
		D[blockIdx.x] = sdata[0];
}

//**************************************************************************//

/*****************************************************************************/

__global__ void calcularMax (const float *g_idata, float *g_odata, const int size) {
  extern __shared__ float sdata[];

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

/*****************************************************************************/

string comprobarIntegridad(float *vec1, float *vec2, int len) {
  for (int i=0; i<len; i++)
    if ((floor(vec1[i])-floor(vec2[i]))>0.9)
      return "fallo " + to_string(vec1[i]) + " : " + to_string(vec2[i]);

  return "OK";
}

/*****************************************************************************/


void imprimir(float *v, int len, char letra) {
  cout << "................................." << endl;
  for (int i=0; i<len;i++)
    cout << letra << "[" << i << "]=" << v[i] << endl;
}

/*****************************************************************************/

int main(int argc, char *argv[]) {
  int Bsize, NBlocks;

  locale mylocale("");
	cout.imbue(mylocale);

  if (argc != 3) {
    cout << "Uso: transformacion Num_bloques Tam_bloque  " << endl;
    return(0);
  }
  else {
    NBlocks = atoi(argv[1]);
    Bsize = atoi(argv[2]);
  }

  const int N = Bsize*NBlocks;
  cout << endl << "N=" << N << "= " << Bsize << "*" << NBlocks << endl << endl;

  //* pointers to host memory */
  float *A, *B, *C, *D;

  //* Allocate arrays a, b and c on host*/
  A = new float[N];
  B = new float[N];
  C = new float[N];
  D = new float[NBlocks];
  float mx; // maximum of C

  /* Initialize arrays */
  for (int i=0; i<N; i++) {
    A[i] = (float) (1  -(i%100)*0.001);
    B[i] = (float) (0.5+(i%10) *0.1  );
    C[i] = 0;
  }

  /**************************************************************************/
  /*	GPU PHASE
  /*
  /**************************************************************************/

  // Pointers to device memory
  float *d_A = NULL, *d_B = NULL, *d_C = NULL, *d_D = NULL, *d_max = NULL;
  // Allocate device memory
  cudaMalloc ((void **) &d_A, sizeof(float)*N);
  cudaMalloc ((void **) &d_B, sizeof(float)*N);
  cudaMalloc ((void **) &d_C, sizeof(float)*N);
  cudaMalloc ((void **) &d_D, sizeof(float)*NBlocks);
  cudaMalloc ((void **) &d_max, sizeof(float));
  // Copy data from host to device
  cudaMemcpy(d_A, A, sizeof(float)*N, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, sizeof(float)*N, cudaMemcpyHostToDevice);
  cudaMemcpy(d_C, C, sizeof(float)*N, cudaMemcpyHostToDevice);
  // Local device storage
  float *gpu_C = new float[N];
  float *gpu_D = new float[NBlocks];
  float *gpu_max = new float;

  // Time measurement
  double t1=clock();

  /**
  * Calcular C
  */
  // Kernel call with NBlocks of Bsize threads each.
  calcularC <<<NBlocks, Bsize>>> (d_A, d_B, d_C, N);
  // calcularC_shared <<<NBlocks, Bsize, 2*Bsize*sizeof(float)>>> (d_A, d_B, d_C, N);
  // Copy data from device to host
  cudaMemcpy(gpu_C, d_C, N*sizeof(float), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  /**
  * Calcular D
  */
  // Kernel call
  calcularD <<<NBlocks, ceil(Bsize>>1), Bsize*sizeof(float)>>> (d_C, d_D, N);
  cudaMemcpy(gpu_D, d_D, NBlocks*sizeof(float), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  /**
  * Calcular Max
  */
  // Calcular máximos locales a cada bloque
  calcularMax <<<NBlocks, ceil(Bsize>>1), Bsize*sizeof(float)>>> (d_C, d_D, N);
  // cudaMemcpy(gpu_D, d_D, NBlocks*sizeof(float), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  // Calcular máximo global
  calcularMax<<<1, ceil((int)NBlocks>>1), NBlocks*sizeof(float)>>> (d_D, d_max, NBlocks);
  cudaMemcpy(gpu_max, d_max, sizeof(float), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  double t2=clock();
  t2=(t2-t1)/CLOCKS_PER_SEC;

  cout << "[GPU] tiempo -> " << t2 << endl << endl;

  // Free device memory
  cudaFree(d_A);    cudaFree(d_B);    cudaFree(d_C);    cudaFree(d_D);    cudaFree(d_max);

  /**************************************************************************/
  /*	CPU PHASE
  /*
  /**************************************************************************/

  // Compute C[i], d[K] and mx
  for (int k=0; k<NBlocks; k++) {
    int istart=k*Bsize;
    int iend = istart+Bsize;
    D[k]=0.0;
    for (int i=istart; i<iend; i++) {
      C[i]=0.0;
      for (int j=istart; j<iend; j++) {
        float a = A[j]*i;
        if ((int)ceil(a) % 2 ==0)
          C[i] += a + B[j];
        else
          C[i] += a - B[j];
      }
      D[k] += C[i];
      mx = (i==1) ? C[0] : max(C[i],mx);
     }
  }

  t2=clock();
  t2=(t2-t1)/CLOCKS_PER_SEC;

  cout << "[CPU] tiempo -> " << t2 << endl << endl;

  /**************************************************************************/
  /*	CHECK PHASE
  /*
  /**************************************************************************/
  cout << "................................." << endl;
  cout << "Comprobando integridad de cálculos...\n";
  cout << "\t[CHECK] Comprobando C...  " << comprobarIntegridad(C, gpu_C, N) << endl;
  cout << "\t[CHECK] Comprobando D...  " << comprobarIntegridad(D, gpu_D, NBlocks) << endl;
  cout << "\t[CHECK] Comprobando Max...  ";
  (mx==*gpu_max) ? cout << "OK\n" : cout << "fallo " << to_string(*gpu_max) << endl;

  // imprimir(A, N, 'A');
  // imprimir(B, N, 'B');
  // imprimir(C, N, 'C');
  // imprimir(D, NBlocks, 'D');
  // imprimir(gpu_C, N, 'C');
  // imprimir(gpu_D, NBlocks, 'D');
  // imprimir(gpu_max, 1, 'M');

  cout << "................................." << endl;
  cout << "El valor máximo en C es:  " << to_string(mx) << endl;


  /* Free the memory */
  delete(A);        delete(B);        delete(C);        delete(D);
  delete(gpu_C);    delete(gpu_D);

  return 0;
}
