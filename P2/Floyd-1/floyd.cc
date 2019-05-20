#include <iostream>
#include <fstream>
#include <string.h>
#include <time.h>
#include "Graph.h"
#include <omp.h>

using namespace std;

/**************************************************************************/

void floyd_Secuencial(const uint nverts, Graph *A) {
	// BUCLE PPAL DEL ALGORITMO
	for(int k = 0; k < nverts; k++) {
    for(int i = 0; i < nverts; i++) {
      for(int j = 0; j < nverts; j++) {
        if (i!=j && i!=k && j!=k) {
          int vikj = min(A->arista(i,k) + A->arista(k,j), A->arista(i,j));
          A->inserta_arista(i, j, vikj);
        }
      }
    }
  }
}

/**************************************************************************/

void floyd_Paralelo(const uint N, const int P, Graph *B) {
  int chunk = N / P;
  int i, j, vikj;
  int filaK[N];

  int blocksize = chunk * N;
  int blockStart = omp_get_thread_num()*chunk;
  int blockEnd = blockStart + chunk;

  // BUCLE PPAL DEL ALGORITMO
  #pragma omp parallel private (filaK, i, j, vikj)
	for(int k = 0; k < N; k++) {
    // Copia privada de la fila K
    for(int x=0; x < N; x++)
      filaK[x] = B->arista(k, x);

    #pragma omp for
    for(i = blockStart; i < blockEnd; i++) {
      for(j = 0; j < N; j++) {
        if (i!=j && i!=k && j!=k) {
          vikj = min(B->arista(i,k) + filaK[j], B->arista(i,j));
          B->inserta_arista(i, j, vikj);
        }
      }
    }
  }

}

/**************************************************************************/

void initOMP(const int N, const int P) {
  cout << "Max: " << omp_get_max_threads() << endl;
  omp_set_dynamic(false);
  if (P < N) {
    omp_set_num_threads(P);
    cout << "Cambiado a " << P << endl;
  }
  else {
    omp_set_num_threads(N);
    cout << "Cambiado a " << N << endl;
  }

  #pragma omp parallel
  {
    #pragma omp critical
    cout << "ID: " << omp_get_thread_num() << " de " << omp_get_num_threads() << endl;
  }
  cout << endl;
}

/**************************************************************************/

void check( Graph A,  Graph B) {
  for(uint i=0; i<A.vertices; i++)
    for (uint j=0; j<A.vertices; j++)
      if(A.arista(i, j) != B.arista(i, j))
        cout << "[ERROR] - " << A.arista(i, j) << " != " << B.arista(i, j) << endl;
}

/**************************************************************************/

int main (int argc, char *argv[]) {

  // Usar coma(,) como delimitador decimal
	locale mylocale("");
	cout.imbue(mylocale);

  if (argc != 3) {
    cerr << "Sintaxis: " << argv[0] << " <archivo de grafo> <número de hebras>" << endl;
  	return(-1);
	}

  // Número de procesos / hebras
  const int P = atoi(argv[2]);

  // G: grafo inicial, A: copia secuencial, B: copia paralelo
  Graph G, A, B;
  G.lee(argv[1]);
  int nverts = G.vertices;
  A = B = G;
  // cout << "El Grafo de entrada es:"<<endl;
  // G.imprime();

  initOMP(nverts, P);

  double t1 = clock();
  floyd_Secuencial(nverts, &A);
  double t2 = clock();
  double t_sec = (t2 - t1) / CLOCKS_PER_SEC;
  cout << "Tiempo secuencial = " << t_sec << endl;

  t1 = clock();
  floyd_Paralelo(nverts, P, &B);
  t2 = clock();
  double t_prl = (t2 - t1) / CLOCKS_PER_SEC;
  cout << "Tiempo paralelo = " << t_prl << endl;

  cout << "Ganancia = " << t_sec / t_prl << endl;

  cout << endl << "Comprobando integridad de los resultados..." << endl;
  check(A, B);
  // A.guardar("A");
  // B.guardar("B");

  return(0);
}
