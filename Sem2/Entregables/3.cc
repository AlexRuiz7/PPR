// ###########################################################################
// Autor: Alejandro Ruiz Becerra
//
// Descripción:
//
//   Modificar el programa solución del cálculo del producto escalar de dos
//   vectores (4.1 Producto Escalar) para que cada proceso inicialice por su
//   cuenta su parte correspondiente del vector B de forma local, de tal forma
//   que no haya necesidad de inicializar todo el vector B en el proceso 0 y
//   repartir sus bloques entre los procesos.
//
// ###########################################################################

#include <iostream>
#include <vector>
#include "mpi.h"

using namespace std;

int main(int argc, char *argv[]) {

  int rank, size, tam_vec;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if(argc != 2) {
    if(rank == 0) {
      cout << "\nUso: <ejecutable> <cantidad>" << endl;
      exit(-1);
    }
  }
  else{
    if(rank == 0) {
      tam_vec = size * atoi(argv[1]);
      if(tam_vec < size) {
        tam_vec = size * 100;
        cout << "Cantidad cambiada a " << tam_vec << endl;
      }
    }
  }

  vector<long> VectorA, /*VectorB,*/ l_VectorA, l_VectorB;
  VectorA.resize(tam_vec);
  // VectorB.resize(tam_vec);
  int tam_trozo = tam_vec/size;
  l_VectorA.resize(tam_trozo);
  l_VectorB.resize(tam_trozo);

  if(rank == 0) {
    for(long i=0; i<tam_vec; i++) {
      VectorA[i] = i+1;         // 1, 2, 3, ... , tam_vec
      // VectorB[i] = (i+1) * 10;  // 10, 20, 30, ... , tam_vec * 10
    }
  }

  // Repartimos el vector A
  MPI_Scatter(&VectorA[0], tam_trozo, MPI_LONG, &l_VectorA[0], tam_trozo, MPI_LONG, 0, MPI_COMM_WORLD);
  // Repartimos el vector B
  // MPI_Scatter(&VectorB[0], tam_trozo, MPI_LONG, &l_VectorB[0], tam_trozo, MPI_LONG, 0, MPI_COMM_WORLD);

  /////////////////////////////////////////
  // ENTREGABLE -- inicialización local  //
  /////////////////////////////////////////
  for(int i=0; i<tam_trozo; i++) {
    l_VectorB[i] = l_VectorA[i] * 10;  // 10, 20, 30, ... , tam_vec * 10
  }

  long producto = 0;
  for(int i=0; i<tam_trozo; i++){
    producto += l_VectorA[i] * l_VectorB[i];
  }

  long total = 0;

  MPI_Reduce(&producto, &total, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

  if(rank == 0)
    cout << "Total: " << total << endl;

  MPI_Finalize();
  return 0;
}
