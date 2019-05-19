// ###########################################################################
// Autor: Alejandro Ruiz Becerra
//
// Descripción:
//
//   Modificar el programa solución del ejercicio 3.2 Send Receive del tutorial
//   para que el proceso 0 difunda su identificador de proceso (0) al resto
//   de procesos con identificadores pares, siguiendo un anillo de procesos
//   pares, y el proceso 1 haga lo mismo con los procesos impares.
//   Se deben tener en cuenta soluciones con cualquier número de procesos.
//
// ###########################################################################

#include <iostream>
#include "mpi.h"

using namespace std;

int main(int argc, char* argv[]) {

  int rank, size, contador;
  MPI_Status estado;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Comienza el encadenamiento de mensajes pares el proceso 0.
  if (rank == 0) {
    if ( (rank + 2) < size )
      MPI_Send(&rank, 1, MPI_INT, rank+2, 0, MPI_COMM_WORLD);
  }
  // Procesos pares
  else if (rank % 2 == 0) {
    MPI_Recv(&contador, 1, MPI_INT, rank-2, 0, MPI_COMM_WORLD, &estado);

    cout << "Soy el proceso " << rank << " y he recibido " << contador << endl;
    contador++;

    if ( (rank != size-1) && ((rank + 2) < size) )
      MPI_Send(&contador, 1, MPI_INT, rank+2, 0, MPI_COMM_WORLD);
  }

  // Comienza el encadenamiento de mensajes impares el proceso 1.
  if (rank == 1) {
    if ( (rank + 2) < size )
      MPI_Send(&rank, 1, MPI_INT, rank+2, 0, MPI_COMM_WORLD);
  }
  // Procesos pares
  else if (rank % 2 == 1) {
    MPI_Recv(&contador, 1, MPI_INT, rank-2, 0, MPI_COMM_WORLD, &estado);

    cout << "Soy el proceso " << rank << " y he recibido " << contador << endl;
    contador++;

    if ( (rank != size-1) && ((rank + 2) < size) )
      MPI_Send(&contador, 1, MPI_INT, rank+2, 0, MPI_COMM_WORLD);
  }


  MPI_Finalize();
  return 0;
}
