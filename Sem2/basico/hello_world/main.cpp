#include <iostream>
#include "mpi.h"

using namespace std;

int main(int argc, char *argv[]) {

  int rank, size;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  cout << "¡Hola Mundo soy proceso " << rank << " de " << size << " que somos";
  cout << endl;

  MPI_Finalize();

  return 0;
}
