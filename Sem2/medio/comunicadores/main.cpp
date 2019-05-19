#include <iostream>
#include "mpi.h"

using namespace std;

int main(int argc, char* argv[]) {

  int rank, size;
  int pi_rank, pi_size;
  int in_rank, in_size;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  MPI_Comm comm_par_impar, comm_inverso;

  int A = 0, B = 0;

  if(rank == 0){
    A = 2000;
    B = 1;
  }

  // Creación de los nuevos comunicadores
  int color = rank % 2;
  MPI_Comm_split(MPI_COMM_WORLD, color, rank, &comm_par_impar);
  MPI_Comm_split(MPI_COMM_WORLD, 0, -rank, &comm_inverso);

  // Obtención de identificadores en y tamaño del comunicador par-impar
  MPI_Comm_size(comm_par_impar, &pi_size);
  MPI_Comm_rank(comm_par_impar, &pi_rank);

  // Obtención de identificadores en y tamaño del comunicador inverso
  MPI_Comm_size(comm_inverso, &in_size);
  MPI_Comm_rank(comm_inverso, &in_rank);


  MPI_Bcast(&A, 1, MPI_INT, 0, comm_par_impar);
  MPI_Bcast(&B, 1, MPI_INT, size - 1, comm_inverso);

  cout << "[" << rank << "/" << size << "] -- MPI_COMM_WORLD" << endl
   << "[" << pi_rank << "/" << pi_size << "] -- COMM_PAR_IMPAR -> A: " << A <<  endl
   << "[" << in_rank << "/" << in_size << "] -- COMM_INVERSO -> B: " << B <<  endl << endl;

  MPI_Finalize();
  return 0;
}
