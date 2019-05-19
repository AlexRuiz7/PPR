// ###########################################################################
// Autor: Alejandro Ruiz Becerra
//
// Descripción:
//
//  Modificar el programa solución del ejercicio 4.4 Comunicadores para que
//  también se realice un Scatter de un vector de enteros desde el proceso 1 del
//  comunicador global a todos los procesos impares de dicho comunicador. Los
//  valores de este vector se escogen arbitrariamente en el proceso 0 (ojo, en
//  el proceso con rango 0 del comunicador de rangos impares que es el proceso
//  1 de MPI Comm World), pero su tamaño debe ser igual número de procesos
//  impares en el comunicador global. El reparto asignará un elemento de dicho
//  vector a cada proceso impar del comunicador global. Se recomienda usar el
//  comunicador de impares para realizar esta tarea.
//
// ###########################################################################

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

  // ENTREGABLE -- Scatter de vector de enteros de tamaño pi_size por el proceso
  //            -- pi_rank == 0 --> rank == 1

  int *V = new int[pi_size];
  int recibido = 0;

  if(pi_rank == 0  /* rank == 1 */) {
    for(int i=0; i<pi_size; i++)
      V[i] = (i+1) * 10;
  }
  // Repartimos el vector. 1 elemento a cada proceso impar
  MPI_Scatter(&V[0], 1, MPI_INT, &recibido, 1, MPI_INT, pi_rank, comm_par_impar);

  cout << "[" << rank << "] -- MPI_COMM_WORLD\t" << "[" << pi_rank  << "] -- COMM_PAR_IMPAR -> R: " << recibido <<  endl;


  MPI_Bcast(&A, 1, MPI_INT, 0, comm_par_impar);
  MPI_Bcast(&B, 1, MPI_INT, size - 1, comm_inverso);

  // cout << "[" << rank << "/" << size << "] -- MPI_COMM_WORLD" << endl
  //  << "[" << pi_rank << "/" << pi_size << "] -- COMM_PAR_IMPAR -> A: " << A <<  endl
  //  << "[" << in_rank << "/" << in_size << "] -- COMM_INVERSO -> B: " << B <<  endl << endl;

  MPI_Comm_free(&comm_par_impar);
  MPI_Comm_free(&comm_inverso);
  MPI_Finalize();
  return 0;
}
