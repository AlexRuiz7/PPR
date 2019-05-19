// ###########################################################################
// Autor: Alejandro Ruiz Becerra
//
// Descripción:
//
//   Modificar el programa solución del cálculo paralelo del número π
//   (3.3 Cálculo de PI) para que los subintervalos de trabajo sean distribuidos
//   por bloques en lugar de cı́clicamente entre los procesos. Por ejemplo, si
//   tuviéramos 3 procesos y n = 11 (número de subintervalos), el proceso 0
//   deberı́a aproximar las integrales numéricas en los primeros 4 subintervalos
//   consecutivos (1,2,3,4), el proceso 1 calcuları́a las integrales en los
//   siguientes 4 subintervalos (5,6,7,8,) y el proceso 2 calcuları́a los últimos
//   tres (9,10,11). Se recomienda empezar derivando matemáticamente un método
//   general para repartir por bloques n subintervalos entre P procesos para
//   cualquier n entero positivo. Modificarlo también la solución para que la
//   aproximación a π se obtenga en todos los procesos.
//
// ###########################################################################

#include <iostream>
#include <cmath>
#include "mpi.h"

using namespace std;

int main(int argc, char* argv[]) {

  int rank, size, n;

  double l_pi, g_pi, h, sum;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if(rank == 0) {
    cout << "Introduce la precisión para el cálculo de PI (n>0): ";
    cin >> n;
  }

  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if(n <= 0) {
    MPI_Finalize();
    return -1;
  }
  else {
    h = 1.0 / (double) n;
    sum = 0.0;
    double x = 0.0;

    // Gestión de tamaño de bloques
    int blockSize = ceil( (double) n / size);
    int lastBlockSize = (n % size != 0) ? (n - (size-1)*blockSize) : blockSize;
    int blockStart = rank * blockSize + 1;
    int blockEnd = (rank != size - 1) ? (blockStart-1 + blockSize) : (blockStart-1 + lastBlockSize);

    // Cálculo del intervalo
    for (int i = blockStart; i <= blockEnd; i++) {
      x = h * ((double) i - 0.5);
      sum += (4.0 / (1.0 + x*x));
    }
    l_pi = h * sum;

    MPI_Reduce(&l_pi, &g_pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // ENTREGABLE -- Broadcast de la aprozimación de PI
    MPI_Bcast(&g_pi, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if(rank == 0)
    cout  << "[" << rank << "] - " << "El valor aproximado de PI es " << g_pi
          << ", con error de " << fabs(g_pi - M_PI) << endl;
  }


  MPI_Finalize();
  return 0;
}
