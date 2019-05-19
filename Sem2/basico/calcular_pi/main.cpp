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


    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
  }

  if(n<=0) {
    MPI_Finalize();
    return -1;
  }
  else {
    h = 1.0 / (double) n;
    sum = 0.0;

    double x = 0.0;
    for (int i = rank + 1; i <= n; i += size) {
      x = h * ((double) i - 0.5);
      sum += (4.0 / (1.0 + x*x));
    }

    l_pi = h * sum;

    MPI_Reduce(&l_pi, &g_pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if(rank == 0)
    cout << "El valor aproximado de PI es " << g_pi << ", con error de " << fabs(g_pi - M_PI) << endl;
  }


  MPI_Finalize();
  return 0;
}
