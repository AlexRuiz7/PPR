#include "Graph.h"
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <string.h>

using namespace std;

/***********************************************************************/

Graph::Graph ()	{}

/***********************************************************************/

void Graph::fija_nverts (const int nverts) {
  A = new int[nverts * nverts];
  vertices = nverts;
}

/***********************************************************************/

void Graph::inserta_arista(const int vertA, const int vertB, const int edge) {
  A[vertA * vertices + vertB] = edge;
}

/***********************************************************************/

int Graph::arista(const int ptA,const int ptB) {
  return A[ptA * vertices + ptB];
}

/***********************************************************************/

void Graph::imprime() {
  int i, j, vij;

  for(i=0; i<vertices; i++) {
    cout << "A[" << i << ",*]= ";

    int index = 0;
    for(j=0; j<vertices; j++) {
      index = i*vertices + j;
      (A[index] == INF) ? (cout << "INF") : (cout << A[index]);

      (j < vertices-1) ? (cout << ",") : (cout << endl);
    }
  }
}

/***********************************************************************/

void Graph::lee(char *filename) {
  #define BUF_SIZE 100
  std::ifstream infile(filename);

  if (!infile) {
    cerr << "Nombre de archivo inválido \"" << filename << "\" !!" << endl;
    cerr << "Saliendo........." << endl;
    exit(-1);
	}

  // Obtén el número de vértices
  char buf[BUF_SIZE];
  infile.getline(buf, BUF_SIZE, '\n');
  vertices = atoi(buf);
  A = new int[ vertices*vertices ];

  int i, j;
  for(i=0; i<vertices; i++)
    for(j=0; j<vertices; j++)
      (i==j) ? (A[ i*vertices + j ] = 0) : (A[ i*vertices + j ] = INF);

  while (infile.getline(buf, BUF_SIZE) && infile.good() && !infile.eof()) {
    char *vertname2 = strpbrk(buf, " \t");
    *vertname2++ = '\0';
    char *buf2 = strpbrk(vertname2, " \t");
    *buf2++ = '\0';
    int weight = atoi(buf2);
    i = atoi(buf);
    j = atoi(vertname2);
    A[ i*vertices + j ] = weight;
  }

}

/***********************************************************************/
