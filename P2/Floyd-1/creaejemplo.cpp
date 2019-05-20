#include <fstream>
#include <string>
#include <cstdlib>

using namespace std;

int main(int argc, char * argv[]){

	int vertices;
	ofstream file;
	int aleatorio;
	string nombre = "input";

  if(argc != 2) {
    printf("Uso: <ejecutable> <número_de_vértices>\n");
    exit(-1);
  }

	vertices = atoi(argv[1]);
	nombre += string(argv[1]);

	int *matriz;

	matriz = new int [vertices * vertices];

  int index = 0;
	for(int i=0; i<vertices; i++) {
		for(int j=0; j<vertices; j++) {
      index = i * vertices + j;

			if(i==j) {
				matriz[index] = -1;
			}
			else {
				aleatorio = (rand() % (vertices*2)) + 1;
				// (25% de huecos)
				if ( (aleatorio > (vertices*11/20)) || (aleatorio < (vertices*9/20)) ) {
					aleatorio=0;
				}
				matriz[index] = aleatorio;
			}
		}
	}

	file.open(nombre.c_str());

	file << vertices << endl;

  index = 0;
	for(int i=0; i<vertices; i++) {
		for(int j=0; j<vertices; j++) {
      index = i * vertices + j;
			if(matriz[index] > 0) {
				file << i << " " << j << " " << matriz[index] << endl;
			}
		}
	}

	file.close();
	return 0;
}
