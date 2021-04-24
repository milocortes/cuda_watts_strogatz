#include <iostream>
#include <stdio.h>
#include <time.h>
#include <math.h>

#define N 20
#define M 3
#define P 0.1
#define BLOCK_SIZE 30

using namespace std;

double serialTimer = 0.0;
float parallelTimer = 0.0;

// Definimos el arreglo en 1D que contendrá los valores de la matriz de adyacencia
int *h_value;
int *d_value;
// Definimos el arreglo en 1D que contendrá los indices de las columnas
int *h_colidx;
int *d_colidx;
// Definimos el arreglo en 1D que contendrá los indices de las filas
int *h_rowidx;
int *d_rowidx;

// Definimos los métodos
void crea_anillo_cpu();
void imprime_coo_cpu() ;
double get_random();
bool in_edges(int node, int edge);
void watts_strogatz_cpu();
void evalua_desconexion(int node,int edge_original,int edge);
void reconecta(int node,int edge_original,int edge);

int main(int argc, char const *argv[]) {

  h_value = (int*)malloc( N*M* sizeof(int));
  h_colidx = (int*)malloc( N*M* sizeof(int));
  h_rowidx = (int*)malloc( N*M* sizeof(int));

  crea_anillo_cpu();
  watts_strogatz_cpu();
  imprime_coo_cpu();
  return 0;
}

void crea_anillo_cpu() {
  for (int i = 0; i < N; i++) {
    for (int j = 1; j <= M; j++) {
      h_value[(i*M) + (j-1)] = 1 ;
      h_rowidx[(i*M) + (j-1)] = i;
      h_colidx[(i*M) +(j-1)] = ( i + j) % N;
    }
  }
}


void watts_strogatz_cpu(){
  for (int i = 0; i < N; i++) {
    for (int j = 1; j <= M; j++) {
      int l = (i+j) % N;
      if(get_random()<P){
        std::cout<<"Entramos a evaluar"<<'\n';
        evalua_desconexion(i,l,(get_random()*N));
      }
    }
  }
}

void imprime_coo_cpu() {
  for (int i = 0; i < (N*M); i++) {
    std::cout << "Rowidx " << h_rowidx[i]<< " Colidx "<< h_colidx[i]<<" Valor " << h_value[i]<<'\n';
  }
}

bool in_edges(int node,int edge){
  int *edges;
  edges = (int*)malloc(M* sizeof(int));

  for(int i=0; i <M; i++){
     edges[i]=h_colidx[(node*M)+i];
  }

  bool flag=false;

  for(int i=0; i<M; i++){
     if(edges[i]==edge){
     	flag = true;
     }
  }

  return flag;

}

double get_random() { return ((double)rand() / (double)RAND_MAX); }

void evalua_desconexion(int node,int edge_original,int edge){
   if((in_edges(node,edge)) || (node ==edge)){
      std::cout<<"Volvemos a evaluar"<< '\n';
      evalua_desconexion(node,edge_original,(get_random()*N));
   }else{
      std::cout<<"Reconectamos"<< '\n';
      std::cout<<"La arista inicial del nodo "<< node<<" es ("<< node<<"," <<edge_original<<"). Reconectamos con el nodo "<< edge<<'\n';
      reconecta(node,edge_original,edge);
   }
}
void reconecta(int node,int edge_original,int edge){

  for(int i=0; i <M; i++){
     if(h_colidx[(node*M)+i]==edge_original){
	h_colidx[(node*M)+i]=edge;
     }
  }

}
