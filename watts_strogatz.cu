/*

Programa: watts_strogatz.cu

*  Compilar
nvcc watts_strogatz.cu -o watts_strogatz

*  Ejecutar.
./watts_strogatz n m p threads
  Donde:
          * n: es la cantidad de vértices de la red
          * m: es la cantidad de vértices vecinos con los que se conecta cada nodo
          * p: es la probabilidad de reconexión
          * threads: es la cantidad de hilos por bloque de la dimensión x

Descripción de problema:

          Watts y Strogatz (1998) presentan un modelo para construir redes de mundo pequeño.

          El algoritmo comienza definiendo una gráfica regular en forma de un anillo con n vértices y m aristas por vértices.

          Posteriormente, se reconectan las aristas de forma aleatoria con una probabilidad p. Cuando p=0, no hay reconexión,
          así que la red queda como anillo, es decir, un red regular. En el caso que $p=1$, la reconexión de las aristas
          genera una red aleatoria. Para valores intermedios 0<p<1, se genera una red de mundo pequeño.

Refencia:
Latora, V., Nicosia, V., and Russo, G. (2017). Complex networks: principles, methods
and applications. Cambridge University Press.
*/

#include <iostream>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <curand.h>
#include <curand_kernel.h>
#include <omp.h>



using namespace std;
/* Definimos los parámetros del modelo

    * N: es la cantidad de vértices de la red
    * M: es la cantidad de vértices vecinos con los que se conecta cada nodo
    * P: es la probabilidad de reconexión
    * HILOS: es la cantidad de hilos por bloque de la dimensión x
*/
int N;
int M;
int P_int;
float P;
int HILOS;

// Definimos las variables para el cálculo de tiempo de las ejecuciones
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
// Definimos un arreglo que contendrá números aleatorios tipo float
float *h_aleatorios_float;
float *d_aleatorios_float;
// Definimos un arreglo que contendrá números aleatorios tipo int
int *h_aleatorios_int;
int *d_aleatorios_int;

// Definimos los arreglos en 1D de la ejecución secuencial
int *secuencial_value;
int *secuencial_colidx;
int *secuencial_rowidx;

// Definimos los métodos
void crea_anillo_cpu();
double get_random();
bool in_edges_cpu(int node, int edge);
void watts_strogatz_cpu();
void reconecta_cpu(int node,int edge_original,int edge);
void gpu_watts_strogatz();


// Función para buscar vecino del device
__device__ bool in_edges_gpu(int node,int edge,int *d_colidx,int M){
  bool flag=false;

  for(int i=0; i<(M*2); i++){
     if(d_colidx[(node*M*2)+i]==edge){
      flag = true;
     }
  }
  return flag;
}

// Kernel gpu_crea_anillo

__global__ void gpu_crea_anillo(int *value,int *rowidx, int *colidx, int N,int M){
  int rownum = blockIdx.x * blockDim.x + threadIdx.x;
  int colnum = blockIdx.y * blockDim.y + threadIdx.y;
  __shared__ int offset ;
  offset = rownum*M*2 + M;
  // Conexión con su l vecino hacia adelante
  value[offset + colnum] = 1 ;
  rowidx[offset + colnum] = rownum;
  colidx[offset + colnum] =  ( rownum + (colnum+1)) % N;
  // Conexión  con su l vecino hacia atrás
  value[offset - (colnum+1)] = 1 ;
  rowidx[offset - (colnum+1)] = rownum;
  int vecino = rownum - (colnum+1);

  if (vecino<0) {
      colidx[offset - (colnum+1)] = N + vecino ;
  }else{
    colidx[offset - (colnum+1)] = vecino ;
  }
}


// Kernel gpu_compute_watts_strogatz que realiza la reconexión de los vértices
__global__ void gpu_compute_watts_strogatz(int *value,int *rowidx,int *colidx, float *aleatorios_float, int *aleatorios_int,int N,int M,float P){

  int rownum = blockIdx.x * blockDim.x + threadIdx.x;
  int colnum = blockIdx.y * blockDim.y + threadIdx.y;

  if (rownum <N && colnum <M) {
    int l = (rownum + (colnum+1)) % N;
    int offset = rownum*M*2 + M;
    if(aleatorios_float[rownum+colnum]<P){
      bool flag = true;
      int aumenta = 0;
      while (flag) {
        int edge_vecino = aleatorios_int[(rownum + colnum +aumenta)%N];
        bool esvecino = in_edges_gpu(rownum,edge_vecino,colidx,M);
        if (esvecino || (rownum ==edge_vecino) || (l ==edge_vecino)) {
            flag = true;
            aumenta +=1;
          }else{
            //printf("Reconectamos la edge (%d,%d) con el nodo %d\n",rownum,l,edge_vecino);
            colidx[offset+colnum]=edge_vecino;
            flag = false;
        }
      }

    }
  }

}

int main(int argc, char const *argv[]) {
  //Recibimos los parámetros
  N =atoi(argv[1]);
  M = atoi(argv[2]);
  P_int = atoi(argv[3]);
  P = (float)P_int/10;
  HILOS = atoi(argv[4]);
  //iter = atoi(argv[5]);

  // Reservamos memoria para los arreglos de la ejecución secuencial
  secuencial_value = (int*)malloc( N*M*2* sizeof(int));
  secuencial_colidx = (int*)malloc( N*M*2* sizeof(int));
  secuencial_rowidx = (int*)malloc( N*M*2* sizeof(int));

  // Reservamos memoria para los arreglos del host
  h_value = (int*)malloc( N*M*2* sizeof(int));
  h_colidx = (int*)malloc( N*M*2* sizeof(int));
  h_rowidx = (int*)malloc( N*M*2* sizeof(int));

  h_aleatorios_float = (float*)malloc( N*M* sizeof(float));
  h_aleatorios_int = (int*)malloc( N*M* sizeof(int));

  // Inicializamos h_aleatorios_float y h_aleatorios_int con OpenMP
  {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < (N*M); i++) {
      h_aleatorios_float[i] = get_random() ;
      h_aleatorios_int[i] = get_random() * N ;
    }
  }

  // Ejecución Serial
  clock_t start = clock();

  // Realizamos la ejecución secuencial
  crea_anillo_cpu();
  watts_strogatz_cpu();
  clock_t end = clock();
  // Imprimimos el tiempo de ejecución serial en segundos
  serialTimer = double (end-start) / double(CLOCKS_PER_SEC);
  cout << "Tiempo serial: " << serialTimer << endl;

  // Realizamos la ejecución paralela
  gpu_watts_strogatz();
  // Imprimimos el tiempo de ejecución paralela en segundos
  cout << "Paralela: " <<(parallelTimer/1000) <<endl;
  // Imprimimos el speedup
  cout << "Speed-up: " << serialTimer / (parallelTimer /1000)<< "X"<<endl;

  //printf("%d,%d,%d,%d,%f,%f,%f,%f\n",iter,N,M,HILOS,P,serialTimer,(parallelTimer /1000), (serialTimer / (parallelTimer /1000)));
  // Liberamos memoria
  free(h_value); free(h_colidx); free(h_rowidx); free(secuencial_value); free(secuencial_colidx); free(secuencial_rowidx);free(h_aleatorios_float); free(h_aleatorios_int);
  cudaFree(d_value); cudaFree(d_colidx); cudaFree(d_rowidx);  cudaFree(d_aleatorios_int); cudaFree(d_aleatorios_float);

  return 0;
}


/*
  Métodos de la ejecución secuencial:
      * void crea_anillo_cpu()
      * void watts_strogatz_cpu()
      * bool in_edges_cpu(int node,int edge)
      * void reconecta_cpu(int node,int edge_original,int edge)
*/

// Genera el anillo
void crea_anillo_cpu() {
  for (int i = 0; i < N; i++) {
    int offset = i*M*2 + M;
    for (int j = 1; j <= M; j++) {
      // Conexión con su l vecino hacia adelante
      secuencial_value[offset + (j-1)] = 1 ;
      secuencial_rowidx[offset + (j-1)] = i;
      secuencial_colidx[offset +(j-1)] = ( i + j) % N;
      // Conexión  con su l vecino hacia atrás
      secuencial_value[offset - j] = 1 ;
      secuencial_rowidx[offset - j] = i;
      int vecino = i - j;

      if (vecino<0) {
          secuencial_colidx[offset - j] = N + vecino ;
      }else{
        secuencial_colidx[offset - j] = vecino ;
      }
    }
  }
}

// Realiza la reconexión de aristas
void watts_strogatz_cpu(){
  for (int i = 0; i < N; i++) {
    for (int j = 1; j <= M; j++) {
      int l = (i+j) % N;
      if(h_aleatorios_float[i+(j-1)]<P){
        bool flag = true;
        int aumenta = 0;
        while (flag) {
          int edge_vecino = h_aleatorios_int[(i+(j-1)+aumenta)% N];

          if((in_edges_cpu(i,edge_vecino)) || (i ==edge_vecino) || (l==edge_vecino)){
              flag = true;
              aumenta +=1;
            }else{
              reconecta_cpu(i,l,edge_vecino);
              flag = false;
          }
        }

      }
    }
  }
}

// Genera números aleatorios de una distribución uniforme
double get_random() { return ((double)rand() / (double)RAND_MAX); }

// Regresa true si ya existe la arista
bool in_edges_cpu(int node,int edge){
  int *edges;
  edges = (int*)malloc(M*2* sizeof(int));

  for(int i=0; i <(M*2); i++){
     edges[i]=secuencial_colidx[(node*M*2)+i];
  }

  bool flag=false;

  for(int i=0; i<(M*2); i++){
     if(edges[i]==edge){
      flag = true;
     }
  }

  return flag;

}

// Realiza la reconexión de la arista
void reconecta_cpu(int node,int edge_original,int edge){

  for(int i=0; i <(M*2); i++){
     if(secuencial_colidx[(node*M*2)+i]==edge_original){
         secuencial_colidx[(node*M*2)+i]=edge;
     }
  }

}


/*
  Ejecución paralela
*/

void gpu_watts_strogatz() {
  // Reservar memoria en device
  cudaMalloc((void **)&d_value, N*M*2* sizeof(int));
  cudaMalloc((void **)&d_colidx, N*M*2* sizeof(int));
  cudaMalloc((void **)&d_rowidx, N*M*2*sizeof(int));
  cudaMalloc((void **)&d_aleatorios_float, N*M*sizeof(float));
  cudaMalloc((void **)&d_aleatorios_int, N*M*sizeof(int));


  // Transferir datos de host h_a device
  cudaMemcpy(d_aleatorios_float, h_aleatorios_float, N*M*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_aleatorios_int, h_aleatorios_int, N*M*sizeof(int), cudaMemcpyHostToDevice);

  // Definimos los bloques de la dimensión x
  int blocks = ceil(N / HILOS) + 1;
  int threads = HILOS;

  // Definimos los timers para el tiempo de ejecución
  cudaEvent_t start, stop;

  // events to take time
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start,0);

  // Definimos las estructuras que contienen los bloques y los hilos por bloque
  dim3 dimGrid(blocks,1, 1);
  dim3 dimBlock(threads, M);

  // Llamamos al kernel para crear el anillo en el device
  gpu_crea_anillo<<<dimGrid,dimBlock>>>(d_value,d_rowidx,d_colidx,N,M);
  // Llamamos al kernel para realizar la desconexión de las aristas
  gpu_compute_watts_strogatz<<<dimGrid, dimBlock>>>(d_value,d_rowidx,d_colidx,d_aleatorios_float,d_aleatorios_int,N,M,P);

  // Transferimo los arreglos que representan la matriz de adyacencia en COO del device al host
  cudaMemcpy(h_value, d_value, N*M*2* sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_rowidx, d_rowidx, N*M*2* sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_colidx, d_colidx, N*M*2* sizeof(int), cudaMemcpyDeviceToHost);

  cudaEventRecord(stop,0);

  cudaEventSynchronize(stop);

  cudaEventElapsedTime(&parallelTimer, start, stop);

}
