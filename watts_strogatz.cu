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

  // Definimos los arreglos en 1D de la ejecución secuencial
  int *secuencial_value;
  int *secuencial_colidx;
  int *secuencial_rowidx;

  // Definimos los métodos
  void crea_anillo_cpu();
  void imprime_coo_cpu() ;
  void imprime_coo_gpu() ;
  double get_random();
  bool in_edges_cpu(int node, int edge);
  void watts_strogatz_cpu();
  void evalua_desconexion_cpu(int node,int edge_original,int edge);
  void reconecta_cpu(int node,int edge_original,int edge);
  void evalua_desconexion_gpu(int node,int edge_original,int edge);
  void reconecta_gpu(int node,int edge_original,int edge);
  bool in_edges_gpu(int node, int edge);
  void gpu_watts_strogatz();
  // Kernel gpu_crea_anillo
  __global__ void gpu_crea_anillo(int *value,int *rowidx, int *colidx){
    int rownum = blockIdx.x;
    for (int j = 1; j <= M; j++) {
      value[(rownum*M) + (j-1)] = 1 ;
      rowidx[(rownum*M) + (j-1)] = rownum;
      colidx[(rownum*M) +(j-1)] = ( rownum + j) % N;
    }
  }
  int main(int argc, char const *argv[]) {

    // Reservamos memoria para los arreglos de la ejecución secuencial
    secuencial_value = (int*)malloc( N*M* sizeof(int));
    secuencial_colidx = (int*)malloc( N*M* sizeof(int));
    secuencial_rowidx = (int*)malloc( N*M* sizeof(int));

    // Reservamos memoria para los arreglos del host
    h_value = (int*)malloc( N*M* sizeof(int));
    h_colidx = (int*)malloc( N*M* sizeof(int));
    h_rowidx = (int*)malloc( N*M* sizeof(int));

    // Ejecución Serial
    clock_t start = clock();
    crea_anillo_cpu();
    //watts_strogatz_cpu();
    clock_t end = clock();
    serialTimer = double (end-start) / double(CLOCKS_PER_SEC);
    cout << "Tiempo serial: " << serialTimer << endl;
    //imprime_coo_cpu();
    // Ejecución Paralela
    gpu_watts_strogatz();
    //imprime_coo_gpu();
    cout << "Serial: " << serialTimer << " Parallel: " << parallelTimer / 1000 <<endl;
    cout << "Speed-up: " << serialTimer / (parallelTimer /1000)<< "X"<<endl;

    return 0;
  }

  void crea_anillo_cpu() {
    for (int i = 0; i < N; i++) {
      for (int j = 1; j <= M; j++) {
        secuencial_value[(i*M) + (j-1)] = 1 ;
        secuencial_rowidx[(i*M) + (j-1)] = i;
        secuencial_colidx[(i*M) +(j-1)] = ( i + j) % N;
      }
    }
  }

  void watts_strogatz_cpu(){
    for (int i = 0; i < N; i++) {
      for (int j = 1; j <= M; j++) {
        int l = (i+j) % N;
        if(get_random()<P){
          //std::cout<<"Entramos a evaluar"<<'\n';
          evalua_desconexion_cpu(i,l,(get_random()*N));
        }
      }
    }
  }

  void imprime_coo_cpu() {
    for (int i = 0; i < (N*M); i++) {
      std::cout << "Rowidx " << secuencial_rowidx[i]<< " Colidx "<< secuencial_colidx[i]<<" Valor " << secuencial_value[i]<<'\n';
    }
  }

  double get_random() { return ((double)rand() / (double)RAND_MAX); }

  // Funciones CPU
  bool in_edges_cpu(int node,int edge){
    int *edges;
    edges = (int*)malloc(M* sizeof(int));

    for(int i=0; i <M; i++){
       edges[i]=secuencial_colidx[(node*M)+i];
    }

    bool flag=false;

    for(int i=0; i<M; i++){
       if(edges[i]==edge){
       	flag = true;
       }
    }

    return flag;

  }

  void evalua_desconexion_cpu(int node,int edge_original,int edge){
     if((in_edges_cpu(node,edge)) || (node ==edge)){
        //std::cout<<"Volvemos a evaluar"<< '\n';
        evalua_desconexion_cpu(node,edge_original,(get_random()*N));
     }else{
        //std::cout<<"Reconectamos"<< '\n';
        //std::cout<<"La arista inicial del nodo "<< node<<" es ("<< node<<"," <<edge_original<<"). Reconectamos con el nodo "<< edge<<'\n';
        reconecta_cpu(node,edge_original,edge);
     }
  }
  void reconecta_cpu(int node,int edge_original,int edge){

    for(int i=0; i <M; i++){
       if(secuencial_colidx[(node*M)+i]==edge_original){
  	       secuencial_colidx[(node*M)+i]=edge;
       }
    }

  }
  // Funciones GPU
  void imprime_coo_gpu() {
    for (int i = 0; i < (N*M); i++) {
      std::cout << "Rowidx " << h_rowidx[i]<< " Colidx "<< h_colidx[i]<<" Valor " << h_value[i]<<'\n';
    }
  }

  bool in_edges_gpu(int node,int edge){
    int *edges;
    edges = (int*)malloc(M* sizeof(int));

    for(int i=0; i <M; i++){
       edges[i]=d_colidx[(node*M)+i];
    }

    bool flag=false;

    for(int i=0; i<M; i++){
       if(edges[i]==edge){
       	flag = true;
       }
    }

    return flag;

  }

  void evalua_desconexion_gpu(int node,int edge_original,int edge){
     if((in_edges_gpu(node,edge)) || (node ==edge)){
        //std::cout<<"Volvemos a evaluar"<< '\n';
        evalua_desconexion_gpu(node,edge_original,(get_random()*N));
     }else{
        //std::cout<<"Reconectamos"<< '\n';
        //std::cout<<"La arista inicial del nodo "<< node<<" es ("<< node<<"," <<edge_original<<"). Reconectamos con el nodo "<< edge<<'\n';
        reconecta_gpu(node,edge_original,edge);
     }
  }
  void reconecta_gpu(int node,int edge_original,int edge){

    for(int i=0; i <M; i++){
       if(d_colidx[(node*M)+i]==edge_original){
  	       d_colidx[(node*M)+i]=edge;
       }
    }
  }

  /*
    Ejecución paralela
  */

  void gpu_watts_strogatz() {
    // Reservar memoria en device
    cudaMalloc((void **)&d_value, N*M * sizeof(int));
    cudaMalloc((void **)&d_colidx, N*M * sizeof(int));
    cudaMalloc((void **)&d_rowidx, N*M*sizeof(double));

    dim3 dimGrid((N - 1) / BLOCK_SIZE + 1, (N - 1) / BLOCK_SIZE + 1, 1);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    // define timers
    cudaEvent_t start, stop;

    // events to take time
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start,0);

    int blocks = ceil(N / 1024) + 1;
    int threads = 1024;

    gpu_crea_anillo<<<blocks, threads>>>(d_value,d_rowidx,d_colidx);

    cudaMemcpy(h_value, d_value, N*M * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_rowidx, d_rowidx, N*M * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_colidx, d_colidx, N*M * sizeof(int), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop,0);

    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&parallelTimer, start, stop);

    cout<< "Elapsed parallel timer: " << parallelTimer << " ms, " << parallelTimer / 1000 << " secs" <<endl;

    // Copy data from device to host

  }
