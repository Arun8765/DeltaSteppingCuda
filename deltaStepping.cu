#include<iostream>
#include<fstream>
#include<algorithm>
#include<vector>
#include<queue>
#include<stack>
#include <sys/time.h>
#include <cuda_runtime.h>
// #define NO_OF_BUCKETS 2500
#define NO_OF_BUCKETS 6000
#define DELTA_SEQUENTIAL 25 // Max Weight is 200 so if delta is 20 then while processing ith bucket at most the i+10th bucket will be touched.
#define INF 2e9
using namespace std; 

int M, N, L;
int inf = INF;
int delta = DELTA_SEQUENTIAL;
const string seqPath = "output/output_seq_";
const string cudaPath = "output/output_";
int maxWeightGlobal = -1;



// vector<vector<int>> matrix;

void readMtxFile(const char *filename , vector<vector<vector<int>>> & matrix){
    ifstream fin(filename);

    while (fin.peek() == '%') fin.ignore(2048, '\n');

    fin >> M >> N >> L;
    matrix = vector<vector<vector<int>>>(M+1, vector<vector<int>>());	     // Creates the array of M*N size

    int maxWeight = -1;
    // Read the data
    for (int l = 0; l < L; l++)
    {
        int m, n;
        int data;
        fin >> m >> n >> data;
        maxWeight = max(maxWeight, data);
        matrix[m-1].push_back({n-1,data});

    }
    fin.close();
    // cout<<"maxWeight: "<<maxWeight<<" L: "<<L<<endl;
    maxWeightGlobal = maxWeight;
    std::cout<<"No of nodes:"<<M<<", No of edges:"<<L<<", maxWeight: "<<maxWeight<<endl;
}

void writePathToFile(const string filename, vector<int> & parent, int source, vector<int> & dist){
    std::cout<<"Writing to file: "<<filename<<endl;
    ofstream outFile(filename, ios::out | ios::binary);
    if(!outFile.is_open()){
        std::cout<<"Error in opening file"<<endl;
        return;
    }
    for(int dest = 0; dest<M;dest++){
        // cout<<dest<<" "<<dist[dest]<<" "<<parent[dest]<<endl;
        if(dist[dest] == inf|| parent[dest] == -1){
            int dummyDist = -1;
            int pathSize = 0;
            outFile.write((char *) &dummyDist, sizeof(int));
            outFile.write((char *) &pathSize, sizeof(int));
            continue;
        }
        stack<int> pathStack;
        int curr = dest;
        pathStack.push(curr);
        while(curr != source){
            // outFile<<curr<<endl;
            pathStack.push(parent[curr]);
            curr = parent[curr];
        }
        vector<int> path(pathStack.size());
        for(int i = 0;i<path.size();i++){
            path[i] = pathStack.top()+1;
            pathStack.pop();
        }
        int pathSize = path.size();
        outFile.write((char *) &dist[dest], sizeof(int));
        outFile.write((char *) &pathSize, sizeof(int));
        outFile.write((char *) &path[0], pathSize*sizeof(int));

    }
    // outFile<<source<<endl;
    outFile.close();
}

// Cuda Kernel for Delta stepping


struct DistParent{
    int dist = INF;
    int parent = -1;
};

__global__ void deltaSteppingKernel(
    // vector<vector<vector<int>>> adjList,
    int M,
    int * rowPtr,
    int * colInd,
    int * weights,
    DistParent * distParent,
    int delta, 
    int currBucket,
    int * changeDone
    ){

    int blockId = blockIdx.x;
    int blockSize = blockDim.x;
    int threadId = threadIdx.x;
    int u = blockId * blockSize + threadId; // Current Node u
    if(u >= M || u < 0) return;

    int d = distParent[u].dist;
    if(d == INF) return;
    
    int currNodeBucket = d / delta;
    if(currNodeBucket != currBucket) return;
    
    for(int i=0;i<rowPtr[u+1] - rowPtr[u];i++){
        int v = colInd[rowPtr[u] + i];
        int w = weights[rowPtr[u] + i];
        if(v == u){
            continue;
        }
        if(d + w < distParent[v].dist){
            int temp = d + w;
            DistParent tempDistParent;
            tempDistParent.dist = temp;
            tempDistParent.parent = u;
            
            atomicExch(reinterpret_cast<unsigned long long*>(&distParent[v]), *reinterpret_cast<unsigned long long*>(&tempDistParent));
            // distParent[v] = tempDistParent;

            *changeDone = 1; // Change done should be in out.
        }
    }
}

void adjListToCSR(vector<vector<vector<int>>> & adjList, vector<int> & rowPtr, vector<int> & colInd, vector<int> & weights){
    int edgeCount = 0;
    for(int i=0;i<adjList.size();i++){
        rowPtr[i] = edgeCount;
        for(int j=0;j<adjList[i].size();j++){
            colInd.push_back(adjList[i][j][0]);
            weights.push_back(adjList[i][j][1]);
            edgeCount++;
        }
    }
    rowPtr[adjList.size()] = edgeCount;
}

void deltaSteppingCuda(vector<vector<vector<int>>> & matrix, const int source, vector<int> & distCuda){

    vector<DistParent> distParent(M);
    
    vector<int> dist(M, inf);
    vector<int> cudaParent(M, -1);


    dist[source] = 0;
    cudaParent[source] = source;
    distParent[source].dist = 0;
    distParent[source].parent = source;
    
    // Convert the adjList to CSR format
    vector<int> rowPtr(M+1);
    vector<int> colInd;
    vector<int> weights;
    adjListToCSR(matrix, rowPtr, colInd, weights);



    int stopperLength = ( maxWeightGlobal/delta ) * 2 + 1;

    struct timeval start_time, end_time;
    gettimeofday(&start_time, NULL);

    int threadBlockSize = 1024;
    // int threadBlockSize = 128;
    int change[1];
    

    int * d_rowPtr;
    int * d_colInd;
    int * d_weights;
    int * d_changeDone;
    DistParent * d_distParent;


    cudaMalloc( (void **) &d_distParent, M * sizeof(DistParent));
    cudaMalloc( (void **) &d_rowPtr, (M+1) * sizeof(int) );
    cudaMalloc( (void **) &d_colInd, colInd.size() * sizeof(int) );
    cudaMalloc( (void **) &d_weights, weights.size() * sizeof(int) );
    cudaMalloc( (void **) &d_changeDone, 1 * sizeof(int));


    cudaMemcpy( d_distParent, &distParent[0], M * sizeof(DistParent), cudaMemcpyHostToDevice );
    cudaMemcpy( d_rowPtr, &rowPtr[0], (M+1) * sizeof(int), cudaMemcpyHostToDevice );
    cudaMemcpy( d_colInd, &colInd[0], colInd.size() * sizeof(int), cudaMemcpyHostToDevice );
    cudaMemcpy( d_weights, &weights[0], weights.size() * sizeof(int), cudaMemcpyHostToDevice );

    cudaDeviceSynchronize();

    // gettimeofday(&start_time, NULL);

    int stopCounter = 0;
    for(int currBucket = 0; true; currBucket++){
        
        change[0] = 1;
        int cntr = 0;
        while(change[0] == 1){

            change[0] = 0;
            cudaMemcpy( d_changeDone, &change[0], 1 * sizeof(int), cudaMemcpyHostToDevice );

            deltaSteppingKernel<<<(M+threadBlockSize-1)/threadBlockSize, threadBlockSize>>>(M, d_rowPtr, d_colInd, d_weights, d_distParent, delta, currBucket, d_changeDone);

            cudaMemcpy(&change[0], d_changeDone, 1 * sizeof(int), cudaMemcpyDeviceToHost);
            cntr++;
        }

        if(cntr <= 1){
            stopCounter++;
        }
        else{
            stopCounter = 0;
        }

        if(stopCounter >= stopperLength){
            // cout<<"Breaking at "<<currBucket<<endl;
            break;
        }

    }

    // Measure the time
    // gettimeofday(&end_time, NULL);

    // cuda
    cudaMemcpy( &distParent[0], d_distParent , M * sizeof(DistParent), cudaMemcpyDeviceToHost);


    gettimeofday(&end_time, NULL);

    long long elapsed_time_microseconds = (end_time.tv_sec - start_time.tv_sec) * 1000000LL +
                                          (end_time.tv_usec - start_time.tv_usec);
    
    // cout<<"Max Degree: "<<maxDegree<<endl;
    std::cout <<"For Delta="<<delta<< ", Cuda Delta Stepping Time: " << elapsed_time_microseconds << " microseconds" << endl;


    for(int i=0;i<M;i++){
        dist[i] = distParent[i].dist;
        cudaParent[i] = distParent[i].parent;
    }
    
    distCuda = dist;
    writePathToFile(cudaPath+to_string(source+1), cudaParent, source, dist);

    cudaFree(d_rowPtr);
    cudaFree(d_colInd);
    cudaFree(d_weights);
    cudaFree(d_changeDone);
    cudaFree(d_distParent);

    cudaDeviceSynchronize();

    // cout<<"Cuda Done"<<endl;


}


void deltaSteppingSequential(vector<vector<vector<int>>> & matrix, const int source, vector<int> & distSeq){

    vector<int> dist(M, inf);
    vector<int> seqParent(M, -1);
    dist[source] = 0;
    seqParent[source] = source;


    vector<priority_queue<pair<int,int>, vector<pair<int,int>>, greater<pair<int,int>>>> buckets(NO_OF_BUCKETS);

    buckets[0].push({0, source});

    struct timeval start_time, end_time;
    gettimeofday(&start_time, NULL);


    for(int currBucket = 0; currBucket < NO_OF_BUCKETS; currBucket++){
        while(!buckets[currBucket].empty()){
            pair<int,int> top = buckets[currBucket].top();
            buckets[currBucket].pop();
            int u = top.second;
            int d = top.first;
            if(d > dist[u]) continue;

            for(vector<int> neighbourNode : matrix[u]){
                int v = neighbourNode[0];
                int w = neighbourNode[1];
                if(dist[u] + w < dist[v]){
                    dist[v] = dist[u] + w;
                    int newBucket = dist[v] / delta;
                    if(newBucket >= NO_OF_BUCKETS) newBucket = NO_OF_BUCKETS - 1;
                    buckets[newBucket].push({dist[v], v});
                    seqParent[v] = u;
                }
            }
        }

    }

    // Measure the time

    gettimeofday(&end_time, NULL);

    long long elapsed_time_microseconds = (end_time.tv_sec - start_time.tv_sec) * 1000000LL +
                                          (end_time.tv_usec - start_time.tv_usec);
    
    cout <<"For Delta="<<delta<< ", Sequential Delta Stepping Time: " << elapsed_time_microseconds << " microseconds" << endl;

    distSeq = dist;
    
    writePathToFile(seqPath+to_string(source+1), seqParent, source, dist);


}



void solve(int argc, char** argv){
    if (argc != 3){
        cout << "Usage: " << argv[0] << " <matrix_filename> <source vertex>" << endl;
        return;
    }

    const char *filename = argv[1];
    int source = atoi(argv[2]);
    
    cout<<"\nSource Node: "<<source<<endl;

    vector<vector<vector<int>>> matrix;
    readMtxFile(filename, matrix);

    source = source - 1;

    vector<int> distSeq(M, inf);
    vector<int> distCuda(M, inf);

    deltaSteppingSequential(matrix, source, distSeq);

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
        if (deviceCount == 0) {
        std::cerr << "Error: No CUDA-enabled GPU device found. Exiting..." << std::endl;
        return;
    }

    std::cout << "CUDA-enabled GPU device(s) found: " << deviceCount << std::endl;

    deltaSteppingCuda(matrix, source, distCuda);
    // cout<<"Cuda Returned\n";

    for(int i=0;i<M;i++){
        if(distSeq[i] != distCuda[i]){
            cout<<"Mismatch at "<<i<<": "<<distSeq[i]<<" "<<distCuda[i]<<endl;
            break;
        }
    }
    cout<<"Comparison Done\n";


}

int main(int argc, char** argv){
    solve(argc, argv);
    return 0;
}