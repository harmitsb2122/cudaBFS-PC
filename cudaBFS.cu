// nvcc -std=c++17 createInput.cpp cudaBFS.cu -o cudaBFS
// ./cudaBFS <width> <depth> > debug.txt

// nvcc -std=c++17 -DDEBUG createInput.cpp cudaBFS.cu -o cudaBFS
// ./cudaBFS <width> <depth> > debug.txt

#include <iostream>
#include <queue>
#include "createInput.h"
#include <fstream>
#include <iomanip>
#include <chrono>
#include <climits>

#define INF INT_MAX
#define SOURCE 0

/**
 * @brief function to compute distance of each node from the root through serial BFS algorithm
 *
 * @param neighbour neighbour array in csr format
 * @param offset offset array in csr format
 * @param n number of vertices
 * @param m number of edges
 * @param dist distance array - stores the distance of each node from the root node i.e 0
 */
void serailBFS(std::vector<int> &neighbour, std::vector<int> &offset, int n, int m, std::vector<int> &dist)
{
    std::queue<int> q;

    q.push(SOURCE);
    dist[SOURCE] = 0;

    while (!q.empty())
    {
        int par = q.front();
        q.pop();

        for (int node = offset[par]; node < offset[par + 1]; node++)
        {
            int child = neighbour[node];
            if (dist[child] > dist[par] + 1)
            {
                dist[child] = dist[par] + 1;
                q.push(child);
            }
        }
    }
}

/**
 * @brief BFS cuda kernel function
 *
 * @param n number of vertices in tree
 * @param d_neighbour neighbour array in csr format
 * @param d_offset offset array in csr format
 * @param d_dist dist array
 * @param d_flag flag to check termination condition
 * @param d_level_no level number or iteration number of BFS algorithm
 * @return __global__
 */
__global__ void BFS_KERNEL(int n, int *d_neighbour, int *d_offset, int *d_dist, int *d_flag, int *d_level_no)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
    {
        int par = tid;
        if (d_dist[par] != *d_level_no)
        {
            return;
        }

        for (int node = d_offset[par]; node < d_offset[par + 1]; node++)
        {
            int child = d_neighbour[node];
            if (d_dist[child] == INF)
            {
                d_dist[child] = *d_level_no + 1;
                *d_flag = 1;
            }
        }
    }
}

/**
 * @brief function to compute distance of each node from the root through cuda BFS algorithm
 *
 * @param neighbour neighbour array in csr format
 * @param offset offset array in csr format
 * @param n number of vertices
 * @param m number of edges
 * @param dist distance array - stores the distance of each node from the root node i.e 0
 * @param time variable to compute total time taken by the algorithm
 */
void cudaBFS(std::vector<int> &neighbour, std::vector<int> &offset, int n, int m, std::vector<int> &dist, float &time)
{
    int flag = 1, level_no = 0;
    dist[SOURCE] = 0;

    // CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // CUDA variable initialization
    int *d_neighbour, *d_offset, *d_dist, *d_flag, *d_level_no;
    cudaMalloc((void **)&d_neighbour, sizeof(int) * (neighbour.size()));
    cudaMalloc((void **)&d_offset, sizeof(int) * (offset.size()));
    cudaMalloc((void **)&d_dist, sizeof(int) * (dist.size()));
    cudaMalloc((void **)&d_flag, sizeof(int));
    cudaMalloc((void **)&d_level_no, sizeof(int));

    // CUDA device properties
    cudaDeviceProp prop;
    int device;
    cudaGetDevice(&device);                 // get current device
    cudaGetDeviceProperties(&prop, device); // get the properties of the device

    int maxThreadsPerBlock = prop.maxThreadsPerBlock; // max threads that can be spawned per block

    // calculate the optimal number of threads and blocks
    int threadsPerBlock = (n < maxThreadsPerBlock) ? n : maxThreadsPerBlock;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Copying variables to device
    cudaMemcpy(d_neighbour, neighbour.data(), sizeof(int) * (neighbour.size()), cudaMemcpyHostToDevice);
    cudaMemcpy(d_offset, offset.data(), sizeof(int) * (offset.size()), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dist, dist.data(), sizeof(int) * (dist.size()), cudaMemcpyHostToDevice);
    cudaMemcpy(d_level_no, &level_no, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_flag, &flag, sizeof(int), cudaMemcpyHostToDevice);

    float kernelTime = 0; // kernelTime - time taken by a BFS kernel function call

    while (flag)
    {
        flag = 0;
        cudaMemcpy(d_level_no, &level_no, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_flag, &flag, sizeof(int), cudaMemcpyHostToDevice);

        cudaEventRecord(start, 0); // start the timer to avoid recording the allocation time
        BFS_KERNEL<<<blocksPerGrid, threadsPerBlock>>>(n, d_neighbour, d_offset, d_dist, d_flag, d_level_no);
        cudaDeviceSynchronize();

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&kernelTime, start, stop);

        cudaMemcpy(&flag, d_flag, sizeof(int), cudaMemcpyDeviceToHost);
        level_no++;

        time += kernelTime;
    }

    // Destroying the time events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(dist.data(), d_dist, sizeof(int) * (dist.size()), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // Destroying local variables
    cudaFree(d_neighbour);
    cudaFree(d_offset);
    cudaFree(d_dist);
    cudaFree(d_flag);
    cudaFree(d_level_no);
}

/**
 * @brief function to check if serial and cuda BFS results are matching
 *
 * @param dist1 distance array from serial BFS
 * @param dist2 distance array from cuda BFS
 * @return true
 * @return false
 */
bool isMatching(std::vector<int> &dist1, std::vector<int> &dist2)
{
    bool status = true;
    for (int i = 0; i < dist1.size(); i++)
    {
        status &= (dist1[i] == dist2[i]);
    }
    return status;
}

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <width> <depth>\n";
        return EXIT_FAILURE;
    }
    int width = std::stoi(argv[1]);
    int depth = std::stoi(argv[2]);

    std::vector<int> neighbour;
    std::vector<int> offset;
    int numVertices, numEdges;

    /**
     * @function createInputGraph
     *
     * Constructs a graph with specified dimensions and outputs its details and CSR representation.
     *
     * @param width (int): The specified width of the graph.
     * @param depth (int): The specified depth of the graph.
     *
     * @param numVertices (int&): Output parameter to hold the total number of vertices in the graph.
     * @param numEdges (int&): Output parameter to hold the total number of edges in the graph.
     * @param neighbour (std::vector<int>&): Output vector to hold the adjacency details of the graph in CSR format.
     * @param offset (std::vector<int>&): Output vector to hold the vertex offsets for the CSR format adjacency representation.
     *
     * @returns:
     *   - No direct return value. The function populates the output parameters with the graph's details.
     */

    createInputGraph(width, depth, numVertices, numEdges, neighbour, offset);

#ifdef DEBUG
    std::cout << "\nPrinting csr graph : \n";
    for (int i = 0; i < numVertices; ++i)
    {
        std::cout << i << ": ";
        for (int j = offset[i]; j < offset[i + 1]; ++j)
        {
            std::cout << neighbour[j] << " ";
        }
        std::cout << std::endl;
    }
#endif

    std::vector<int> dist1(numVertices, INF), dist2(numVertices, INF);

    float time1 = 0, time2 = 0;
    auto start = std::chrono::high_resolution_clock::now();

    serailBFS(neighbour, offset, numVertices, numEdges, dist1);

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

    time1 = (float)duration.count();

#ifdef DEBUG
    std::cout << "\nPrinting serial bfs dist : \n";
    for (int i = 0; i < numVertices; ++i)
    {
        if (dist1[i] == INF)
        {
            std::cout << "INF ";
        }
        else
        {
            std ::cout << dist1[i] << ' ';
        }
    }
    std::cout << std::endl;
#endif

    cudaBFS(neighbour, offset, numVertices, numEdges, dist2, time2);
#ifdef DEBUG
    std::cout << "\nPrinting cuda bfs dist : \n";
    for (int i = 0; i < numVertices; ++i)
    {
        if (dist2[i] == INF)
        {
            std::cout << "INF ";
        }
        else
        {
            std ::cout << dist2[i] << ' ';
        }
    }
    std::cout << std::endl;
#endif

    if (isMatching(dist1, dist2))
    {
        std::cout << "SUCCESS : Results of both serial and cuda BFS are same.\n\n";
    }
    else
    {
        std::cout << "INCORRECT : Results of both serial and cuda BFS are different.\n\n";
    }

    std::cout << "Time taken by serial BFS (time1): " << std::fixed << std::setprecision(5) << time1 << " milli seconds" << std::endl;
    std::cout << "Time taken by cudaBFS (time2): " << std::fixed << std::setprecision(5) << time2 << " milli seconds" << std::endl;
    std::cout << "Speedup (time1/time2) : " << std::fixed << std::setprecision(5) << (time1 / time2) << std::endl;
    return EXIT_SUCCESS;
}
