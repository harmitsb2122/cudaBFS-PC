#include "createInput.h"

bool edgeStreamToCSR(std::vector<int>& neighbour, std::vector<int>& offset) {

    std::ifstream inputFile("output.txt");

    if(!inputFile) {
        std::cerr <<"Unable to open file for reading\n";
        return false;
    }

    int numVertices, numEdges;
    inputFile >> numVertices >> numEdges;

    int u, v;

    std::vector<std::vector<int>> adjlist(numVertices);

    for(int i = 0; i < numEdges; ++i) {
        inputFile >> u >> v;
        adjlist[u].push_back(v);
        adjlist[v].push_back(u);
    }

    offset.resize(numVertices + 1);

    offset[0] = 0;

    for(int i = 0; i < adjlist.size(); ++i) {
        offset[i+1] = offset[i] + adjlist[i].size();
        for(int j = 0; j < adjlist[i].size(); ++j) {
            neighbour.push_back(adjlist[i][j]);
        }
    }

    return true;
}

bool createInputGraph(int width, int depth, int& numVertices, int& numEdges, std::vector<int>& neighbour, std::vector<int>& offset) {

    std::ofstream outFile("output.txt");
    if (!outFile) {
        std::cerr << "Error: Unable to create file.\n";
        return false;
    }

    numEdges = width*depth;
    numVertices = numEdges + 1;

    outFile << numVertices <<" " << numEdges << std::endl;

    for(int i = 1; i <= width; ++i)
        outFile << "0" <<" " << i << std::endl;
    depth--;

    int start_index;
    int itr_num = 0;

    while(depth--) {
        start_index = width * itr_num;

        for(int i = 1; i <= width; ++i) {
            outFile << start_index + i << " " << start_index + i + width << std::endl;
        }
        itr_num++;
    }

    outFile.close();

    edgeStreamToCSR(neighbour, offset);

    return true;
}
