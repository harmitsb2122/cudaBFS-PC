#ifndef CREATE_INPUT_H
#define CREATE_INPUT_H

#include <iostream>
#include <fstream>
#include <vector>

bool edgeStreamToCSR(std::vector<int>& neighbour, std::vector<int>& offset);
bool createInputGraph(int width, int depth, int& numVertices, int& numEdges, std::vector<int>& neighbour, std::vector<int>& offset);

#endif //CREATE_INPUT_H