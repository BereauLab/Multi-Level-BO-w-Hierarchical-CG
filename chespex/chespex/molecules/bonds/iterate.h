#ifndef ITERATE_H
#define ITERATE_H

#include <math.h>
#include <omp.h>
#include <algorithm>
#include <utility>
#include <vector>
#include <map>
#include <queue>
#include <unordered_map>
#include <iostream>

#define MAX_BONDS 4

typedef std::vector<int> repr;
typedef uint64_t triu;

class Graph
{
public:
    Graph(uint nBeads, uint *beads, triu adjacencyTriu, triu *degreeFilter, std::vector<std::vector<int>> *permutations);
    uint nBeads;
    repr representation;
    bool connected;
    std::vector<std::vector<bool>> adjacencyMatrix;
    bool operator==(const Graph &other) const;
    bool isConnected() const;
    std::vector<bool> getTriuVector() const;
    std::vector<std::pair<uint, uint>> getEdges() const;

private:
    uint *beads;
    triu adjacencyTriu;
    triu *degreeFilter;
    std::vector<std::vector<int>> *permutations;
    repr getRepresentation() const;
    std::vector<std::vector<bool>> getAdjacencyMatrix() const;
};

#pragma omp declare reduction(merge : std::vector<Graph> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
#pragma omp declare reduction(merge : std::vector<triu> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))

void setDegreeFilter(triu *result, uint n_beads);
int BitCount(unsigned int u);
bool checkDegrees(triu v, const triu *degs, uint n_beads);
triu nextPermutation(triu v);
void generatePermutations(std::vector<std::vector<int>> &permutations, const uint *numbers, const uint nBeads,
                          std::vector<int> &currentPermutation, std::unordered_map<int, std::vector<int>> &indexMap, uint currentIndex);
std::vector<std::vector<std::pair<uint, uint>>> getBonds(uint nBeads, uint *beads); // main function

#endif
