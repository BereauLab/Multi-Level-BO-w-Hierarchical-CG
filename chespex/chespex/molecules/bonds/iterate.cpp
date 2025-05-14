#include "iterate.h"

/*
 * This function fills an array with binary representations of an upper triangular adjacency matrix
 * filter. I.e. the first element is set to one at all positions which correspond to the first row
 * of the adjacency matrix, the second element is set to one at all positions which correspond to the
 * second row of the adjacency matrix, and so on.
 */
void setDegreeFilter(triu *result, uint n_beads)
{
    uint max_size = n_beads * (n_beads - 1) / 2;
    for (uint k = 0; k < n_beads; k++)
    {
        std::vector<std::vector<int>> matrix(n_beads, std::vector<int>(n_beads, 0));
        for (uint i = 0; i < n_beads; i++)
        {
            matrix[k][i] = 1;
            matrix[i][k] = 1;
        }
        result[k] = 0;
        uint count = 1;
        for (uint i = 0; i < n_beads; i++)
        {
            for (uint j = i + 1; j < n_beads; j++)
            {
                if (matrix[i][j] == 1)
                {
                    result[k] += 1 << (max_size - count);
                }
                count++;
            }
        }
    }
}

/*
 * Function to count the number of set bits in a binary number
 * https://web.archive.org/web/20151229003112/http://blogs.msdn.com/b/jeuge/archive/2005/06/08/hakmem-bit-count.aspx
 */
int BitCount(unsigned int u)
{
    unsigned int uCount;
    uCount = u - ((u >> 1) & 033333333333) - ((u >> 2) & 011111111111);
    return ((uCount + (uCount >> 3)) & 030707070707) % 63;
}

/*
 * Function to check if the adjacency matrix represented by a binary number is valid, i.e.,
 * all beads have at least one bond and at most MAX_BONDS bonds
 */
bool checkDegrees(triu v, const triu *degs, uint n_beads)
{
    for (uint i = 0; i < n_beads; i++)
    {
        int count = BitCount(v & degs[i]);
        if (count > MAX_BONDS || count == 0)
        {
            return false;
        }
    }
    return true;
}

/*
 * Function to find the next permutation of the binary representation of a number
 * https://graphics.stanford.edu/~seander/bithacks.html#NextBitPermutation
 */
triu nextPermutation(triu v)
{
    triu t = v | (v - 1);
    return (t + 1) | (((~t & -~t) - 1) >> (__builtin_ctz(v) + 1));
}

/*
* This function generates permutations of a binary representation of an upper triangular adjacency matrix.
*/
void generatePermutations(std::vector<std::vector<int>> &permutations, const uint *numbers, const uint nBeads,
                          std::vector<int> &currentPermutation, std::unordered_map<int, std::vector<int>> &indexMap, uint currentIndex)
{
    if (currentIndex == nBeads)
    {
        // Permutation complete
        permutations.push_back(std::vector<int>(currentPermutation));
        return;
    }

    int currentValue = numbers[currentIndex];
    auto &indices = indexMap[currentValue];

    for (uint i = 0; i < indices.size(); ++i)
    {
        int index = indices[i];
        indices.erase(indices.begin() + i); // Remove temporarily

        currentPermutation[currentIndex] = index;
        generatePermutations(permutations, numbers, nBeads, currentPermutation, indexMap, currentIndex + 1);

        indices.insert(indices.begin() + i, index); // Restore (backtrack)
    }
}

/*
* This is the main function of the module. It generates all valid bond configurations for a given sequence of ordered beads.
* The beads should be in strictly increasing order and start with 0, e.g., [ 0, 0, 1, 1, 2, 3, 4] or [ 0, 0, 0, 1, 1]
*/
std::vector<std::vector<std::pair<uint, uint>>> getBonds(uint nBeads, uint *beads)
{
    // Prepare degree filter
    triu degreeFilter[nBeads];
    setDegreeFilter(degreeFilter, nBeads);

    // Calculate valid adjacency matrices with min and max number of bonds as fast as possible
    std::vector<triu> trius;
    uint totalMaxBonds = std::min(nBeads * MAX_BONDS / 2, nBeads * (nBeads - 1) / 2);
    uint triuSize = nBeads * (nBeads - 1) / 2;
#pragma omp parallel for reduction(merge : trius) // Parallelizable across different numbers of bonds
    for (uint i = nBeads - 1; i <= totalMaxBonds; i++)
    {
        triu v = (1 << i) - 1;
        triu end = v << (triuSize - i);
        while (v <= end)
        {
            if (checkDegrees(v, degreeFilter, nBeads))
                trius.push_back(v);
            v = nextPermutation(v);
        }
    }

    // Generate permuations of beads
    std::unordered_map<int, std::vector<int>> indexMap;
    for (uint i = 0; i < nBeads; ++i)
    {
        indexMap[beads[i]].push_back(i);
    }
    std::vector<int> permutation(nBeads);
    std::vector<std::vector<int>> permutations;
    generatePermutations(permutations, beads, nBeads, permutation, indexMap, 0);

    // Create graphs from adjacency matrices (easy to parallelize!)
    std::vector<Graph> graphs;
#pragma omp parallel for reduction(merge : graphs)
    for (uint i = 0; i < trius.size(); i++)
    {
        Graph graph = Graph(nBeads, beads, trius[i], degreeFilter, &permutations);
        if (graph.connected)
            graphs.push_back(graph);
    }

    // Filter out identical graphs
    std::map<repr, std::vector<Graph>> uniqueGraphs;
    for (auto &graph : graphs)
    {
        auto graphMapPtr = uniqueGraphs.find(graph.representation);
        if (graphMapPtr == uniqueGraphs.end())
        {
            uniqueGraphs[graph.representation] = std::vector<Graph>();
            uniqueGraphs[graph.representation].push_back(graph);
        }
        else
        {
            bool found = false;
            for (auto &uniqueGraph : graphMapPtr->second)
            {
                bool check = uniqueGraph == graph;
                if (check)
                {
                    found = true;
                    break;
                }
            }
            if (!found)
            {
                graphMapPtr->second.push_back(graph);
            }
        }
    }

    // Combine remaining graphs into a single list
    std::vector<Graph> uniqueGraphsList;
    for (auto &graphMap : uniqueGraphs)
    {
        uniqueGraphsList.insert(uniqueGraphsList.end(), graphMap.second.begin(), graphMap.second.end());
    }

    // Convert graphs to edge lists and return
    std::vector<std::vector<std::pair<uint, uint>>> bondsResult;
    for (auto &graph : uniqueGraphsList)
    {
        bondsResult.push_back(graph.getEdges());
    }
    return bondsResult;
}
