#include "iterate.h"

Graph::Graph(uint nBeads, uint *beads, triu adjacencyTriu, triu *degreeFilter, std::vector<std::vector<int>> *permutations)
    : nBeads(nBeads), beads(beads), adjacencyTriu(adjacencyTriu), degreeFilter(degreeFilter), permutations(permutations)
{
    this->adjacencyMatrix = this->getAdjacencyMatrix();
    this->connected = this->isConnected();
    this->representation = this->getRepresentation();
}

bool Graph::operator==(const Graph &other) const
{
    if (this->representation != other.representation)
        return false;

    for (auto &permutation : *this->permutations)
    {
        bool equal = true;
        for (uint i = 0; i < this->nBeads; i++)
        {
            for (uint j = i + 1; j < this->nBeads; j++)
            {
                if (this->adjacencyMatrix[i][j] != other.adjacencyMatrix[permutation[i]][permutation[j]])
                {
                    equal = false;
                    break;
                }
            }
            if (!equal)
                break;
        }
        if (equal)
            return true;
    }
    return false;
}

repr Graph::getRepresentation() const
{
    repr representation;
    // Total number of bonds (one integer)
    representation.push_back(BitCount(this->adjacencyTriu));
    // Degree of each bead sorted within each bead type group (nBeads integers)
    uint maxBead = this->beads[this->nBeads - 1] + 1;
    auto degreesList = std::vector<std::vector<int>>(maxBead, std::vector<int>());
    for (uint i = 0; i < this->nBeads; i++)
    {
        degreesList[beads[i]].push_back(BitCount(this->adjacencyTriu & this->degreeFilter[i]));
    }
    for (auto &degrees : degreesList)
    {
        std::sort(degrees.begin(), degrees.end());
        representation.insert(representation.end(), degrees.begin(), degrees.end());
    }
    // Number of bonds for each bond type (size depends on number of different bead types)
    std::map<std::pair<uint, uint>, uint> bondList;
    for (uint i = 0; i < this->nBeads; i++)
    {
        for (uint j = i + 1; j < this->nBeads; j++)
        {
            if (beads[i] > beads[j])
                bondList[{beads[j], beads[i]}] = 0;
            else
                bondList[{beads[i], beads[j]}] = 0;
        }
    }
    uint triuMaxIndex = this->nBeads * (this->nBeads - 1) / 2 - 1;
    uint count = 0;
    for (uint i = 0; i < this->nBeads; i++)
    {
        for (uint j = i + 1; j < this->nBeads; j++)
        {
            if ((this->adjacencyTriu & (1 << (triuMaxIndex - count))) != 0)
            {
                if (beads[i] > beads[j])
                    bondList[{beads[j], beads[i]}]++;
                else
                    bondList[{beads[i], beads[j]}]++;
            }
            count++;
        }
    }
    for (auto &bond : bondList)
    {
        representation.push_back(bond.second);
    }
    return representation;
}

std::vector<std::vector<bool>> Graph::getAdjacencyMatrix() const
{
    std::vector<std::vector<bool>> adjacencyMatrix(this->nBeads, std::vector<bool>(this->nBeads, false));
    uint triuMaxIndex = this->nBeads * (this->nBeads - 1) / 2 - 1;
    uint count = 0;
    for (uint i = 0; i < this->nBeads; i++)
    {
        for (uint j = i + 1; j < this->nBeads; j++)
        {
            bool value = this->adjacencyTriu & (1 << (triuMaxIndex - count));
            adjacencyMatrix[i][j] = value;
            adjacencyMatrix[j][i] = value;
            count++;
        }
    }
    return adjacencyMatrix;
}

bool Graph::isConnected() const
{
    std::vector<bool> visited(this->nBeads, false);
    std::queue<uint> queue;
    queue.push(0);
    visited[0] = true;
    while (!queue.empty())
    {
        uint current = queue.front();
        queue.pop();
        for (uint i = 0; i < this->nBeads; i++)
        {
            if (this->adjacencyMatrix[current][i] && !visited[i])
            {
                visited[i] = true;
                queue.push(i);
            }
        }
    }
    return std::all_of(visited.begin(), visited.end(), [](bool v)
                       { return v; });
}

std::vector<bool> Graph::getTriuVector() const
{
    std::vector<bool> triuVector;
    uint triuMaxIndex = this->nBeads * (this->nBeads - 1) / 2 - 1;
    for (uint i = 0; i <= triuMaxIndex; i++)
    {
        triuVector.push_back((this->adjacencyTriu & (1 << (triuMaxIndex - i))) != 0);
    }
    return triuVector;
}

std::vector<std::pair<uint, uint>> Graph::getEdges() const
{
    std::vector<std::pair<uint, uint>> edges;
    for (uint i = 0; i < this->nBeads; i++)
    {
        for (uint j = i + 1; j < this->nBeads; j++)
        {
            if (this->adjacencyMatrix[i][j])
            {
                edges.push_back({i, j});
            }
        }
    }
    return edges;
}
