#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "iterate.h"

static PyObject *generate_bonds(PyObject *self, PyObject *args)
{
    // Convert the tuple to a vector of integers and perform some checks
    uint size = (uint)PyTuple_GET_SIZE(args);
    uint beads[size];
    for (uint i = 0; i < size; i++)
    {
        PyObject *item = PyTuple_GET_ITEM(args, i);
        if (!PyLong_Check(item))
        {
            PyErr_SetString(PyExc_TypeError, "All elements of the tuple must be integers");
            return NULL;
        }
        uint beadType = PyLong_AsLong(item);
        if (beadType < 0)
        {
            PyErr_SetString(PyExc_ValueError, "All elements of the tuple must be non-negative integers");
            return NULL;
        }
        if (i == 0 && beadType != 0)
        {
            PyErr_SetString(PyExc_ValueError, "The first element of the tuple must be 0");
            return NULL;
        }
        else if (i > 0 && (beadType != beads[i - 1] && beadType != beads[i - 1] + 1))
        {
            PyErr_SetString(PyExc_ValueError, "The tuple must be strictly increasing");
            return NULL;
        }
        beads[i] = beadType;
    }

    // Generate all possible bond configurations
    if (size > 1)
    {
        std::vector<std::vector<std::pair<uint, uint>>> bonds = getBonds(size, beads);

        PyObject *result = PyList_New(bonds.size());
        for (uint i = 0; i < bonds.size(); i++)
        {
            PyObject *bondList = PyList_New(bonds[i].size());
            for (uint j = 0; j < bonds[i].size(); j++)
            {
                PyList_SET_ITEM(bondList, j, Py_BuildValue("(ii)", bonds[i][j].first, bonds[i][j].second));
            }
            PyList_SET_ITEM(result, i, bondList);
        }

        return result;
    }
    else
    {
        // If there is only one bead, there are no bonds
        PyObject *result = PyList_New(1);
        PyList_SET_ITEM(result, 0, PyList_New(0));
        return result;
    }
}

static PyMethodDef BondsGeneratorMethods[] = {
    {"_generate", generate_bonds, METH_VARARGS,
     "Generate all valid bond configurations for a given sequence of ordered beads."},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef bondgeneratorModule = {
    PyModuleDef_HEAD_INIT,
    "bond_generator", /* name of module */
    "Helper module for fast bond generation. Not intended for direct use.",
    -1, /* size of per-interpreter state of the module,
           or -1 if the module keeps state in global variables. */
    BondsGeneratorMethods};

PyMODINIT_FUNC
PyInit_bond_generator(void)
{
    return PyModule_Create(&bondgeneratorModule);
}
