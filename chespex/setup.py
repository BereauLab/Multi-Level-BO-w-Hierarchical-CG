""" Setup file to compile the C++ extension module
for the molecules module of the chespex package. """
from setuptools import setup, Extension

setup_args = dict(
    ext_modules = [
        Extension(
            'chespex.molecules.bond_generator',
            [
                'chespex/molecules/bonds/bondgenerator.cpp',
                'chespex/molecules/bonds/iterate.cpp',
                'chespex/molecules/bonds/graph.cpp'
            ],
            extra_compile_args = ['-fopenmp'],
            py_limited_api = True
        )
    ]
)
setup(**setup_args)
