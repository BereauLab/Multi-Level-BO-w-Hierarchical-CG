""" Script to setup the membrane systems. """
import os
import argparse
import subprocess
from chespex.simulation import Simulation, SimulationSetup

# Make sure that you have the most recent version of the insane tool.
# The version on PyPI might be outdated.

membranes = {
    'DPPC': {
        'command': '-x {size} -y {size} -z {height} -l M3.DPPC -sol W -pbc cubic -solr 0.7',
        'membrane_selection': lambda x: x[x['resname'] == 'DPPC'],
        'default_height': 10.5
    },
    'DIPC': {
        # We use DIPC from Martini2 since it is not available in Martini3 and
        # the bead types are the same
        'command': '-x {size} -y {size} -z {height} -l M2.DIPC -sol W -pbc cubic -solr 0.7',
        'membrane_selection': lambda x: x[x['resname'] == 'DIPC'],
        'default_height': 11.5
    },
    'MIX': {
        # We use DIPC from Martini2 since it is not available in Martini3 and
        # the bead types are the same
        'command': '-x {size} -y {size} -z {height} -l M3.DPPC:7 ' + \
            '-l M2.DIPC:4.7 -l M3.CHOL:5 -sol W -pbc cubic -solr 0.7',
        'membrane_selection': lambda x: x[x['resname'].isin(['DPPC', 'DIPC', 'CHOL'])],
        'default_height': 10.5
    }
}

def main(membrane_type: str, size: float, height: float, directory: str):
    """ Setup a membrane system """
    props = membranes[membrane_type]
    # Check if the directory already exists or create it
    if os.path.exists(directory):
        print(f'Aborting: Directory {directory} already exists')
        return
    os.makedirs(directory)
    # Generate membrane
    insane_args = f'insane -o {directory}/membrane.gro -p {directory}/membrane.top '
    height = height if height is not None else props['default_height']
    insane_args += props["command"].format(size=size, height=height)
    print(insane_args)
    subprocess.run(insane_args, shell=True, check=True)
    # Create index file
    setup = SimulationSetup(f'{directory}/membrane')
    setup.create_index(f'{directory}/membrane.ndx', {
        'membrane': props['membrane_selection'],
        'water': lambda x: x[x['resname'] == 'W'],
    })
    setup.replace_string_in_topology('#include "martini.itp"', [
        '#include "martini3.ff/martini.itp"',
        '#include "martini3.ff/martini_solvents.itp"',
        '#include "martini3.ff/martini_ions.itp"',
        '#include "martini3.ff/martini_phospholipids.itp"',
        '#include "martini3.ff/martini_sterols.itp"'
    ])
    # Run energy minimizations
    minimization = Simulation(
        f'{directory}/membrane.gro',
        f'{directory}/membrane.top',
        'minimization.mdp',
        f'{directory}/minimization.tpr'
    )
    minimization.run()
    # Run NPT equilibration
    equilibration = Simulation(
        f'{directory}/minimization.gro',
        f'{directory}/membrane.top',
        f'{directory}/membrane.ndx',
        'equilibration.mdp',
        f'{directory}/equilibration.tpr'
    )
    equilibration.run()

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Setup membrane systems')
    argparser.add_argument('-t', '--type', choices=['DPPC', 'DIPC', 'MIX'], default='MIX',
        help='Type of membrane to setup')
    argparser.add_argument('-s', '-xy', '--size', type=float, default=6.0,
        help='X and Y size of the membrane')
    argparser.add_argument('-z', '--height', type=float, default=None,
        help='Height of the simulation box')
    argparser.add_argument('-d', '--directory', default='membrane',
        help='Directory to store the membrane files')
    args = argparser.parse_args()
    main(args.type, args.size, args.height, args.directory)
