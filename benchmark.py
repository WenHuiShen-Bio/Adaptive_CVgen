from __future__ import print_function
import sys
sys.path.append('/public/home/xdsc0166/sim_engine/ParmEd-master')
import os

import parmed as pmd
import parmed.unit as u
from parmed.openmm import StateDataReporter, NetCDFReporter, RestartReporter

import openmm.app as app
import openmm as mm
import openmm.unit as unit

import mdtraj as md
import numpy as np
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB import DSSP
import multiprocessing

from datetime import datetime
import argparse


def cpuinfo():
    """Return CPU info"""
    import platform, subprocess, re
    if platform.system() == "Windows":
        return platform.processor()
    elif platform.system() == "Darwin":
        os.environ['PATH'] = os.environ['PATH'] + os.pathsep + '/usr/sbin'
        command ="sysctl -n machdep.cpu.brand_string"
        return subprocess.check_output(command, shell=True, text=True).strip()
    elif platform.system() == "Linux":
        command = "cat /proc/cpuinfo"
        all_info = subprocess.check_output(command, shell=True, text=True).strip()
        for line in all_info.split("\n"):
            if "model name" in line:
                return re.sub( ".*model name.*: ", "", line,1)
    return ""

def appendTestResult(filename=None, result=None, system_info=None):
    """Append a test result to a JSON or YAML file.

    TODO: The efficiency of this could be improved.

    Parameters
    ----------
    filename : str, optional, default=None
        The filename to append a result to, ending either in .yaml or .json
        If None, no result is written
    test_result : dict, optional, default=None
        The test result to append to the 'benchmarks' blcok
    system_info : dict, optional, default=None
        System info to append to the 'system' block
    """
    # Do nothing if filename is None
    if filename is None:
        return

    all_results = { 'benchmarks' : list() }
    if system_info is not None:
        all_results['system'] = system_info

    if filename.endswith('.yaml'):
        # Append test result to a YAML file, creating if file does not exist.
        import yaml
        if os.path.exists(filename):
            with open(filename, 'rt') as infile:
                all_results = yaml.safe_load(infile)
        if result is not None:
            all_results['benchmarks'].append(result)
        with open(filename, 'wt') as outfile:
            yaml.dump(all_results, outfile, sort_keys=False)
    elif filename.endswith('.json'):
        # Append test result to a JSON file, creating if file does not exist.
        import json
        if os.path.exists(filename):
            with open(filename, 'rt') as infile:
                all_results = json.load(infile)
        if result is not None:
            all_results['benchmarks'].append(result)
        with open(filename, 'wt') as outfile:
            json.dump(all_results, outfile, sort_keys=False, indent=4)
    elif filename.endswith('.log'):
        if os.path.exists(filename):
            with open(filename, 'r') as infile:
                all_results = infile.read()
        if result is not None:
            all_results['benchmarks'].append(result)
        with open(filename, 'w') as outfile:
            outfile.write(all_results)
    else:
        raise ValueError('--output filename must end with .json or .yaml')

def printTestResult(result, options):
    """Render a test result

    Parameters
    ----------
    result : dict
        The test result
    options :
        Options structure
    """
    if options.style == 'simple':
        for (key, value) in result.items():
            print(f'{key}: {value}')
        print('')
    elif options.style == 'table':
        print('%-18s%-12s%-14s%-15s%-10g%-11s%-11s%-g' %
              (result['precision'],
               result['timestep_in_fs'],
               result['platform'],
               result['ns_per_day']))
    else:
        raise ValueError(f"style must be one of ['simple', 'table']")

def timeIntegration(sim, dsteps):
    """Integrate a Context for a specified number of steps, then return how many seconds it took."""
    #context.getIntegrator().step(initialSteps) # Make sure everything is fully initialized
    #context.getState(getEnergy=True)
    start = datetime.now()
    sim.step(dsteps)
    #context.getIntegrator().step(steps)
    #context.getState(getEnergy=True)
    end = datetime.now()
    elapsed = end-start
    return elapsed.seconds + elapsed.microseconds*1e-6

import functools
@functools.lru_cache(maxsize=None)
def retrieveTestSystem(inpdb):
    """Retrieve a benchmark system

    Parameters
    ----------
    topfile : str
        The file name of the topology
    inpcrd : str
        The file name of input coordinate

    Returns
    -------
    system : openmm.System
        The test system object
    positions : openmm.unit.Quantity with shape (natoms,3)
        The initial positions
    parameters : dict of str : str
        Special test parameters to report in test results

    """

    # Create dictionary of test parameters
    parameters = dict()

    #protein = pmd.load_file(topfile, inpcrd)
    pdb = app.PDBFile(inpdb)
    forcefield = app.ForceField('amber14/protein.ff14SB.xml', 'implicit/gbn2.xml')

    # # Create the OpenMM system
    print('Creating OpenMM System')
    system = forcefield.createSystem(pdb.topology, 
                                     nonbondedMethod=app.NoCutoff,
                                	constraints=app.HBonds
                                )
    # # Create the OpenMM system
    # print('Creating OpenMM System')
    # system = forcefield.createSystem(nonbondedMethod=app.NoCutoff,
    #                             constraints=app.HBonds, implicitSolvent=app.GBn2,
    #                             implicitSolventSaltConc=0.1*u.moles/u.liter,
    #                             )

    # # Create the integrator to do Langevin dynamics
    integrator = mm.LangevinIntegrator(
                            300*u.kelvin,       # Temperature of heat bath
                            1.0/u.picoseconds,  # Friction coefficient
                            2.5*u.femtoseconds  # Time step
    )

    # Set the particle positions
    positions = pdb.positions
    topology = pdb.topology


    return topology, system, integrator, positions, parameters

def runOneTest(options):

    """Perform a single benchmarking simulation."""

    topology, system, integrator, positions, parameters = retrieveTestSystem(inpdb=options.inpdb)

    # Create a copy of the basic parameters dict (which may be cached) to report the test results
    result = parameters.copy()


    properties = {}
    platform = mm.Platform.getPlatformByName(options.platform)
    if options.device is not None and 'DeviceIndex' in platform.getPropertyNames():
        properties['DeviceIndex'] = options.device
        if ',' in options.device or ' ' in options.device:
            initialSteps = 250
    if options.opencl_platform is not None and 'OpenCLPlatformIndex' in platform.getPropertyNames():
        properties['OpenCLPlatformIndex'] = options.opencl_platform
    if (options.precision is not None) and ('Precision' in platform.getPropertyNames()):
        properties['Precision'] = options.precision

    #properties = {"Precision": "single"}
    # Create the Context
    #integrator.setConstraintTolerance(1e-5)
    if len(properties) > 0:
        sim = app.Simulation(topology, system, integrator, platform, properties)
        print(properties)
        #context = mm.Context(system, integrator, platform, properties)
    else:
        sim = app.Simulation(topology, system, integrator, platform)
        #context = mm.Context(system, integrator, platform)

    #sim.context = context

    # Store information about the Platform used by the Context
    platform = sim.context.getPlatform()
    result['platform'] = platform.getName()
    result['platform_properties'] = { property_name : platform.getPropertyValue(sim.context, property_name) for property_name in platform.getPropertyNames() }

    # Prepare the simulation
    sim.context.setPositions(positions)
    steps = options.steps
    dsteps = options.dsteps
    nth_round = options.nth_round

    

    # pramed report
    log_file = options.outdir + '/log/replica' + str(options.replicaid) + '/' + str(nth_round) + '.log'
    if os.path.isfile(log_file):
        sim.reporters.append(
            app.StateDataReporter(log_file, dsteps, step=True, potentialEnergy=True,
                                   kineticEnergy=True, temperature=True, append=True)
        )
    else:
        sim.reporters.append(
            app.StateDataReporter(log_file, dsteps, step=True, potentialEnergy=True,
                                   kineticEnergy=True, temperature=True)
        )
    # sim.reporters.append(
    #         NetCDFReporter(options.outdir + '/traj/' + str(options.replicaid) + '.nc', dsteps, crds=True)
    # )
    dcd_file = options.outdir + '/traj/replica' + str(options.replicaid) + '/' + str(nth_round) + '.dcd'
    if os.path.isfile(dcd_file):
        sim.reporters.append(
            app.DCDReporter(dcd_file, dsteps, append=True)
        )
    else:
        sim.reporters.append(
            app.DCDReporter(dcd_file, dsteps)
        )
    sim.reporters.append(
            app.CheckpointReporter(options.outdir + '/checkpoint/replica' + str(options.replicaid) + '/' + str(nth_round) + '.chk', dsteps)
    )

    chk_file = options.outdir + '/checkpoint/replica' + str(options.replicaid) + '/' + str(nth_round) + '.chk'
    if os.path.isfile(chk_file):
        print('Loading a exit checkpoint file ...')
        sim.loadCheckpoint(chk_file)

    # # # Minimize the energy
    # print('Minimizing energy')
    # sim.minimizeEnergy(maxIterations=50000)

    elapsed_time = timeIntegration(sim, steps)
    result['steps'] = steps
    result['elapsed_time'] = elapsed_time
    time_per_step = elapsed_time * unit.seconds / steps 
    ns_per_day = (integrator.getStepSize() / time_per_step) / (unit.nanoseconds/unit.day)
    result['ns_per_day'] = ns_per_day

    printTestResult(result, options)
    appendTestResult(result=result, filename=options.outdir + '/' + str(options.replicaid) + '.json')

    # Clean up
    del sim, integrator

# Parse the command line options.

platform_speeds = { mm.Platform.getPlatform(i).getName() : mm.Platform.getPlatform(i).getSpeed() for i in range(mm.Platform.getNumPlatforms()) }
PLATFORMS = [platform for platform, speed in sorted(platform_speeds.items(), key=lambda item: item[1], reverse=True)]
PRECISIONS = ('single', 'mixed', 'double')
STYLES = ('simple', 'table')

parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                 description="Run one or more benchmarks of OpenMM",
                                epilog="""
Example: run the full suite of benchmarks for the CUDA platform, printing the results as a table

    python benchmark.py --platform=CUDA --style=table

Example: run the apoa1pme benchmark for the CPU platform with a reduced cutoff distance

    python benchmark.py --platform=CPU --test=apoa1pme --pme-cutoff=0.8

Example: run the full suite in mixed precision mode, saving the results to a YAML file

    python benchmark.py --platform=CUDA --precision=mixed --outfile=benchmark.yaml""")

#parser.add_argument('--topfile', dest='topfile', help='name of the topology filename')
parser.add_argument('--inpdb', dest='inpdb', help='name of the coordinate filename')
parser.add_argument('--steps', dest='steps', type=int, help='total number of steps to run the simulation')
parser.add_argument('--dsteps', dest='dsteps', type=int, help='number of steps interval to adaptivly check the simulation')
parser.add_argument('--nth_round', dest='nth_round', type=int, help='nth round')
parser.add_argument('--platform', dest='platform', choices=PLATFORMS, help='name of the platform to benchmark')
parser.add_argument('--seconds', default=60, dest='seconds', type=float, help='target simulation length in seconds [default: 60]')
parser.add_argument('--device', default=None, dest='device', help='device index for CUDA or OpenCL')
parser.add_argument('--opencl-platform', default=None, dest='opencl_platform', help='platform index for OpenCL')
parser.add_argument('--precision', default='single', dest='precision', help=f'precision modes for CUDA or OpenCL: {PRECISIONS} [default: single]')
parser.add_argument('--style', default='simple', dest='style', choices=STYLES, help=f'output style: {STYLES} [default: simple]')
parser.add_argument('--outdir', default=None, dest='outdir', type=str, help='output filen directory')
parser.add_argument('--replicaid', default=None, dest='replicaid', type=int, help='output replica id')
parser.add_argument('--verbose', default=False, action='store_true', dest='verbose', help='if specified, print verbose output')
args = parser.parse_args()
if args.platform is None:
    parser.error('No platform specified')

if args.steps is None:
    parser.error('Need to specify "--steps"')

if args.dsteps is None:
    parser.error('Need to specify "--dsteps"')

# Collect system information
system_info = dict()
import socket, platform
system_info['hostname'] = socket.gethostname()
system_info['timestamp'] = datetime.utcnow().isoformat()
system_info['openmm_version'] = mm.version.version
system_info['cpuinfo'] = cpuinfo()
system_info['cpuarch'] = platform.processor()
system_info['system'] = platform.system()
# TODO: Capture information on how many CPU threads will be used

# Attempt to get GPU info
try:
    import subprocess
    cmd = 'rocm-smi --showfwinfo --save=1.txt'
    output = subprocess.check_output(cmd, shell=True, text=True)
    system_info['nvidia_driver'], system_info['gpu'] = output.strip().split(', ')
except Exception as e:
    pass

for key, value in system_info.items():
    print(f'{key}: {value}')

if (args.outdir + '/' + str(args.replicaid) + '.json') is not None:
    # Remove output file
    if os.path.exists(args.outdir + '/' + str(args.replicaid) + '.json'):
        os.unlink(args.outdir + '/' + str(args.replicaid) + '.json')
    # Write system info
    appendTestResult(args.outdir + '/' + str(args.replicaid) + '.json', system_info=system_info)

precisions = args.precision.split(',')
if args.platform == 'Reference':
    precisions = ['double']
if args.platform == 'CPU':
    precisions = ['mixed']
if not set(precisions).issubset(PRECISIONS):
    parser.error(f'Available precisions: {PRECISIONS}')


# Combinatorially run all requested benchmarks, ignoring combinations that cannot be run
from openmm import OpenMMException

if args.style == 'simple':
    try:
        runOneTest(args)
    except OpenMMException as e:
        if args.verbose:
            print(e)
        pass
elif args.style == 'table':
    print()
    print('              Precision   dt (fs)   Platform   ns/day')
    try:
        runOneTest(args)
    except OpenMMException as e:
        if args.verbose:
            print(e)
        pass
else:
    raise ValueError(f"style {args.style} unknown; must be one of ['simple', 'table']")
