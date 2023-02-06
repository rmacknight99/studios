import sys, os, subprocess
from os import system
from os import path

import cctk
import autode as ade

import numpy as np
import copy

import ipywidgets
from ipywidgets import interact, fixed, IntSlider
import warnings
warnings.filterwarnings('ignore')
import py3Dmol

from rdkit.Chem import AllChem as Chem
from rdkit.Chem import Draw
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("/etc/opt/orca-5.0.3/")
sys.path.append("/etc/opt/openmpi-4.1.1/bin")

ORCA_PATH = '/etc/opt/orca-5.0.3'
OPEN_MPI_PATH = '/etc/opt/openmpi-4.1.1/'
OPEN_MPI_BIN_PATH = '/etc/opt/openmpi-4.1.1/bin'
OPEN_MPI_LIB_PATH = '/etc/opt/openmpi-4.1.1/lib'

os.environ['PATH'] += os.pathsep + ORCA_PATH
os.environ['PATH'] += os.pathsep + OPEN_MPI_PATH
os.environ['PATH'] += os.pathsep + OPEN_MPI_BIN_PATH

#os.environ['NBOBIN'] = '/opt/nbo7/bin/'
os.environ['NBOEXE'] = '/opt/nbo7/bin/nbo7.i4.exe'
os.environ['GENEXE'] = '/opt/nbo7/bin/gennbo.i4.exe'

old = os.environ.get("LD_LIBRARY_PATH")
if old:
    os.environ["LD_LIBRARY_PATH"] = old + ":" + 'PATH'
else:
    os.environ["LD_LIBRARY_PATH"] = 'PATH'

os.environ['LD_LIBRARY_PATH'] += os.pathsep + ORCA_PATH
os.environ['LD_LIBRARY_PATH'] += os.pathsep + OPEN_MPI_LIB_PATH

os.environ['OMPI_ALLOW_RUN_AS_ROOT'] = '1'
os.environ['OMPI_ALLOW_RUN_AS_ROOT_CONFIRM'] = '1'

ade.Config.n_cores = 16
ade.Config.max_core = 2000

def MolTo3DView(xyz, size=(300, 300), style="stick", surface=False, opacity=0.5):
    """Draw molecule in 3D
    
    Args:
    ----
        mol: rdMol, molecule to show
        size: tuple(int, int), canvas size
        style: str, type of drawing molecule
               style can be 'line', 'stick', 'sphere', 'carton'
        surface, bool, display SAS
        opacity, float, opacity of surface, range 0.0-1.0
    Return:
    ----
        viewer: py3Dmol.view, a class for constructing embedded 3Dmol.js views in ipython notebooks.
    """
    assert style in ('line', 'stick', 'sphere', 'carton')
    # mblock = Chem.MolToMolBlock(mol)
    # xyz = mblock
    viewer = py3Dmol.view(width=size[0], height=size[1])
    viewer.addModel(open(xyz, 'r').read(), 'xyz')
    viewer.setStyle({style:{}})
    viewer.setBackgroundColor('0xeeeeee')

    viewer.setHoverable({},True,'''function(atom,viewer,event,container) {
                   if(!atom.label) {
                    atom.label = viewer.addLabel(atom.atom+":"+atom.serial,{position: atom, backgroundColor: 'mintcream', fontColor:'black'});
                   }}''',
               '''function(atom,viewer) { 
                   if(atom.label) {
                    viewer.removeLabel(atom.label);
                    delete atom.label;
                   }
                }''')


    if surface:
        viewer.addSurface(py3Dmol.SAS, {'opacity': opacity})
    viewer.zoomTo()
    return viewer

#################################################################
#                                                               #
#                      06-310 Utilities                         #
#                                                               #
#################################################################
"""
def vibration(xyz, frequency, size=(300, 300), style="stick", surface=False, opacity=0.5):
    assert style in ('line', 'stick', 'sphere', 'carton')
    viewer = py3Dmol.view(width=size[0], height=size[1])
    viewer.addModel(open(xyz, 'r').read(), 'xyz', {'vibrate': {'frames':10,'amplitude':frequency}})
    viewer.setStyle({style:{}})
    viewer.setBackgroundColor('0xeeeeee')

    viewer.setHoverable({},True,'''function(atom,viewer,event,container) {
                   if(!atom.label) {
                    atom.label = viewer.addLabel(atom.atom+":"+atom.serial,{position: atom, backgroundColor: 'mintcream', fontColor:'black'});
                   }}''',
               '''function(atom,viewer) { 
                   if(atom.label) {
                    viewer.removeLabel(atom.label);
                    delete atom.label;
                   }
                }''')


    if surface:
        viewer.addSurface(py3Dmol.SAS, {'opacity': opacity})
    viewer.zoomTo()
    return viewer

def view_scan(directory):
    confs = [os.path.join(directory, xyz) for xyz in os.listdir(directory) if xyz.endswith("xyz")]
    numbers = [float(i.split("/")[1].split("_")[2]) for i in confs]
    zipped = zip(confs, numbers)
    sorted_result = sorted(zipped, key = lambda x: x[1])
    confs, numbers = zip(*sorted_result)
    return confs, numbers

def structure_viewer(idx):
    mol = structures[idx]
    return MolTo3DView(mol).show()

def get_normal_mode(molecule, normal_mode):
    elements = molecule.atomic_symbols
    coords = np.array(molecule.coordinates) # To transform from au to A
    natm = molecule.n_atoms
    vib_xyz = "%d\n\n" % natm
    nm = normal_mode.reshape(natm, 3)
    for i in range(natm):
        # add coordinates:
        vib_xyz += elements[i] + " %15.7f %15.7f %15.7f " % (coords[i,0], coords[i,1], coords[i,2])
        # add displacements:
        vib_xyz += "%15.7f %15.7f %15.7f\n" % (nm[i,0], nm[i,1], nm[i,2])
    return vib_xyz

def plotter(x, y, data_label, title, x_label="X", y_label="Y", color="black", marker=None, multiple=False, scatter=False):
    # make figure and axes
    fig, ax = plt.subplots(figsize=(12, 8))
    # aesthetic settings for plot
    font = 'Arial'
    SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE = 16, 20, 24
    ax.grid(True, linewidth=1.0, color='0.95')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_axisbelow(True)
    for axis in ['bottom', 'left']:
        ax.spines[axis].set_linewidth(3.0)
    for tick in ax.get_yticklabels():
        tick.set_fontname(font)
        tick.set_fontsize(SMALL_SIZE)
    for tick in ax.get_xticklabels():
        tick.set_fontname(font)
        tick.set_fontsize(SMALL_SIZE)

    if multiple:
        for i in range(len(y)):
            plt.plot(x, y[i], label = data_label[i], color=color[i], marker=marker, linewidth=2.0)
    elif scatter:
        plt.scatter(x, y, label=data_label, color=color, marker=".", linewidth=2.0)
    else:
        plt.plot(x, y, label=data_label, color=color, marker=marker, linewidth=2.0)
    ax.legend(fontsize=SMALL_SIZE, loc=0)
    plt.title(title, fontsize=SMALL_SIZE, fontname=font)
    plt.xlabel(f"{x_label}", fontsize=SMALL_SIZE, fontname=font)
    plt.ylabel(f"{y_label}", fontsize=SMALL_SIZE, fontname=font)
    
    return fig, ax
"""