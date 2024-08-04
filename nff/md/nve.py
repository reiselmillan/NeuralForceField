import numpy as np
import os, sys
import time
from ase import units
from ase.md.velocitydistribution import (MaxwellBoltzmannDistribution,
                                         Stationary,
                                         ZeroRotation)
from ase.md.verlet import VelocityVerlet
from nff.md.npt import NoseHoovernpt
from ase.io import Trajectory, iread
from ase.io import write as asewrite
from nff.nn.utils import exploded

from nff.md.utils import NeuralMDLogger, write_traj

DEFAULTNVEPARAMS = {
    'T_init': 120.0,
    # thermostat can be NoseHoover, Langevin, NPT, NVT,
    # Thermodynamic Integration...
    'thermostat': VelocityVerlet,
    'thermostat_params': {'timestep': 0.5},
    'nbr_list_update_freq': 20,
    'steps': 3000,
    'save_frequency': 10,
    'thermo_filename': './thermo.log',
    'traj_filename': './atoms.traj',
    'skip': 0
}


class Dynamics:

    def __init__(self,
                 atomsbatch,
                 mdparam=DEFAULTNVEPARAMS,
                 atomsbatch_to_log=None,
                 ):

        # initialize the atoms batch system
        self.atomsbatch = atomsbatch
        if atomsbatch_to_log is None:
            self.atomsbatch_to_log = atomsbatch
        else:
            self.atomsbatch_to_log = atomsbatch_to_log
        self.mdparam = mdparam
        
        # track time
        self.max_time = mdparam.get("max_time", 255600) # 72 hours
        self.init_time = time.time()

        self.steps = int(self.mdparam['steps'])
        self.restart_on_crash = self.mdparam.get("restart_on_crash", False)
        self.restart_from = self.mdparam.get("restart_from", -1)
        self.stop = False
        
        # set temp ramp
        if mdparam.get("temp_ramp", False):
            self.tempramp = (self.mdparam['temperature'] - 100)*self.mdparam['nbr_list_update_freq']/self.steps
            print("setting temperature ramp ", self.tempramp) 
            self.mdparam['temperature'] = 100
        else:
            self.tempramp = 0.0

        # todo: structure optimization before starting

        # intialize system momentum
        MaxwellBoltzmannDistribution(self.atomsbatch,
                                     temperature_K = self.mdparam['temperature'])
        Stationary(self.atomsbatch)  # zero linear momentum
        ZeroRotation(self.atomsbatch)

        # set thermostats
        integrator = self.mdparam['thermostat']
        if integrator == VelocityVerlet:
            dt = self.mdparam['thermostat_params']['timestep']
            self.integrator = integrator(self.atomsbatch,
                                         timestep=dt * units.fs)
        else:
            self.integrator = integrator(self.atomsbatch,
                                         **self.mdparam['thermostat_params'],
                                         **self.mdparam)


        self.check_restart()
        # if self.steps == int(self.mdparam['steps']):
            # attach trajectory dump
            # self.traj = Trajectory(
            #     self.mdparam['traj_filename'], 'w', self.atomsbatch_to_log)
        self.integrator.attach(
            self.write_frame, interval=self.mdparam['save_frequency'])

        if hasattr(self.atomsbatch.calc, "write"):
            self.integrator.attach(
                self.write_cv, interval=1)

        # attach log file
        requires_stress = 'stress' in self.atomsbatch.calc.properties
        self.integrator.attach(NeuralMDLogger(self.integrator,
                                            self.atomsbatch_to_log,
                                            self.mdparam['thermo_filename'],
                                            stress=requires_stress,
                                            mode='a'),
                            interval=self.mdparam['save_frequency'])


    def check_restart(self, nframe=-1):
        if os.path.exists(self.mdparam['traj_filename']):
            # new_atoms = Trajectory(self.mdparam['traj_filename'])[-1]

            # calculate number of steps remaining
            # self.steps = ( int(self.mdparam['steps'])
            #     - ( int(self.mdparam['save_frequency']) *
            #     len(Trajectory(self.mdparam['traj_filename'])) ) )

            # get last atom
            traj = iread(self.mdparam['traj_filename'])
            new_atoms = None
            for n, atoms in enumerate(traj):
                new_atoms = atoms
                if n != -1 and n == nframe:
                    break

            if new_atoms is None:
                print("Restart atoms failed !!")
                return
            print("set new velocities")
            self.atomsbatch.set_cell(new_atoms.get_cell())
            self.atomsbatch.set_positions(new_atoms.get_positions())
            self.atomsbatch.set_velocities(new_atoms.get_velocities())
            # self.integrator.atomsbatch = self.atomsbatch

    def setup_restart(self, restart_param):
        """If you want to restart a simulations with predfined mdparams but
        longer you need to provide a dictionary like the following:

         note that the thermo_filename and traj_name should be different

         restart_param = {'atoms_path': md_log_dir + '/atom.traj',
                          'thermo_filename':  md_log_dir + '/thermo_restart.log',
                          'traj_filename': md_log_dir + '/atom_restart.traj',
                          'steps': 100
                          }

        Args:
            restart_param (dict): dictionary to contains restart paramsters and file paths
        """

        if restart_param['thermo_filename'] == self.mdparam['thermo_filename']:
            raise ValueError("{} is also used, \
                please change a differnt thermo file name".format(restart_param['thermo_filename']))

        if restart_param['traj_filename'] == self.mdparam['traj_filename']:
            raise ValueError("{} is also used, \
                please change a differnt traj file name".format(restart_param['traj_filename']))

        self.restart_param = restart_param
        new_atoms = Trajectory(restart_param['atoms_path'])[-1]

        self.atomsbatch.set_positions(new_atoms.get_positions())
        self.atomsbatch.set_velocities(new_atoms.get_velocities())

        # set thermostats
        integrator = self.mdparam['thermostat']
        self.integrator = integrator(
            self.atomsbatch, **self.mdparam['thermostat_params'])

        # attach trajectory dump
        self.traj = Trajectory(
            self.restart_param['traj_filename'], 'w', self.atomsbatch)
        self.integrator.attach(
            self.traj.write, interval=self.mdparam['save_frequency'])

        # attach log file
        requires_stress = 'stress' in self.atomsbatch.calc.properties
        self.integrator.attach(NeuralMDLogger(
            self.integrator,
            self.atomsbatch,
            self.restart_param['thermo_filename'],
            stress=requires_stress,
            mode='a'),
            interval=self.mdparam['save_frequency'])

        self.mdparam['steps'] = restart_param['steps']

    def run(self):

        epochs = int(self.steps //
                     self.mdparam['nbr_list_update_freq'])
        # In case it had neighbors that didn't include the cutoff skin,
        # for example, it's good to update the neighbor list here
        self.atomsbatch.update_nbr_list()
        print("NVE loop steps: ", epochs)
        for step in range(epochs):
            if time.time() - self.init_time >= self.max_time:
                return
            if self.stop:
                return
            
            self.integrator.increment_temperature(self.tempramp)
            try:
              self.integrator.run(self.mdparam['nbr_list_update_freq'])
            except Exception as e:
              print(e)
              if self.restart_on_crash:
                print(f"restarting from the {self.restart_from} frame. Error of the integrator")
                self.check_restart(self.restart_from)


            # # unwrap coordinates if mol_idx is defined
            # if self.atomsbatch.props.get("mol_idx", None) :
            #     self.atomsbatch.set_positions(self.atoms.get_positions(wrap=True))
            #     self.atomsbatch.set_positions(reconstruct_atoms(atoms, self.atomsbatch.props['mol_idx']))
            self.atomsbatch.update_nbr_list()

        #self.traj.close()

    def write_frame(self):
        if exploded(self.atomsbatch, 0.7):
            print("System exploded !!!")
            if self.restart_on_crash:
                print(f"restarting from the {self.restart_from} frame")
                self.check_restart(self.restart_from)
                return
            
            self.stop = True
            self.integrator.stop = True
            sys.exit(0)

        asewrite(self.mdparam['traj_filename'], self.atomsbatch, append=True)
        with open("UNCERTAINTIES.dat", "a") as f:
            # uncertainty in kcal
            f.write(f"{self.atomsbatch.results['forces_std'].mean()*23.06}\n") 
    
    def write_cv(self):
        self.atomsbatch.calc.write(self.atomsbatch)

    def save_as_xyz(self, filename='./traj.xyz'):

        traj = Trajectory(self.mdparam['traj_filename'], mode='r')

        xyz = []

        skip = self.mdparam['skip']
        traj = list(traj)[skip:] if len(traj) > skip else traj

        for snapshot in traj:
            frames = np.concatenate([
                snapshot.get_atomic_numbers().reshape(-1, 1),
                snapshot.get_positions().reshape(-1, 3)
            ], axis=1)

            xyz.append(frames)

        write_traj(filename, np.array(xyz))
