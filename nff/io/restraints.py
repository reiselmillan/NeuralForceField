import torch
from torch.autograd import grad
from nff.io.colvars import get_cv_from_dic
from ase.calculators.calculator import Calculator, all_changes


class HarmonicRestraintStatic:
    def __init__(self, cvdic, device="cpu") -> None:
        self.cvs = [] # list of collective variables (CV) objects
        self.kappas = []  # list of constant values for every CV
        self.eq_values = [] # list equilibrium values for every CV
        self.device = device
        self.results = {}
        self.setup(cvdic, device)

    def setup(self, cvdic, device):       
        """ Initializes the collectiva variables objects
        Args:
            cvdic (dict): Dictionary contains the information to define the collective variables
                          and the harmonic restraint.
            max_steps (int): maximum number of steps of the MD simulation
            device: device
        """
        for _ , val in cvdic.items():
            cv = get_cv_from_dic(val, device)
            self.cvs.append(cv)
            self.kappas.append(val["restraint"]["kappa"])
            self.eq_values.append(val["restraint"]["eq_val"])

    def get_energy(self, positions):
        """ Calculates the retraint energy of an arbritrary state of the system
        Args:
            positions (torch.tensor): atomic positions
            step (int): current step
        Returns:
            float: energy
        """
        tot_energy = 0
        for n, cv in enumerate(self.cvs):
            kappa = self.kappas[n]
            eq_val = self.eq_values[n]
            cv_value = cv.get_value(positions)
            energy = 0.5 * kappa * (cv_value - eq_val) * (cv_value - eq_val)
            tot_energy += energy
        return tot_energy

    def get_bias(self, positions):
        """ Calculates the bias energy and force

        Args:
            positions (torch.tensor): atomic positions
            step (int): current step

        Returns:
            float: forces
            float: energy
        """
        energy = self.get_energy(positions)
        forces = -grad(energy, positions)[0]
        return forces, energy
    
    # ase interface
    def calculate(self, atoms, properties=["energy", "forces"], all_changes=all_changes):
        forces, energy = self.get_bias(torch.tensor(atoms.positions, 
                                                requires_grad=True, 
                                                device=self.device))
        self.results["forces"] = forces.detach().cpu().numpy()
        self.results["energy"] = energy.detach().cpu().numpy()


class HarmonicRestraint:
    """Class to apply a harmonic restraint on a MD simulations
    Params
    ------
    cvdic (dict): Dictionary contains the information to define the collective variables
                  and the harmonic restraint.
    max_steps (int): maximum number of steps of the MD simulation
    device: device
    """
    def __init__(self, cvdic, max_steps, device="cpu"):
        self.cvs = [] # list of collective variables (CV) objects
        self.kappas = []  # list of constant values for every CV
        self.eq_values = [] # list equilibrium values for every CV
        self.steps = []  # list of lists with the steps of the MD simulation
        self.device = device
        self.setup_contraint(cvdic, max_steps, device)

    def setup_contraint(self, cvdic, max_steps, device):
        """ Initializes the collectiva variables objects
        Args:
            cvdic (dict): Dictionary contains the information to define the collective variables
                          and the harmonic restraint.
            max_steps (int): maximum number of steps of the MD simulation
            device: device
        """
        for _, val in cvdic.items():
            cv = get_cv_from_dic(val, device)
            self.cvs.append(cv)
            steps, kappas, eq_values = self.create_time_dependec_arrays(val["restraint"], max_steps)
            self.kappas.append(kappas)
            self.steps.append(steps)
            self.eq_values.append(eq_values)

    def create_time_dependec_arrays(self, restraint_list, max_steps):
        """creates lists of steps, kappas and equilibrium values that will be used along
        the simulation to determine the value of kappa and equilibrium CV at each step

        Args:
            restraint_list (list of dicts): contains dictionaries with the desired values of
                                            kappa and CV at arbitrary step in the simulation
            max_steps (int): maximum number of steps of the simulation

        Returns:
            list: all steps e.g [1,2,3 .. max_steps]
            list: all kappas for every step, e.g [0.5, 0.5, 0.51, .. ] same size of steps
            list: all equilibrium values for every step, e.g. [1,1,1,3,3,3, ...], same size of steps
        """
        steps = []
        kappas = []
        eq_vals = []
        # in case the restraint does not start at 0
        templist = list(range(0, restraint_list[0]['step']))
        steps += templist
        kappas += [0 for _ in templist]
        eq_vals += [0 for _ in templist]

        for n, rd in enumerate(restraint_list[1:]):
            # rd and n are out of phase by 1, when n = 0, rd points to index 1
            templist = list(range(restraint_list[n]['step'], rd['step']))
            steps += templist
            kappas += [restraint_list[n]['kappa'] for _ in templist]
            dcv = rd['eq_val'] - restraint_list[n]['eq_val']
            cvstep = dcv/len(templist) # get step increase
            eq_vals += [restraint_list[n]['eq_val'] + cvstep * tind for tind, _ in enumerate(templist)]

        # in case the last step is lesser than the max_step
        templist = list(range(restraint_list[-1]['step'], max_steps))
        steps += templist
        kappas += [restraint_list[-1]['kappa'] for _ in templist]
        eq_vals += [restraint_list[-1]['eq_val'] for _ in templist]

        return steps, kappas, eq_vals

    def get_energy(self, positions, step):
        """ Calculates the retraint energy of an arbritrary state of the system
        Args:
            positions (torch.tensor): atomic positions
            step (int): current step
        Returns:
            float: energy
        """
        tot_energy = 0
        for n, cv in enumerate(self.cvs):
            kappa = self.kappas[n][step]
            eq_val = self.eq_values[n][step]
            cv_value = cv.get_value(positions)
            energy = 0.5 * kappa * (cv_value - eq_val) * (cv_value - eq_val)
            tot_energy += energy
        return tot_energy

    def get_bias(self, positions, step):
        """ Calculates the bias energy and force

        Args:
            positions (torch.tensor): atomic positions
            step (int): current step

        Returns:
            float: forces
            float: energy
        """
        if len(self.cvs) == 0:
            return torch.tensor(0), torch.tensor(0)
        energy = self.get_energy(positions, step)
        forces = -grad(energy, positions)[0]
        return forces, energy


