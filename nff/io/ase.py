import os, sys
import numpy as np
import torch

from ase import Atoms
from ase.neighborlist import neighbor_list
from ase.calculators.calculator import Calculator, all_changes
from ase import units
from ase.stress import voigt_6_to_full_3x3_stress, full_3x3_to_voigt_6_stress

import nff.utils.constants as const
from nff.nn.utils import torch_nbr_list #, numpy_nbr_list
from nff.utils.cuda import batch_to
from nff.data.sparse import sparsify_array
from nff.train.builders.model import load_model
from nff.utils.geom import compute_distances, batch_compute_distance
from nff.utils.scatter import compute_grad
from nff.data import Dataset
from nff.nn.graphop import split_and_sum

from nff.nn.models.schnet import SchNet, SchNetDiabat
from nff.nn.models.hybridgraph import HybridGraphConv
from nff.nn.models.schnet_features import SchNetFeatures
from nff.nn.models.cp3d import OnlyBondUpdateCP3D

from nff.utils.dispersion import clean_matrix, lattice_points_in_supercell
from nff.data import collate_dicts

from torch.autograd import grad

from .restraints import HarmonicRestraint, HarmonicRestraintStatic

DEFAULT_CUTOFF = 5.0
DEFAULT_DIRECTED = False
DEFAULT_SKIN = 1.0
UNDIRECTED = [SchNet,
              SchNetDiabat,
              HybridGraphConv,
              SchNetFeatures,
              OnlyBondUpdateCP3D]


def check_directed(model, atoms):
    model_cls = model.__class__.__name__
    msg = f"{model_cls} needs a directed neighbor list"
    assert atoms.directed, msg


class AtomsBatch(Atoms):
    """Class to deal with the Neural Force Field and batch several
       Atoms objects.
    """

    def __init__(
            self,
            *args,
            props=None,
            cutoff=DEFAULT_CUTOFF,
            directed=DEFAULT_DIRECTED,
            requires_large_offsets=False,
            cutoff_skin=DEFAULT_SKIN,
            spin=0,
            charge=0,
            device="cuda",
            **kwargs
    ):
        """

        Args:
            *args: Description
            nbr_list (None, optional): Description
            pbc_index (None, optional): Description
            cutoff (TYPE, optional): Description
            cutoff_skin (float): extra distance added to cutoff
                            to ensure we don't miss neighbors between nbr
                            list updates.
            **kwargs: Description
        """
        super().__init__(*args, **kwargs)

        if props is None:
            props = {}

        self.props = props
        self.nbr_list = props.get('nbr_list', None)
        self.offsets = props.get('offsets', None)
        self.directed = directed
        self.num_atoms = (props.get('num_atoms',
                                    torch.LongTensor([len(self)]))
                          .reshape(-1))
        self.props['num_atoms'] = self.num_atoms
        self.cutoff = cutoff
        self.cutoff_skin = cutoff_skin
        self.device = device
        self.requires_large_offsets = requires_large_offsets
        self.mol_nbrs, self.mol_idx = None, None
        self.spin = spin
        self.charge = charge
        
        # print("device is: ", self.device)
        if not torch.cuda.is_available():
            # print("changing device since cuda is not available")
            self.device = "cpu"

    def get_mol_nbrs(self, r_cut=95):
        """
        Dense directed neighbor list for each molecule, in case that's needed
        in the model calculation
        """

        # periodic systems
        if np.array([atoms.pbc.any() for atoms in self.get_list_atoms()]).any():
            nbrs = []
            nbrs_T = []
            nbrs = []
            z = []
            N = []
            lattice_points = []
            mask_applied = []
            _xyzs = []
            xyz_T = []
            num_atoms = []
            for atoms in self.get_list_atoms():
                nxyz = np.concatenate([
                            atoms.get_atomic_numbers().reshape(-1, 1),
                            atoms.get_positions().reshape(-1, 3)
                        ], axis=1)
                _xyz = torch.from_numpy(nxyz[:,1:])
                # only works if the cell for all crystals in batch are the same
                cell = atoms.get_cell()

                # cutoff specified by r_cut in Bohr (a.u.)
                # estimate getting close to the cutoff with supercell expansion
                a_mul = int(np.ceil(
                            (r_cut*const.BOHR_RADIUS) / np.linalg.norm(cell[0])
                                    ))
                b_mul = int(np.ceil(
                            (r_cut*const.BOHR_RADIUS) / np.linalg.norm(cell[1])
                                    ))
                c_mul = int(np.ceil(
                            (r_cut*const.BOHR_RADIUS) / np.linalg.norm(cell[2])
                                    ))
                supercell_matrix = np.array([[a_mul, 0, 0],
                                             [0, b_mul, 0],
                                             [0, 0, c_mul]])
                supercell = clean_matrix(supercell_matrix @ cell)

                # cartesian lattice points
                lattice_points_frac = lattice_points_in_supercell(supercell_matrix)
                _lattice_points = np.dot(lattice_points_frac, supercell)

                # need to get all negative lattice translation vectors
                # but remove duplicate 0 vector
                zero_idx = np.where(
                            np.all(_lattice_points.__eq__(np.array([0,0,0])),
                                                                        axis=1)
                                    )[0][0]
                _lattice_points = np.concatenate(
                                        [_lattice_points[zero_idx:, :],
                                         _lattice_points[:zero_idx, :]]
                                                )

                _z = torch.from_numpy(nxyz[:,0]).long().to(self.device)
                _N = len(_lattice_points)
                # perform lattice translations on positions
                lattice_points_T = (torch.tile(
                                        torch.from_numpy(_lattice_points),
                                    ( (len(_xyz),) +
                                        (1,)*(len(_lattice_points.shape)-1) )
                                            )/ const.BOHR_RADIUS).to(self.device)
                _xyz_T = ((torch.repeat_interleave(_xyz, _N, dim=0)
                                        / const.BOHR_RADIUS).to(self.device))
                _xyz_T = _xyz_T + lattice_points_T

                # get valid indices within the cutoff
                num = _xyz.shape[0]
                idx = torch.arange(num)
                x, y = torch.meshgrid(idx, idx, indexing='xy')
                _nbrs = torch.cat([x.reshape(-1, 1), y.reshape(-1, 1)],
                                                        dim=1).to(self.device)
                _lattice_points = (torch.tile(
                            torch.from_numpy(_lattice_points).to(self.device),
                                ( (len(_nbrs),) +
                                  (1,)*(len(_lattice_points.shape)-1) )
                                            ))

                # convert everything from Angstroms to Bohr
                _xyz = _xyz / const.BOHR_RADIUS
                _lattice_points = _lattice_points / const.BOHR_RADIUS

                _nbrs_T = torch.repeat_interleave(_nbrs, _N, dim=0).to(self.device)
                # ensure that A != B when T=0
                # since first index in _lattice_points corresponds to T=0
                # get the idxs on which to apply the mask
                idxs_to_apply = torch.tensor([True]*len(_nbrs_T)).to(self.device)
                idxs_to_apply[::_N] = False
                # get the mask that we want to apply
                mask = _nbrs_T[:,0] != _nbrs_T[:,1]
                # do a joint boolean operation to get the mask
                _mask_applied = torch.logical_or(idxs_to_apply, mask)
                _nbrs_T = _nbrs_T[_mask_applied]
                _lattice_points = _lattice_points[_mask_applied]

                nbrs_T.append(_nbrs_T)
                nbrs.append(_nbrs)
                z.append(_z)
                N.append(_N)
                lattice_points.append(_lattice_points)
                mask_applied.append(_mask_applied)
                xyz_T.append(_xyz_T)
                _xyzs.append(_xyz)

                num_atoms.append(len(_xyz))

            nbrs_info = (nbrs_T, nbrs, z, N, lattice_points, mask_applied)

            mol_idx = torch.cat([torch.zeros(num) + i
                                for i, num in enumerate(num_atoms)]
                                ).long()

            return nbrs_info, mol_idx

        # non-periodic systems
        else:
            counter = 0
            nbrs = []

            for atoms in self.get_list_atoms():
                nxyz = np.concatenate([
                    atoms.get_atomic_numbers().reshape(-1, 1),
                    atoms.get_positions().reshape(-1, 3)
                ], axis=1)

                n = nxyz.shape[0]
                idx = torch.arange(n)
                x, y = torch.meshgrid(idx, idx, indexing='xy')

                # undirected neighbor list
                these_nbrs = torch.cat([x.reshape(-1, 1), y.reshape(-1, 1)], dim=1)
                these_nbrs = these_nbrs[these_nbrs[:, 0] != these_nbrs[:, 1]]

                nbrs.append(these_nbrs + counter)
                counter += n

            nbrs = torch.cat(nbrs)
            mol_idx = torch.cat([torch.zeros(num) + i
                                for i, num in enumerate(self.num_atoms)]
                                ).long()

            return nbrs, mol_idx

    def get_nxyz(self):
        """Gets the atomic number and the positions of the atoms
           inside the unit cell of the system.
        Returns:
            nxyz (np.array): atomic numbers + cartesian coordinates
                             of the atoms.
        """
        nxyz = np.concatenate([
            self.get_atomic_numbers().reshape(-1, 1),
            self.get_positions().reshape(-1, 3)
        ], axis=1)

        return nxyz

    def get_batch(self):
        """Uses the properties of Atoms to create a batch
           to be sent to the model.
           Returns:
              batch (dict): batch with the keys 'nxyz',
                            'num_atoms', 'nbr_list' and 'offsets'
        """

        if "mol_nbrs" not in self.props:
            self.cutoff = torch.inf
            self.update_nbr_list()
            self.props['nbr_list'] = self.nbr_list
            self.props['offsets'] = self.offsets

        if self.nbr_list is None or self.offsets is None:
            self.update_nbr_list()

        self.props['nbr_list'] = self.nbr_list
        self.props['offsets'] = self.offsets
        if self.pbc.any():
            self.props['cell'] = torch.Tensor(np.array(self.cell))

        self.props['nxyz'] = torch.Tensor(self.get_nxyz())
        if self.props.get('num_atoms') is None:
            self.props['num_atoms'] = torch.LongTensor([len(self)])

        if self.mol_nbrs is not None:
            self.props['mol_nbrs'] = self.mol_nbrs

        if self.mol_idx is not None:
            self.props['mol_idx'] = self.mol_idx
        
        self.props["charge"] =  torch.Tensor([self.charge])
        self.props["spin"] =  torch.Tensor([self.spin])

        return self.props

    def get_list_atoms(self):
        #print("Getting list atoms ", len(self), self.directed, self.props.get('num_atoms')) # Reisel
        self.directed = True # Reisel This is to avoid the error with Plumed calculator !!
        if not self.props.get('num_atoms'):
            self.props['num_atoms'] = torch.LongTensor([len(self)])
            #self.props['num_atoms'] = torch.LongTensor([len(self.get_positions())])

        mol_split_idx = self.props['num_atoms'].tolist()

        positions = torch.Tensor(self.get_positions())
        Z = torch.LongTensor(self.get_atomic_numbers())
        
        #print(positions.shape, mol_split_idx, self.props["num_atoms"])
        positions = list(positions.split(mol_split_idx))
        Z = list(Z.split(mol_split_idx))
        masses = list(torch.Tensor(self.get_masses())
                      .split(mol_split_idx))

        Atoms_list = []

        for i, molecule_xyz in enumerate(positions):
            atoms = Atoms(Z[i].tolist(),
                          molecule_xyz.numpy(),
                          cell=self.cell,
                          pbc=self.pbc)

            # in case you artificially changed the masses
            # of any of the atoms
            atoms.set_masses(masses[i])

            Atoms_list.append(atoms)

        return Atoms_list

    def update_nbr_list(self):
        """Update neighbor list and the periodic reindexing
           for the given Atoms object.
           Args:
           cutoff(float): maximum cutoff for which atoms are
                                          considered interacting.
           Returns:
           nbr_list(torch.LongTensor)
           offsets(torch.Tensor)
           nxyz(torch.Tensor)
        """

        Atoms_list = self.get_list_atoms()

        ensemble_nbr_list = []
        ensemble_offsets_list = []

        for i, atoms in enumerate(Atoms_list):
            edge_from, edge_to, offsets = torch_nbr_list(
                atoms,
                (self.cutoff + self.cutoff_skin),
                device=self.device,
                directed=self.directed,
                requires_large_offsets=self.requires_large_offsets)
            # print(not np.any(edge_from == tedge_from))
            # quit()
            nbr_list = torch.LongTensor(np.stack([edge_from, edge_to], axis=1))
            these_offsets = sparsify_array(offsets.dot(self.get_cell()))

            # non-periodic
            if isinstance(these_offsets, int):
                these_offsets = torch.Tensor(offsets)

            ensemble_nbr_list.append(
                self.props['num_atoms'][: i].sum() + nbr_list)
            ensemble_offsets_list.append(these_offsets)

        ensemble_nbr_list = torch.cat(ensemble_nbr_list)

        if all([isinstance(i, int) for i in ensemble_offsets_list]):
            ensemble_offsets_list = torch.Tensor(ensemble_offsets_list)
        else:
            ensemble_offsets_list = torch.cat(ensemble_offsets_list)

        self.nbr_list = ensemble_nbr_list
        self.offsets = ensemble_offsets_list

        return ensemble_nbr_list, ensemble_offsets_list

    def get_batch_energies(self):

        if self._calc is None:
            raise RuntimeError('Atoms object has no calculator.')

        if not hasattr(self._calc, 'get_potential_energies'):
            raise RuntimeError(
                'The calculator for atomwise energies is not implemented')

        energies = self.get_potential_energies()

        batched_energies = split_and_sum(torch.Tensor(energies),
                                         self.props['num_atoms'].tolist())

        return batched_energies.detach().cpu().numpy()

    def get_batch_kinetic_energy(self):

        if self.get_momenta().any():
            atomwise_ke = torch.Tensor(
                0.5 * self.get_momenta() * self.get_velocities()).sum(-1)
            batch_ke = split_and_sum(
                atomwise_ke, self.props['num_atoms'].tolist())
            return batch_ke.detach().cpu().numpy()

        else:
            print("No momenta are set for atoms")

    def get_batch_T(self):

        T = (self.get_batch_kinetic_energy() /
             (1.5 * units.kB * self.props['num_atoms']
              .detach().cpu().numpy()))
        return T

    def batch_properties():
        pass

    def batch_virial():
        pass

    @classmethod
    def from_atoms(cls, atoms, **kwargs):
        return cls(
            atoms,
            positions=atoms.positions,
            numbers=atoms.numbers,
            props={},
            **kwargs
        )


class BulkPhaseMaterials(Atoms):
    """Class to deal with the Neural Force Field and batch molecules together
    in a box for handling boxphase.
    """

    def __init__(
            self,
            *args,
            props={},
            cutoff=DEFAULT_CUTOFF,
            nbr_torch=False,
            device='cpu',
            directed=DEFAULT_DIRECTED,
            **kwargs
    ):
        """

        Args:
        *args: Description
        nbr_list (None, optional): Description
        pbc_index (None, optional): Description
        cutoff (TYPE, optional): Description
        **kwargs: Description
        """
        super().__init__(*args, **kwargs)

        self.props = props
        self.nbr_list = self.props.get('nbr_list', None)
        self.offsets = self.props.get('offsets', None)
        self.num_atoms = self.props.get('num_atoms', None)
        self.cutoff = cutoff
        self.nbr_torch = nbr_torch
        self.device = device
        self.directed = directed

    def get_nxyz(self):
        """Gets the atomic number and the positions of the atoms
           inside the unit cell of the system.
        Returns:
                nxyz (np.array): atomic numbers + cartesian coordinates
                                                 of the atoms.
        """
        nxyz = np.concatenate([
            self.get_atomic_numbers().reshape(-1, 1),
            self.get_positions().reshape(-1, 3)
        ], axis=1)

        return nxyz

    def get_batch(self):
        """Uses the properties of Atoms to create a batch
           to be sent to the model.

        Returns:
           batch (dict): batch with the keys 'nxyz',
           'num_atoms', 'nbr_list' and 'offsets'
        """

        if self.nbr_list is None or self.offsets is None:
            self.update_nbr_list()
            self.props['nbr_list'] = self.nbr_list
            self.props['atoms_nbr_list'] = self.atoms_nbr_list
            self.props['offsets'] = self.offsets

        self.props['nbr_list'] = self.nbr_list
        self.props['atoms_nbr_list'] = self.atoms_nbr_list
        self.props['offsets'] = self.offsets
        self.props['nxyz'] = torch.Tensor(self.get_nxyz())

        return self.props

    def update_system_nbr_list(self, cutoff, exclude_atoms_nbr_list=True):
        """Update undirected neighbor list and the periodic reindexing
           for the given Atoms object.

           Args:
           cutoff (float): maximum cutoff for which atoms are
           considered interacting.

           Returns:
           nbr_list (torch.LongTensor)
           offsets (torch.Tensor)
                nxyz (torch.Tensor)
        """

        if self.nbr_torch:
            edge_from, edge_to, offsets = torch_nbr_list(
                self, self.cutoff, device=self.device)
            nbr_list = torch.LongTensor(np.stack([edge_from, edge_to], axis=1))
        else:
            edge_from, edge_to, offsets = neighbor_list(
                'ijS',
                self,
                self.cutoff)
            nbr_list = torch.LongTensor(np.stack([edge_from, edge_to], axis=1))
            if not getattr(self, "directed", DEFAULT_DIRECTED):
                offsets = offsets[nbr_list[:, 1] > nbr_list[:, 0]]
                nbr_list = nbr_list[nbr_list[:, 1] > nbr_list[:, 0]]

        if exclude_atoms_nbr_list:
            offsets_mat = torch.zeros(len(self),
                                      len(self), 3)
            nbr_list_mat = (torch.zeros(len(self),
                                        len(self))
                            .to(torch.long))
            atom_nbr_list_mat = (torch.zeros(len(self),
                                             len(self))
                                 .to(torch.long))

            offsets_mat[nbr_list[:, 0], nbr_list[:, 1]] = offsets
            nbr_list_mat[nbr_list[:, 0], nbr_list[:, 1]] = 1
            atom_nbr_list_mat[self.atoms_nbr_list[:, 0],
                              self.atoms_nbr_list[:, 1]] = 1

            nbr_list_mat = nbr_list_mat - atom_nbr_list_mat
            nbr_list = nbr_list_mat.nonzero()
            offsets = offsets_mat[nbr_list[:, 0], nbr_list[:, 1], :]

        self.nbr_list = nbr_list
        self.offsets = sparsify_array(
            offsets.matmul(torch.Tensor(self.get_cell())))

    def get_list_atoms(self):

        mol_split_idx = self.props['num_subgraphs'].tolist()

        positions = torch.Tensor(self.get_positions())
        Z = torch.LongTensor(self.get_atomic_numbers())

        positions = list(positions.split(mol_split_idx))
        Z = list(Z.split(mol_split_idx))

        Atoms_list = []

        for i, molecule_xyz in enumerate(positions):
            Atoms_list.append(Atoms(Z[i].tolist(),
                                    molecule_xyz.numpy(),
                                    cell=self.cell,
                                    pbc=self.pbc))

        return Atoms_list

    def update_atoms_nbr_list(self, cutoff):

        Atoms_list = self.get_list_atoms()

        intra_nbr_list = []
        for i, atoms in enumerate(Atoms_list):
            edge_from, edge_to = neighbor_list('ij', atoms, cutoff)
            nbr_list = torch.LongTensor(np.stack([edge_from, edge_to], axis=1))

            if not self.directed:
                nbr_list = nbr_list[nbr_list[:, 1] > nbr_list[:, 0]]

            intra_nbr_list.append(
                self.props['num_subgraphs'][: i].sum() + nbr_list)

        intra_nbr_list = torch.cat(intra_nbr_list)
        self.atoms_nbr_list = intra_nbr_list

    def update_nbr_list(self):
        self.update_atoms_nbr_list(self.props['atoms_cutoff'])
        self.update_system_nbr_list(self.props['system_cutoff'])


class CHGNetCalc(Calculator):
    implemented_properties = ["energy", "forces"]
    def __init__(self, model, device="cpu", properties=["energy", "forces"] ,**kwargs):
        from pymatgen.io.ase import AseAtomsAdaptor
        self.aseconv = AseAtomsAdaptor
        self.model = model
        self.device = device
        self.properties = properties
        Calculator.__init__(self, **kwargs)

        # self.model.to(device)

    def calculate(self, atoms, properties=["energy", "forces"], all_changes=all_changes):
        if getattr(self, "properties", None) is None:
            self.properties = properties

        #Calculator.calculate(self, atoms, self.properties, all_changes)

        # get_batch takes care of update_nbr_list if it is not set
        # the Dynamics class takes care of this in md runs  
        struct = self.aseconv.get_structure(atoms)
        prediction = self.model.predict_structure(struct)

        # energy is per atom
        self.results["energy"] = prediction["e"] * atoms.get_global_number_of_atoms() #.detach().cpu().numpy() #* (1 / const.EV_TO_KCAL_MOL)#.reshape(-1)
        self.results["forces"] = prediction["f"]#.detach().cpu().numpy() #* (1 / const.EV_TO_KCAL_MOL)#.reshape(-1, 3)

    def get_en_forces(self, frame, **args):
        # interface for tigre
        atoms = Atoms(symbols=frame.symbols, positions=frame.coords, cell=frame.lattice)
        atoms = AtomsBatch(
                    atoms,
                    cutoff=5.0,
                    cutoff_skin=1.0,
                    directed=True,
                    device=self.device
                )
        self.calculate(atoms)
        forces = self.results["forces"]
        energy = self.results["energy"]
        return energy, forces


class CP2K(Calculator):
    implemented_properties = ["energy", "forces"]
    def __init__(self, nprocs=32, properties=["energy", "forces"] ,**kwargs):
        self.properties = properties
        self.calc = None
        self.nprocs = nprocs
        self.rootdir = os.getcwd()
        self.calcdir = os.path.join(self.rootdir, "cp2kwd")
        self.setup()
        Calculator.__init__(self, **kwargs)

    def get_en_forces(self, output_file, natoms):
        en = None
        forces = []

        normal_end = False
        with open(output_file) as f:
            for line in f:
                if "- Atoms:" in line:
                    natoms = int(line.split()[-1])
                
                if 'ENERGY|' in line:
                    en = float(line.split()[-1])
            
                if line.startswith(" ATOMIC FORCES"):
                    forces = []
                    f.readline(); f.readline()  # skip the two next lines
                    for _ in range(natoms):
                        fline = f.readline()
                        forcei = [float(i) for i in fline.split()[-3:]]
                        forces.append(forcei)
                if "PROGRAM ENDED AT" in line:
                    normal_end = True
        # to eV
        forces = np.array(forces)
        return en, forces

    def sub_Po_for_Cu(self, atoms):
        nas = [i if i != 84 else 29 for i in atoms.get_atomic_numbers() ]
        atoms.set_atomic_numbers(nas)
        return atoms

    def setup(self):
        from pycp2k import CP2K
        self.calc = CP2K()
        self.calc.mpi_on=True
        self.calc.mpi_command = "module purge && module load cesga/2022 gcc/system openmpi/4.1.4 cp2k/2023.1 ; mpirun"
        self.calc.mpi_n_processes = self.nprocs

        if not os.path.isdir(self.calcdir):
            os.mkdir(self.calcdir)
        os.chdir(self.calcdir)
        
        self.calc.working_directory = "./"
        self.calc.project_name = "scf_"

        #================= An existing input file can be parsed  =======================
        self.calc.parse("cp2k.inp")
        os.chdir(self.rootdir)


    def calculate(self, atoms, properties=["energy", "forces"], all_changes=all_changes):
        if getattr(self, "properties", None) is None:
            self.properties = properties
        # print("DFT calculate called")
        # update the substitution
        atoms = self.sub_Po_for_Cu(atoms)

        #==================== Define shortcuts for easy access =========================
        os.chdir(self.calcdir)
        CP2K_INPUT = self.calc.CP2K_INPUT
        # GLOBAL = CP2K_INPUT.GLOBAL
        FORCE_EVAL = CP2K_INPUT.FORCE_EVAL_list[-1]  # Repeatable items have to be first created
        SUBSYS = FORCE_EVAL.SUBSYS
        self.calc.create_cell(SUBSYS, atoms)
        self.calc.create_coord(SUBSYS, atoms)

        #============ Run the simulation or just write the input file ================
        self.calc.write_input_file()
        self.calc.run()
        en, forces = self.get_en_forces("scf_.out", len(atoms))
        # print(en, forces)
        os.chdir(self.rootdir)

        self.results["energy"] = en * const.AU_TO_EV
        self.results["forces"] = forces * const.AU_TO_EV / const.BOHR_RADIUS 


class TorchCalc(Calculator):
    implemented_properties = ["energy", "forces"]
    def __init__(self, model, device="cpu", properties=["energy", "forces"] ,**kwargs):
        self.model = model
        self.model.eval()
        self.device = device
        self.properties = properties
        Calculator.__init__(self, **kwargs)

        self.model.to(device)


    def set_props(self, atoms, results):
        if "charge" in results:
            for atom, ch in zip(atoms, results["charge"]):
                atom.charge = ch
        return atoms

    def calculate(self, atoms, properties=["energy", "forces"], all_changes=all_changes):
        if getattr(self, "properties", None) is None:
            self.properties = properties

        #Calculator.calculate(self, atoms, self.properties, all_changes)

        # get_batch takes care of update_nbr_list if it is not set
        # the Dynamics class takes care of this in md runs
        # print("TORCHCALC len(atoms) in calculated func ", len(atoms), type(atoms), atoms.directed)
        #atoms.directed = True
        batch = batch_to(atoms.get_batch(), self.device)

        prediction = self.model(batch)
        self.results["energy"] = prediction["energy"].detach().cpu().numpy() * (1 / const.EV_TO_KCAL_MOL)#.reshape(-1)
        self.results["forces"] = -prediction["energy_grad"].detach().cpu().numpy() * (1 / const.EV_TO_KCAL_MOL)#.reshape(-1, 3)
        
        for k in prediction:
            if k in ["energy", "energy_grad"]: continue 
            self.results[k] = prediction[k].detach().cpu().numpy()

        self.set_props(atoms, self.results)

    def get_en_forces(self, frame, **args):
        # interface for tigre
        atoms = Atoms(symbols=frame.symbols, positions=frame.coords, cell=frame.lattice)
        atoms = AtomsBatch(
                    atoms,
                    cutoff=5.0,
                    cutoff_skin=1.0,
                    directed=True,
                    device=self.device
                )
        self.calculate(atoms)
        forces = self.results["forces"]
        energy = self.results["energy"]
        return energy, forces


class TorchNeuralRestraint(TorchCalc):
    implemented_properties = ["energy", "forces"]
    def __init__(self, model, cvs, max_steps, device="cpu", properties=["energy", "forces"],**kwargs):
        TorchCalc.__init__(self, model, device, properties=properties, **kwargs)
        self.hr = HarmonicRestraint(cvs, max_steps, device)
        self.step = -1
        self.bias_energy = 0

    def calculate(self, atoms, properties=["energy", "forces"], all_changes=all_changes):
        TorchCalc.calculate(self, atoms, properties, all_changes=all_changes)
        #try:
        bias_forces, bias_energy = self.hr.get_bias(torch.tensor(atoms.get_positions(), requires_grad=True, device=self.device), self.step)         
        #except:
        #    print("error in step : ", self.step)
        #    quit()
        self.results["energy"] += bias_energy.detach().cpu().numpy()
        self.results["forces"] += bias_forces.detach().cpu().numpy() 

        self.bias_energy = bias_energy

    def write(self, atoms):
        self.step += 1

        with open("colvar", "a") as f:
            # f.write("{} ".format(self.step*0.5))
            f.write("{} ".format(self.step))
            # ARREGLAR, SI YA ESTA CALCULADO PARA QUE RECALCULAR LA CVS
            for cv in self.hr.cvs:
                curr_cv_val = float(cv.get_value(torch.tensor(atoms.get_positions(), device=self.device)))
                f.write(" {:.6f} ".format(curr_cv_val))
            f.write("{:.6f} \n".format(float(self.bias_energy)))

    def get_en_forces(self, frame, **args):
        # interface for tigre
        atoms = Atoms(symbols=frame.symbols, positions=frame.coords, cell=frame.lattice)
        atoms = AtomsBatch(
                    atoms,
                    cutoff=5.0,
                    cutoff_skin=1.0,
                    directed=True,
                    device=self.device
                )
        self.calculate(atoms)
        forces = self.results["forces"]
        energy = self.results["energy"]
        return energy, forces


class TorchRestraintStatic(TorchCalc):
    implemented_properties = ["energy", "forces"]
    def __init__(self, cvdic, model, device="cpu", properties=["energy", "forces"],**kwargs):
        TorchCalc.__init__(self, model, device, properties=properties, **kwargs)
        self.hr =  HarmonicRestraintStatic(cvdic, device)

    def calculate(self, atoms, properties=["energy", "forces"], all_changes=all_changes):
        if np.any(np.isnan(atoms.positions)):
            print("System exploded. Stoping")
            sys.exit(1)

        TorchCalc.calculate(self, atoms, properties, all_changes=all_changes)
        bias_forces, bias_energy = self.hr.get_bias(torch.tensor(atoms.get_positions(), requires_grad=True, device=self.device))         
        self.results["energy"] += bias_energy.detach().cpu().numpy()
        self.results["forces"] += bias_forces.detach().cpu().numpy() 

    def get_en_forces(self, frame, step):
        atoms = frame.as_ase_atoms()
        self.calculate(atoms, step)
        return self.results["energy"], self.results["forces"]


class NeuralFF(Calculator):
    """ASE calculator using a pretrained NeuralFF model"""

    implemented_properties = ['energy', 'forces', 'stress']

    def __init__(
            self,
            model,
            device='cpu',
            en_key='energy',
            properties=['energy', 'forces'],
            model_kwargs=None,
            **kwargs
    ):
        """Creates a NeuralFF calculator.nff/io/ase.py

        Args:
        model (TYPE): Description
        device (str): device on which the calculations will be performed
        properties (list of str): 'energy', 'forces' or both and also stress for only
        schnet  and painn
        **kwargs: Description
        model (one of nff.nn.models)
        """

        Calculator.__init__(self, **kwargs)
        self.model = model
        self.model.eval()
        self.device = device
        self.to(device)
        self.en_key = en_key
        self.properties = properties
        self.model_kwargs = model_kwargs

    def to(self, device):
        self.device = device
        self.model.to(device)

    def calculate(
            self,
            atoms=None,
            properties=['energy', 'forces'],
            system_changes=all_changes,
    ):
        """Calculates the desired properties for the given AtomsBatch.

        Args:
        atoms (AtomsBatch): custom Atoms subclass that contains implementation
            of neighbor lists, batching and so on. Avoids the use of the Dataset
            to calculate using the models created.
        system_changes (default from ase)
        """

        if not any([isinstance(self.model, i) for i in UNDIRECTED]):
            check_directed(self.model, atoms)

        # for backwards compatability
        if getattr(self, "properties", None) is None:
            self.properties = properties

        Calculator.calculate(self, atoms, self.properties, system_changes)

        # run model
        # atomsbatch = AtomsBatch(atoms)
        # batch_to(atomsbatch.get_batch(), self.device)
        batch = batch_to(atoms.get_batch(), self.device)
        print(batch)
        # add keys so that the readout function can calculate these properties
        grad_key = self.en_key + "_grad"
        batch[self.en_key] = []
        batch[grad_key] = []

        kwargs = {}
        requires_stress = "stress" in self.properties
        if requires_stress:
            kwargs["requires_stress"] = True
        kwargs["grimme_disp"] = False
        if getattr(self, "model_kwargs", None) is not None:
            kwargs.update(self.model_kwargs)

        prediction = self.model(batch, **kwargs)

        # change energy and force to numpy array

        energy = (prediction[self.en_key].detach()
                  .cpu().numpy() * (1 / const.EV_TO_KCAL_MOL))

        if grad_key in prediction:
            energy_grad = (prediction[grad_key].detach()
                           .cpu().numpy() * (1 / const.EV_TO_KCAL_MOL))

        self.results = {
            'energy': energy.reshape(-1)
        }
        if 'e_disp' in prediction:
            self.results['energy'] = self.results['energy'] + prediction['e_disp']

        if 'forces' in self.properties:
            self.results['forces'] = -energy_grad.reshape(-1, 3)
            if 'forces_disp' in prediction:
                self.results['forces'] = self.results['forces'] + prediction['forces_disp']

        if requires_stress:
            stress = (prediction['stress_volume'].detach()
                      .cpu().numpy() * (1 / const.EV_TO_KCAL_MOL))
            self.results['stress'] = stress * (1 / atoms.get_volume())
            if 'stress_disp' in prediction:
                self.results['stress'] = self.results['stress'] + prediction['stress_disp']
            self.results['stress'] = full_3x3_to_voigt_6_stress(self.results['stress'])

    @classmethod
    def from_file(
            cls,
            model_path,
            device='cuda',
            **kwargs
    ):

        model = load_model(model_path, **kwargs)
        out = cls(model=model,
                  device=device,
                  **kwargs)
        return out


class EnsembleNFF(Calculator):
    """Produces an ensemble of NFF calculators to predict the
       discrepancy between the properties"""
    implemented_properties = ['energy', 'forces', 'stress']

    def __init__(
            self,
            models: list,
            device='cpu',
            jobdir=None,
            properties=['energy', 'forces'],
            model_kwargs=None,
            **kwargs
    ):
        """Creates a NeuralFF calculator.nff/io/ase.py

        Args:
        model(TYPE): Description
        device(str): device on which the calculations will be performed
        **kwargs: Description
        model(one of nff.nn.models)

        """

        Calculator.__init__(self, **kwargs)
        self.models = models
        for m in self.models:
            m.eval()
        self.device = device
        self.jobdir = jobdir
        self.to(device)
        self.properties = properties
        self.model_kwargs = model_kwargs

    def to(self, device):
        self.device = device
        for m in self.models:
            m.to(device)

    def log_ensemble(self,
                     jobdir,
                     log_filename,
                     props
    ):
        """For the purposes of logging the std on-the-fly, to help with
        sampling after calling NFF on geometries with high uncertainty."""

        log_file = os.path.join(jobdir, log_filename)
        if os.path.exists(log_file):
            log = np.load(log_file)
        else:
            log = None

        if log is not None:
            log = np.concatenate([log, props], axis=0)
        else:
            log = props

        np.save(log_filename, log)

        return

    def calculate(
            self,
            atoms=None,
            properties=['energy', 'forces'],
            system_changes=all_changes,
    ):
        """Calculates the desired properties for the given AtomsBatch.

        Args:
        atoms (AtomsBatch): custom Atoms subclass that contains implementation
            of neighbor lists, batching and so on. Avoids the use of the Dataset
            to calculate using the models created.
        properties (list of str): 'energy', 'forces' or both
        system_changes (default from ase)
        """
        # print("calculate called")

        for model in self.models:
            if not any([isinstance(model, i) for i in UNDIRECTED]):
                check_directed(model, atoms)

        if getattr(self, "properties", None) is None:
            self.properties = properties

        Calculator.calculate(self, atoms, properties, system_changes)

        kwargs = {}
        requires_stress = "stress" in self.properties
        if requires_stress:
            kwargs["requires_stress"] = True
        kwargs["grimme_disp"] = False
        if getattr(self, "model_kwargs", None) is not None:
            kwargs.update(self.model_kwargs)

        # run model
        # atomsbatch = AtomsBatch(atoms)
        # batch_to(atomsbatch.get_batch(), self.device)
        batch = batch_to(atoms.get_batch(), self.device)

        # add keys so that the readout function can calculate these properties
        batch['energy'] = []
        if 'forces' in properties:
            batch['energy_grad'] = []

        energies = []
        gradients = []
        stresses = []
        for model in self.models:
            prediction = model(batch, **kwargs)

            # change energy and force to numpy array
            energies.append(
                prediction['energy']
                    .detach()
                    .cpu()
                    .numpy()
                * (1 / const.EV_TO_KCAL_MOL)
            )
            gradients.append(
                prediction['energy_grad']
                .detach()
                .cpu()
                .numpy()
                * (1 / const.EV_TO_KCAL_MOL)
            )
            if 'stress_volume' in prediction:
                stresses.append(
                    prediction['stress_volume']
                    .detach()
                    .cpu()
                    .numpy()
                     * (1 / const.EV_TO_KCAL_MOL)
                    * (1 / atoms.get_volume())
                )

        energies = np.stack(energies)
        gradients = np.stack(gradients)
        if len(stresses) > 0:
            stresses = np.stack(stresses)

        self.results = {
            'energy': energies.mean(0).reshape(-1),
            'energy_std': energies.std(0).reshape(-1),
        }
        if 'e_disp' in prediction:
            self.results['energy'] = self.results['energy'] + prediction['e_disp']
        if self.jobdir is not None and system_changes:
            energy_std = self.results['energy_std'][None]
            self.log_ensemble(self.jobdir,'energy_nff_ensemble.npy',energies)

        if 'forces' in properties:
            self.results['forces'] = -gradients.mean(0).reshape(-1, 3)
            self.results['forces_std'] = gradients.std(0).reshape(-1, 3)
            if 'forces_disp' in prediction:
                self.results['forces'] = self.results['forces'] + prediction['forces_disp']
            if self.jobdir is not None:
                forces_std = self.results['forces_std'][None,:,:]
                self.log_ensemble(self.jobdir,'forces_nff_ensemble.npy',-gradients)

        if 'stress' in properties:
            self.results['stress'] = stresses.mean(0)
            self.results['stress_std'] = stresses.std(0)
            if 'stress_disp' in prediction:
                self.results['stress'] = self.results['stress'] + prediction['stress_disp']
            if self.jobdir is not None:
                stress_std = self.results['stress_std'][None,:,:]
                self.log_ensemble(self.jobdir,'stress_nff_ensemble.npy',stresses)

        atoms.results = self.results.copy()

    @classmethod
    def from_files(
            cls,
            model_paths: list,
            device='cuda',
            **kwargs
    ):
        models = [
            load_model(path)
            for path in model_paths
        ]
        return cls(models, device, **kwargs)


class TorchNeuralRestraintEnsemble(TorchNeuralRestraint, EnsembleNFF):
    implemented_properties = ["energy", "forces"]
    def __init__(self, models, cv, max_steps, device="cpu", properties=["energy", "forces"], **kwargs):
        EnsembleNFF.__init__(self, models, device, properties=properties, **kwargs)
        # TorchNeuralRestraint.__init__(self, cv, max_steps, models[0], device)
        self.hr = HarmonicRestraint(cv, max_steps, device)
        self.step = -1
        self.max_steps = max_steps
        self.bias_energy = 0

    def calculate(self, atoms, properties=["energy", "forces"], all_changes=all_changes):
        EnsembleNFF.calculate(self, atoms, properties)
        bias_forces, bias_energy = self.hr.get_bias(torch.tensor(atoms.get_positions(), requires_grad=True, device=self.device), self.step)         
        self.results["energy"] += bias_energy.detach().cpu().numpy()
        self.results["forces"] += bias_forces.detach().cpu().numpy() 

        self.bias_energy = bias_energy
        print("setting resutls")
        atoms.results = self.results.copy()


class NeuralOptimizer:
    def __init__(
            self,
            optimizer,
            nbrlist_update_freq=5
    ):
        self.optimizer = optimizer
        self.update_freq = nbrlist_update_freq

    def run(self, fmax=0.2, steps=1000):
        epochs = steps // self.update_freq

        for step in range(epochs):
            self.optimizer.run(fmax=fmax, steps=self.update_freq)
            self.optimizer.atoms.update_nbr_list()


class NeuralMetadynamics(NeuralFF):

    def __init__(self,
                 model,
                 pushing_params,
                 old_atoms=None,
                 device='cpu',
                 en_key='energy',
                 directed=DEFAULT_DIRECTED,
                 **kwargs):

        NeuralFF.__init__(self,
                          model=model,
                          device=device,
                          en_key=en_key,
                          directed=DEFAULT_DIRECTED,
                          **kwargs)

        self.pushing_params = pushing_params
        self.old_atoms = old_atoms if (old_atoms is not None) else []
        self.steps_from_old = []

        # only apply the bias to certain atoms
        self.exclude_atoms = torch.LongTensor(self.pushing_params
                                              .get("exclude_atoms", []))
        self.keep_idx = None

    def get_keep_idx(self, atoms):
        # correct for atoms not in the biasing potential

        if self.keep_idx is not None:
            assert len(self.keep_idx) + len(self.exclude_atoms) == len(atoms)
            return self.keep_idx

        keep_idx = torch.LongTensor([i for i in range(len(atoms))
                                     if i not in self.exclude_atoms])
        self.keep_idx = keep_idx
        return keep_idx

    def make_dsets(self,
                   atoms):

        keep_idx = self.get_keep_idx(atoms)
        # put the current atoms as the second dataset because that's the one
        # that gets its positions rotated and its gradient computed
        props_1 = {"nxyz": [torch.Tensor(atoms.get_nxyz())
                            [keep_idx, :]]}
        props_0 = {"nxyz": [torch.Tensor(old_atoms.get_nxyz())[keep_idx, :]
                            for old_atoms in self.old_atoms]}

        dset_0 = Dataset(props_0, do_copy=False)
        dset_1 = Dataset(props_1, do_copy=False)

        return dset_0, dset_1

    def rmsd_prelims(self, atoms):

        num_atoms = len(atoms)

        # f_damp is the turn-on on timescale, measured in
        # number of steps. From https://pubs.acs.org/doi/pdf/
        # 10.1021/acs.jctc.9b00143

        kappa = self.pushing_params["kappa"]
        steps_from_old = torch.Tensor(self.steps_from_old)
        f_damp = (2 / (1 + torch.exp(-kappa * steps_from_old)) -
                  1)

        # given in mHartree / atom in CREST paper
        k_i = ((self.pushing_params['k_i'] / 1000 *
                units.Hartree * num_atoms))

        # given in Bohr^(-2) in CREST paper
        alpha_i = ((self.pushing_params['alpha_i'] /
                    units.Bohr ** 2))

        dsets = self.make_dsets(atoms)

        return k_i, alpha_i, dsets, f_damp

    def rmsd_push(self, atoms):

        if not self.old_atoms:
            return np.zeros((len(atoms), 3)), 0.0

        k_i, alpha_i, dsets, f_damp = self.rmsd_prelims(atoms)

        delta_i, _, xyz_list = compute_distances(
            dataset=dsets[0],
            # do this on CPU - it's a small RMSD
            # and gradient calculation, so the
            # dominant time is data transfer to GPU.
            # Testing it out confirms that you get a
            # big slowdown from doing it on GPU
            device='cpu',
            # device=self.device,
            dataset_1=dsets[1],
            store_grad=True,
            collate_dicts=collate_dicts)

        v_bias = (f_damp * k_i * torch.exp(-alpha_i * delta_i.reshape(-1) ** 2)
                  ).sum()

        f_bias = -compute_grad(inputs=xyz_list[0],
                               output=v_bias).sum(0)

        keep_idx = self.get_keep_idx(atoms)
        final_f_bias = torch.zeros(len(atoms), 3)
        final_f_bias[keep_idx] = f_bias.detach().cpu()
        nan_idx = torch.bitwise_not(torch.isfinite(final_f_bias))
        final_f_bias[nan_idx] = 0

        return final_f_bias.detach().numpy(), v_bias.detach().numpy()

    def get_bias(self, atoms):
        bias_type = self.pushing_params['bias_type']
        if bias_type == "rmsd":
            results = self.rmsd_push(atoms)
        else:
            raise NotImplementedError

        return results

    def append_atoms(self, atoms):
        self.old_atoms.append(atoms)
        self.steps_from_old.append(0)

        max_ref = self.pushing_params.get("max_ref_structures")
        if max_ref is None:
            max_ref = 10

        if len(self.old_atoms) >= max_ref:
            self.old_atoms = self.old_atoms[-max_ref:]
            self.steps_from_old = self.steps_from_old[-max_ref:]

    def calculate(self,
                  atoms,
                  properties=['energy', 'forces'],
                  system_changes=all_changes,
                  add_steps=True):

        if not any([isinstance(self.model, i) for i in UNDIRECTED]):
            check_directed(self.model, atoms)

        super().calculate(atoms=atoms,
                          properties=properties,
                          system_changes=system_changes)

        # Add metadynamics energy and forces

        f_bias, _ = self.get_bias(atoms)

        self.results['forces'] += f_bias
        self.results['f_bias'] = f_bias

        if add_steps:
            for i, step in enumerate(self.steps_from_old):
                self.steps_from_old[i] = step + 1


class BatchNeuralMetadynamics(NeuralMetadynamics):

    def __init__(self,
                 model,
                 pushing_params,
                 old_atoms=None,
                 device='cpu',
                 en_key='energy',
                 directed=DEFAULT_DIRECTED,
                 **kwargs):

        NeuralMetadynamics.__init__(self,
                                    model=model,
                                    pushing_params=pushing_params,
                                    old_atoms=old_atoms,
                                    device=device,
                                    en_key=en_key,
                                    directed=directed,
                                    **kwargs)

        self.query_nxyz = None
        self.mol_idx = None

    def rmsd_prelims(self, atoms):

        # f_damp is the turn-on on timescale, measured in
        # number of steps. From https://pubs.acs.org/doi/pdf/
        # 10.1021/acs.jctc.9b00143

        kappa = self.pushing_params["kappa"]
        steps_from_old = torch.Tensor(self.steps_from_old)
        f_damp = (2 / (1 + torch.exp(-kappa * steps_from_old)) -
                  1)

        # k_i depends on the number of atoms so must be done by batch
        # given in mHartree / atom in CREST paper

        k_i = ((self.pushing_params['k_i'] / 1000 *
                units.Hartree * atoms.num_atoms))

        # given in Bohr^(-2) in CREST paper
        alpha_i = ((self.pushing_params['alpha_i'] /
                    units.Bohr ** 2))

        return k_i, alpha_i, f_damp

    def get_query_nxyz(self, keep_idx):
        if self.query_nxyz is not None:
            return self.query_nxyz

        query_nxyz = torch.stack([torch.Tensor(old_atoms.get_nxyz())[keep_idx, :]
                                  for old_atoms in self.old_atoms])
        self.query_nxyz = query_nxyz

        return query_nxyz

    def append_atoms(self, atoms):
        super().append_atoms(atoms)
        self.query_nxyz = None

    def make_nxyz(self,
                  atoms):

        keep_idx = self.get_keep_idx(atoms)
        ref_nxyz = torch.Tensor(atoms.get_nxyz())[keep_idx, :]
        query_nxyz = self.get_query_nxyz(keep_idx)

        return ref_nxyz, query_nxyz, keep_idx

    def get_mol_idx(self,
                    atoms,
                    keep_idx):

        if self.mol_idx is not None:
            assert self.mol_idx.max() + 1 == len(atoms.num_atoms)
            return self.mol_idx

        num_atoms = atoms.num_atoms
        counter = 0

        mol_idx = []

        for i, num in enumerate(num_atoms):
            mol_idx.append(torch.ones(num).long() * i)
            counter += num

        mol_idx = torch.cat(mol_idx)[keep_idx]
        self.mol_idx = mol_idx

        return mol_idx

    def get_num_atoms_tensor(self,
                             mol_idx,
                             atoms):

        num_atoms = torch.LongTensor([(mol_idx == i).nonzero().shape[0]
                                      for i in range(len(atoms.num_atoms))])

        return num_atoms

    def get_v_f_bias(self,
                     rmsd,
                     ref_xyz,
                     k_i,
                     alpha_i,
                     f_damp):

        v_bias = (f_damp.reshape(-1, 1) * k_i *
                  torch.exp(-alpha_i * rmsd ** 2)).sum()

        f_bias = -compute_grad(inputs=ref_xyz,
                               output=v_bias)

        output = [v_bias.reshape(-1).detach().cpu(),
                  f_bias.detach().cpu()]

        return output

    def rmsd_push(self, atoms):

        if not self.old_atoms:
            return np.zeros((len(atoms), 3)), np.zeros(len(atoms.num_atoms))

        k_i, alpha_i, f_damp = self.rmsd_prelims(atoms)

        ref_nxyz, query_nxyz, keep_idx = self.make_nxyz(atoms=atoms)
        mol_idx = self.get_mol_idx(atoms=atoms,
                                   keep_idx=keep_idx)
        num_atoms_tensor = self.get_num_atoms_tensor(mol_idx=mol_idx,
                                                     atoms=atoms)

        # note - everything is done on CPU, which is much faster than GPU. E.g. for
        # 30 molecules in a batch, each around 70 atoms, it's 4 times faster to do
        # this on CPU than GPU

        rmsd, ref_xyz = batch_compute_distance(ref_nxyz=ref_nxyz,
                                               query_nxyz=query_nxyz,
                                               mol_idx=mol_idx,
                                               num_atoms_tensor=num_atoms_tensor,
                                               store_grad=True)

        v_bias, f_bias = self.get_v_f_bias(rmsd=rmsd,
                                           ref_xyz=ref_xyz,
                                           k_i=k_i,
                                           alpha_i=alpha_i,
                                           f_damp=f_damp)

        final_f_bias = torch.zeros(len(atoms), 3)
        final_f_bias[keep_idx] = f_bias
        nan_idx = torch.bitwise_not(torch.isfinite(final_f_bias))
        final_f_bias[nan_idx] = 0

        return final_f_bias.numpy(), v_bias.numpy()


class NeuralGAMD(NeuralFF):
    """
    NeuralFF for Gaussian-accelerated molecular dynamics (GAMD)
    """

    def __init__(self,
                 model,
                 k_0,
                 V_min,
                 V_max,
                 device=0,
                 en_key='energy',
                 directed=DEFAULT_DIRECTED,
                 **kwargs):

        NeuralFF.__init__(self,
                          model=model,
                          device=device,
                          en_key=en_key,
                          directed=DEFAULT_DIRECTED,
                          **kwargs)
        self.V_min = V_min
        self.V_max = V_max

        self.k_0 = k_0
        self.k = self.k_0 / (self.V_max - self.V_min)

    def calculate(self,
                  atoms,
                  properties=['energy', 'forces'],
                  system_changes=all_changes):

        if not any([isinstance(self.model, i) for i in UNDIRECTED]):
            check_directed(self.model, atoms)

        super().calculate(atoms=atoms,
                          properties=properties,
                          system_changes=system_changes)

        old_en = self.results['energy']
        if old_en < self.V_max:
            old_forces = self.results['forces']
            f_bias = -self.k * (self.V_max - old_en) * old_forces

            self.results['forces'] += f_bias
