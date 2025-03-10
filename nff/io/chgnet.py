"""Convert NFF Dataset to CHGNet StructureData"""

from typing import Dict

import torch
from chgnet.data.dataset import StructureData
from pymatgen.io.ase import AseAtomsAdaptor

from nff.data import Dataset
from nff.io import AtomsBatch
from nff.utils.cuda import batch_detach, detach


def convert_nff_to_chgnet_structure_data(
    dataset: Dataset,
    cutoff: float = 5.0,
    shuffle: bool = True,
):
    """The function `convert_nff_to_chgnet_structure_data` converts a dataset in NFF format to a dataset in
    CHGNet structure data format.

    Parameters
    ----------
    dataset : Dataset
        The `dataset` parameter is an object of the `Dataset` class.
    cutoff : float
        The `cutoff` parameter is a float value that represents the distance cutoff for constructing the
    neighbor list in the conversion process. It determines the maximum distance between atoms within
    which they are considered neighbors. Any atoms beyond this distance will not be included in the
    neighbor list.
    shuffle : bool
        The `shuffle` parameter is a boolean value that determines whether the dataset should be shuffled

    Returns:
    -------
    a `chgnet_dataset` object of type `StructureData`.

    """
    dataset = dataset.copy()
    dataset.to_units("eV/atom")  # convert units to eV
    print(f"current units: {dataset.units}")
    atoms_batch_list = dataset.as_atoms_batches(cutoff=cutoff)
    pymatgen_structures = [AseAtomsAdaptor.get_structure(atoms_batch) for atoms_batch in atoms_batch_list]

    energies_per_atom = dataset.props["energy"]

    energy_grads = dataset.props["energy_grad"]
    forces = [-x for x in energy_grads] if isinstance(energy_grads, list) else -energy_grads
    stresses = dataset.props.get("stress", None)
    magmoms = dataset.props.get("magmoms", None)

    return StructureData(
        structures=pymatgen_structures,
        energies=energies_per_atom,
        forces=forces,
        stresses=stresses,
        magmoms=magmoms,
        shuffle=shuffle,
    )


def convert_data_batch(
    data_batch: Dict,  # noqa: FA100
    cutoff: float = 5.0,
    shuffle: bool = True,
):
    """Converts a dataset in NFF format to a dataset in
    CHGNet structure data format.

    Parameters
    ----------
    data_batch : Dict
        A dictionary of properties for each structure in the batch.
        Basically the props in NFF Dataset
        Example:
            props = {
                'nxyz': [np.array([[1, 0, 0, 0], [1, 1.1, 0, 0]]),
                            np.array([[1, 3, 0, 0], [1, 1.1, 5, 0]])],
                'lattice': [np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                            np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])],
                'num_atoms': [2, 2],
            }
    cutoff : float
        The `cutoff` parameter is a float value that represents the distance cutoff for constructing the
    neighbor list in the conversion process. It determines the maximum distance between atoms within
    which they are considered neighbors. Any atoms beyond this distance will not be included in the
    neighbor list.
    shuffle : bool
        The `shuffle` parameter is a boolean value that determines whether the dataset should be shuffled

    Returns:
    -------
    a `chgnet_dataset` object of type `StructureData`.

    """
    detached_batch = batch_detach(data_batch)
    nxyz = detached_batch["nxyz"]
    atoms_batch = AtomsBatch(
        nxyz[:, 0].long(),
        props=detached_batch,
        positions=nxyz[:, 1:],
        cell=(detached_batch["lattice"][0] if "lattice" in detached_batch else None),
        pbc="lattice" in detached_batch,
        cutoff=cutoff,
        dense_nbrs=False,
    )
    atoms_list = atoms_batch.get_list_atoms()

    pymatgen_structures = [AseAtomsAdaptor.get_structure(atoms_batch) for atoms_batch in atoms_list]

    energies = data_batch.get("energy")
    if energies is not None and len(energies) > 0:
        energies_per_atom = energies
    else:
        # fake energies
        energies_per_atom = torch.Tensor([0.0] * len(pymatgen_structures))
    energy_grads = data_batch.get("energy_grad")

    num_atoms = detach(data_batch["num_atoms"]).tolist()
    stresses = data_batch.get("stress")
    magmoms = data_batch.get("magmoms")

    if energy_grads is not None and len(energy_grads) > 0:
        forces = [-x for x in energy_grads] if isinstance(energy_grads, list) else -energy_grads
    else:
        forces = None

    if forces is not None and len(forces) > 0:
        # organize forces per structure
        forces = torch.split(torch.atleast_2d(forces), num_atoms)
    else:
        # fake forces for NFF Calculator
        forces = [torch.zeros_like(torch.Tensor(atoms_batch.get_positions())) for atoms_batch in atoms_list]

    if stresses is not None and len(stresses) > 0:
        stresses = torch.split(torch.atleast_2d(stresses), num_atoms)
    if magmoms is not None and len(magmoms) > 0:
        magmoms = torch.split(torch.atleast_2d(magmoms), num_atoms)

    return StructureData(
        structures=pymatgen_structures,
        energies=energies_per_atom,
        forces=forces,
        stresses=stresses,
        magmoms=magmoms,
    )
