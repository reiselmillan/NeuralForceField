
import torch
import numpy as np

class CVBase:
    def __init__(self, update_steps=[], **kwargs) -> None:
        self.update_steps = update_steps

    def update_idx(self, *args, **kwargs):
        pass


class CenterCageWall:
    def __init__(self, mol_idx=[], cage_idx=[], radius=0,  device='cpu'):
        self.mol_idx = torch.LongTensor(mol_idx)
        self.cage_idx = torch.LongTensor(cage_idx)
        self.radius = float(radius)
        self.device = device

    def get_value(self, positions):
        if not isinstance(positions, torch.Tensor):
            positions = torch.tensor(positions)
        mol_pos = positions[self.mol_idx]
        cage_pos = positions[self.cage_idx]
        mol_centroid = mol_pos.mean(axis=0) # mol center
        cage_centroid = cage_pos.mean(axis=0) # centroid of the whole structure

        dist = mol_centroid - cage_centroid
        dist = torch.linalg.norm(dist)
        if dist > self.radius:
            return dist - self.radius
        else:
            return 0
        

class ProjVectorVector:
    """
    Collective variable class. Projection of a position vector onto a reference vector
    Atomic indices are used to determine the coordiantes of the vectors.
    Params
    ------
    vector: list of int
       List of the indices of atoms that define the vector on which the position vector is projected
    indices: list if int
       List of indices of the mol/fragment
    reference: list of int
       List of atomic indices that are used as reference for the position vector

    note: the position vector is calculated in the method get_value
    """
    def __init__(self, vector=[], indices=[], reference=[], device='cpu'):
        self.vector = torch.tensor([float(i) for i in vector], device=device)
        self.vector = self.vector / torch.linalg.norm(self.vector) # normalize
        self.mol_inds = torch.LongTensor(indices)
        self.reference_inds = reference
        self.device = device

    def get_value(self, positions):
        mol_pos = positions[self.mol_inds]
        reference_pos = positions[self.reference_inds]
        mol_centroid = mol_pos.mean(axis=0) # mol center
        reference_centroid = reference_pos.mean(axis=0) # centroid of the whole structure

        # position vector with respect to the structure centroid
        rel_mol_pos = mol_centroid - reference_centroid

        # projection
        cv = torch.dot(rel_mol_pos.to(torch.float32), self.vector.to(torch.float32))
        return cv

class ProjVectorCentroid:
    """
    Collective variable class. Projection of a position vector onto a reference vector
    Atomic indices are used to determine the coordiantes of the vectors.
    Params
    ------
    note: the position vector is calculated in the method get_value
    """
    def __init__(self, refvecidx, idx, refcenteridx, abs=False, device='cpu', **kargs):
        self.vector_inds = torch.LongTensor(refvecidx)
        self.mol_inds = torch.LongTensor(idx)
        self.reference_inds = torch.LongTensor(refcenteridx)
        self.abs = abs
        print("INIT PVC: ")
        print(self.vector_inds, self.mol_inds, self.reference_inds, self.abs)

    def get_value(self, positions): 
        vector_pos = positions[self.vector_inds]
        vector = vector_pos[1] - vector_pos[0]
        vector = vector / torch.linalg.norm(vector)
        mol_pos = positions[self.mol_inds]
        reference_pos = positions[self.reference_inds]
        mol_centroid = mol_pos.mean(axis=0) # mol center
        reference_centroid = reference_pos.mean(axis=0) # centroid of the whole structure

        # position vector with respect to the structure centroid
        rel_mol_pos = mol_centroid - reference_centroid

        # projection
        cv = torch.dot(rel_mol_pos, vector)
        if self.abs:
            cv = abs(cv)
        return cv


class ProjVectorPlane:
    """
    Collective variable class. Projection of a position vector onto a the average plane
    of an arbitrary ring defined in the structure
    Atomic indices are used to determine the coordiantes of the vectors.
    Params
    ------
    mol_inds: list of int
       List of indices of the mol/fragment tracked by the CV
    ring_inds: list of int
       List of atomic indices of the ring for which the average plane is calculated.

    note: the position vector is calculated in the method get_value
    """
    def __init__(self, mol_inds = [], ring_inds = [], isabs=False):
        self.mol_inds = torch.LongTensor(mol_inds) # list of indices
        self.ring_inds = torch.LongTensor(ring_inds) # list of indices
        self.abs = isabs
        # both self.mol_coors and self.ring_coors torch tensors with atomic coordinates
        # initiallized as list but will be set to torch tensors with set_positions
        self.mol_coors = []
        self.ring_coors = []

    def set_positions(self, positions):
        # update coordinate torch tensors from the positions tensor
        self.mol_coors = positions[self.mol_inds]
        self.ring_coors = positions[self.ring_inds]

    def get_indices(self):
        return self.mol_inds + self.ring_inds

    def get_value(self, positions):
        """Calculates the values of the CV for a specific atomic positions

        Args:
            positions (torch tensor): atomic positions

        Returns:
            float: current values of the collective variable
        """
        self.set_positions(positions)
        mol_cm = self.mol_coors.mean(axis=0) # mol center
        ring_cm = self.ring_coors.mean(axis=0) # ring center
        # ring atoms to center
        self.ring_coors = self.ring_coors - ring_cm

        r1 = torch.zeros(3, device=self.ring_coors.device)
        N = len(self.ring_coors) # number of atoms in the ring
        for i, rl0 in enumerate(self.ring_coors):
            r1 = r1 + rl0 * np.sin(2 * np.pi * i / N)
        r1 = r1/N

        r2 = torch.zeros(3, device=self.ring_coors.device)
        for i, rl0 in enumerate(self.ring_coors):
            r2 = r2 + rl0 * np.cos(2 * np.pi * i / N)
        r2 = r2/N

        plane_vec = torch.cross(r1, r2)
        plane_vec = plane_vec / torch.linalg.norm(plane_vec)
        pos_vec = mol_cm - ring_cm

        cv = torch.dot(pos_vec, plane_vec)
        if self.abs:
            cv = abs(cv)
        return cv


class ProjOrthoVectorsPlane:
    """
    Collective variable class. Projection of a position vector onto a the average plane
    of an arbitrary ring defined in the structure
    Atomic indices are used to determine the coordiantes of the vectors.
    Params
    ------
    mol_inds: list of int
       List of indices of the mol/fragment tracked by the CV
    ring_inds: list of int
       List of atomic indices of the ring for which the average plane is calculated.

    note: the position vector is calculated in the method get_value
    """
    def __init__(self, mol_inds = [], ring_inds = []):
        self.mol_inds = torch.LongTensor(mol_inds) # list of indices
        self.ring_inds = torch.LongTensor(ring_inds) # list of indices
        # both self.mol_coors and self.ring_coors torch tensors with atomic coordinates
        # initiallized as list but will be set to torch tensors with set_positions
        self.mol_coors = []
        self.ring_coors = []

    def set_positions(self, positions):
        # update coordinate torch tensors from the positions tensor
        self.mol_coors = positions[self.mol_inds]
        self.ring_coors = positions[self.ring_inds]

    def get_indices(self):
        return self.mol_inds + self.ring_inds

    def get_value(self, positions):
        """Calculates the values of the CV for a specific atomic positions

        Args:
            positions (torch tensor): atomic positions

        Returns:
            float: current values of the collective variable
        """
        self.set_positions(positions)
        mol_cm = self.mol_coors.mean(axis=0) # mol center
        ring_cm = self.ring_coors.mean(axis=0) # ring center
        # ring atoms to center
        self.ring_coors = self.ring_coors - ring_cm

        r1 = torch.zeros(3, device=self.ring_coors.device)
        N = len(self.ring_coors) # number of atoms in the ring
        for i, rl0 in enumerate(self.ring_coors):
            r1 = r1 + rl0 * np.sin(2 * np.pi * i / N)
        r1 = r1/N

        r2 = torch.zeros(3, device=self.ring_coors.device)
        for i, rl0 in enumerate(self.ring_coors):
            r2 = r2 + rl0 * np.cos(2 * np.pi * i / N)
        r2 = r2/N

        # normalize r1 and r2
        r1 = r1 / torch.linalg.norm(r1)
        r2 = r2 / torch.linalg.norm(r2)
        # project position vector on r1 and r2
        pos_vec = mol_cm - ring_cm
        proj1 = torch.dot(pos_vec, r1)
        proj2 = torch.dot(pos_vec, r2)
        cv = proj1 + proj2
        return abs(cv)


class Distance(CVBase):
    def __init__(self, at1, at2, device="cpu", 
                update_at1=[], 
                update_at2=[], 
                update_steps = [], **kwargs):
        self.at1 = at1
        self.at2 = at2
        self.device = device
        self.update1 = update_at1
        self.update2 = update_at2
        super().__init__(update_steps, **kwargs)

    def get_value(self, positions):
        d = positions[self.at1] - positions[self.at2]
        d2 = d*d
        return torch.sqrt(d2.sum())
    
    def update_idx(self, positions):
        print("Updating idx for class Distance")
        
        currd2 = positions[self.at1] - positions[self.at2]
        currd2 = (currd2*currd2).sum()
        for ui in self.update1:
            d = positions[ui] - positions[self.at2]
            d2 = (d*d).sum()
            if d2 < currd2:
                self.at1 = ui
                currd2 = d2

        currd2 = positions[self.at1] - positions[self.at2]
        currd2 = (currd2*currd2).sum()
        for ui in self.update2:
            d = positions[ui] - positions[self.at1]
            d2 = (d*d).sum()
            if d2 < currd2:
                self.at2 = ui
                currd2 = d2   
        print(f"new atom1 {self.at1}  atom2 {self.at2}")  


class DiffDistance2:
    def __init__(self, idx=[], device="cpu"):
        self.idx = idx
        self.device = device

    def get_value(self, positions):
        d1 = positions[self.idx[0]] - positions[self.idx[1]]
        d2 = positions[self.idx[2]] - positions[self.idx[3]]
        
        d1_2 = d1*d1
        d2_2 = d2*d2
        return torch.sqrt(d2_2.sum()) - torch.sqrt(d1_2.sum())
    

class Dihedral:
    """
    Collective variable class. Projection of a position vector onto a reference vector
    Atomic indices are used to determine the coordiantes of the vectors.
    Params
    ------
    vector: list of int
       List of the indices of atoms that define the vector on which the position vector is projected
    indices: list if int
       List of indices of the mol/fragment
    reference: list of int
       List of atomic indices that are used as reference for the position vector

    note: the position vector is calculated in the method get_value
    """
    def __init__(self, idx=[], device="cpu"):
      self.idx  = idx
      self.device = device

    def get_value(self, positions):
      ba = -(positions[self.idx[0]] - positions[self.idx[1]])
      bc = positions[self.idx[1]] - positions[self.idx[2]]
      cd = positions[self.idx[3]] - positions[self.idx[2]]

      ba = ba/torch.linalg.norm(ba)
      bc = bc/torch.linalg.norm(bc)
      cd = cd/torch.linalg.norm(cd)

      n1 = torch.cross(ba, bc)
      n2 = torch.cross(bc, cd)
      m = torch.cross(n1, bc)
      x = torch.dot(n1, n2)
      y = torch.dot(m, n2)

      ang = 180.0 / torch.pi * torch.arctan2(y, x)
      return ang

cvtypes = {
    "pvc": ProjVectorCentroid,
    "distance": Distance
}

def get_cv_from_dic(val, device="cpu"):
    if val["type"].lower() == "pvp":
        mol_inds = val["mol_idx"]  # caution check type
        ring_inds = val["ring_idx"]
        isabs = val.get("abs", False)
        cv = ProjVectorPlane(mol_inds, ring_inds, isabs)
    elif val["type"].lower() == "pvv":
        mol_inds = val["mol_idx"]
        reference = val['ref_idx']
        vector = val["vector"]
        cv = ProjVectorVector(vector, mol_inds, reference, device=device) # z components
    elif val["type"].lower() == "proj_ortho_vectors_plane":
        mol_inds = [i-1 for i in val["mol_idx"]]  # caution check type
        ring_inds = [i-1 for i in val["ring_idx"]]
        cv = ProjOrthoVectorsPlane(mol_inds, ring_inds) # z components
    elif val["type"].lower() == "torsion":
        cv = Dihedral(val["idx"], device)
    elif val["type"].lower() in ["center_wall", "wall_center"]:
        cv = CenterCageWall(val["mol_idx"], val["center_idx"], val["radius"], device=device)
    elif val["type"].lower() in ["diff_distance2"]:
        cv = DiffDistance2(val["idx"], device)
    else:
        return cvtypes[val["type"]](**val)
        raise TypeError("Bad CV type")
    return cv
