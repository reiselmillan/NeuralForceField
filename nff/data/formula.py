import re
import numpy as np

def get_atom_dic(atoms):
    atoms_dic = {}
    for at in atoms:
        if at.symbol in atoms_dic.keys():
            atoms_dic[at.symbol] += 1
        else:
            atoms_dic[at.symbol] = 1
    return atoms_dic

def get_atom_count(formula):
    import re
    dictio = dict(re.findall('([A-Z][a-z]?)([0-9]*)', formula))
    for key, val in dictio.items():
        dictio[key] = int(val) if val.isdigit() else 1

    return dictio

def all_atoms(ground_en):
    atom_list = []
    for formula in ground_en.keys():
        dictio = get_atom_count(formula)
        atom_list += list(dictio.keys())
    atom_list = list(set(atom_list))

    return atom_list

def reg_atom_count(formula, atoms):
    dictio = get_atom_count(formula)
    count_array = np.array([dictio.get(atom, 0) for atom in atoms])
    return count_array

def get_ref_energy(formula, stoidict):
    # get reference energy dependget_atom_count(f)ing on the formula
    elem_dic = get_atom_count(formula)
    ref_en = 0
    for ele, num in elem_dic.items():
        ref_en += num * stoidict[ele]
    ref_en += stoidict['offset']
    return ref_en