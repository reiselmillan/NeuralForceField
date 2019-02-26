from torch.nn import functional as F
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.autograd.gradcheck import zero_gradients
from torch.autograd import grad

from projects.NeuralForceField.layers import * 
from projects.NeuralForceField.scatter import * 

class GraphDis(torch.nn.Module):

    """Compute distance matrix on the fly 
    
    Attributes:
        box_len (numpy.array): Length of the box, dim = (3, )
        cutoff (float): cufoff 
        device (int or str): GPU id or "cpu"
        F (int): Fr + Fe
        Fe (int): edge feature length
        Fr (int): node geature length
    """
    
    def __init__(self, Fr, Fe, device, cutoff, box_len=None):
        super(GraphDis, self).__init__()

        self.Fr = Fr
        self.Fe = Fe # include distance
        self.F = Fr + Fe
        self.device = device
        self.cutoff = cutoff
        if box_len is not None:
            self.box_len = torch.Tensor(box_len).to(self.device)
        else:
            self.box_len = None
    
    def get_bond_vector_matrix(self, frame):
        """A function to compute the distance matrix 
        
        Args:
            frame (torch.FloatTensor): coordinates of (B, N, 3)
        
        Returns:
            torch.FloatTensor: distance matrix of dim (B, N, N, 1)
        """
        device = self.device
        cutoff = self.cutoff
        
        N_atom = frame.shape[1]
        frame = frame.view(-1, N_atom, 1, 3)
        dis_mat = frame.expand(-1, N_atom, N_atom, 3) - frame.expand(-1, N_atom, N_atom, 3).transpose(1,2)
        
        if self.box_len is not None:

            # build minimum image convention 
            box_len = self.box_len
            mask_pos = dis_mat.ge(0.5*box_len).type(torch.FloatTensor).to(self.device)
            mask_neg = dis_mat.lt(-0.5*box_len).type(torch.FloatTensor).to(self.device)
            
            # modify distance 
            dis_add = mask_neg * box_len
            dis_sub = mask_pos * box_len
            dis_mat = dis_mat + dis_add - dis_sub
        
        # create cutoff mask
        dis_sq = dis_mat.pow(2).sum(3)                  # compute squared distance of dim (B, N, N)
        mask = (dis_sq <= cutoff ** 2) & (dis_sq != 0)                 # byte tensor of dim (B, N, N)
        A = mask.unsqueeze(3).type(torch.FloatTensor).to(self.device) #         
        
        # 1) PBC 2) # gradient of zero distance 
        dis_sq = dis_sq.unsqueeze(3)
        dis_sq = (dis_sq * A) + 1e-8# to make sure the distance is not zero, otherwise there will be inf gradient 
        dis_mat = dis_sq.sqrt()
        
        # compute degree of nodes 
        # d = A.sum(2).squeeze(2) - 1
        return(dis_mat, A.squeeze(3)) 

    def forward(self, r, xyz=None):

        F = self.F  # F = Fr + Fe
        Fr = self.Fr # number of node feature
        Fe = self.Fe # number of edge featue
        B = r.shape[0] # batch size
        N = r.shape[1] # number of nodes 
        device = self.device

        # Compute the bond_vector_matrix, which has shape (B, N, N, 3), and append it to the edge matrix
        e, A = self.get_bond_vector_matrix(frame=xyz)# .type(torch).to(device=device.index)
        e = e.type(torch.FloatTensor).to(self.device)
        A = A.type(torch.FloatTensor).to(self.device)
        #d = d.type(torch.LongTensor).to(self.device)
        
        return(r, e, A)


class InteractionBlock(nn.Module):

    """The convolution layer with filter. To be merged with GraphConv class.
    
    Attributes:
        AtomFilter (TYPE): Description
        avg_flag (Boolean): if True, perform a mean pooling 
        Dense1 (Dense()): dense layer 1 to obtain the updated atomic embedding 
        Dense2 (Dense()): dense layer 2 to obtain the updated atomic embedding
        DistanceFilter1 (Dense()): dense layer 1 for filtering gaussian expanded distances 
        DistanceFilter2 (Dense()): dense layer 1 for filtering gaussian expanded distances 
        smearing (GaussianSmearing()): gaussian basis expansion for distance matrix of dimension B, N, N, 1
        smearing_graph (GaussianSmearing()): gaussian basis expansion for distance list of dimension N, N_nbh, 1
    """
    
    def __init__(self, n_atom_basis, n_filters, n_gaussians, cutoff_soft, trainable_gauss, avg_flag=False):
        super(InteractionBlock, self).__init__()

        self.avg_flag = avg_flag
        self.smearing = GaussianSmearing(start=0.0, stop=cutoff_soft,
                                         n_gaussians=n_gaussians, trainable=trainable_gauss)
        self.smearing_graph = GaussianSmearing(start=0.0, stop=cutoff_soft,
                                         n_gaussians=n_gaussians, trainable=trainable_gauss, graph=True)

        self.DistanceFilter1 = Dense(in_features= n_gaussians, out_features=n_gaussians, activation=shifted_softplus)
        self.DistanceFilter2 = Dense(in_features= n_gaussians, out_features=n_filters)
        self.AtomFilter = Dense(in_features=n_atom_basis, out_features=n_filters, bias=False)
        self.Dense1 = Dense(in_features=n_filters, out_features= n_atom_basis, activation=shifted_softplus)
        self.Dense2 = Dense(in_features=n_atom_basis, out_features= n_atom_basis, activation=None)
        
    def forward(self, r, e, A=None, a=None):       
        if a is None: # non-batch case ...
            
            if A is None:
                raise ValueError('need to input A')
            
            e = self.smearing(e.reshape(-1, r.shape[1], r.shape[1]))
            W = self.DistanceFilter1(e)
            W = self.DistanceFilter2(e)

            r = self.AtomFilter(r)
            
           # adjacency matrix used as a mask
            A = A.unsqueeze(3).expand(-1, -1, -1, r.shape[2])
            W = W #* A
            y = r[:, None, :, :].expand(-1, r.shape[1], -1, -1)
            
            # filtering 
            y = y * W * A

            # Aggregate 
            if self.avg_flag == False:
                 # sum pooling 
                y = y.sum(2)
            else:
                # mean pooling
                y = y * A.sum(2).reciprocal().expand_as(y) 
                y = y.sum(2)

            y = self.Dense1(y)
            y = self.Dense2(y)
            
        else:
            e = self.smearing_graph(e)
            W = self.DistanceFilter1(e)
            W = self.DistanceFilter2(e)
            W = W.squeeze()

            r = self.AtomFilter(r)
            # Filter 
            y = r[a[:, 1]].squeeze()

            y= y * W

            # Atomwise sum 
            y = scatter_add(src=y, index=a[:, 0], dim=0)
            
            # feed into Neural networks 
            y = self.Dense1(y)
            y = self.Dense2(y)

        return y