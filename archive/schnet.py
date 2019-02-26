from torch.nn import functional as F
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import Parameter
#from GraphFP_qm9 import GraphDis
import torch.nn.functional as F
from torch.autograd.gradcheck import zero_gradients
from torch.autograd import grad

from torch.nn.init import xavier_uniform_

from functools import partial
from torch.nn.init import constant_
zeros_initializer = partial(constant_, val=0.)

class GraphDis(torch.nn.Module):

    """Compute distance matrix on the fly 
    
    Attributes:
        box_len (TYPE): Length of the box
        cutoff (TYPE): cufoff 
        device (TYPE): which gpu to use
        F (TYPE): Fr + Fe
        Fe (TYPE): edge feature length
        Fr (TYPE): node geature length
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
        '''
        input:  xyz torch.Tensor (B, N, 3)
        return:   edge feature matrix torch.Tensor (B, N, N, 3 or 1)
        '''
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
    

def compute_grad(inputs, output):
    """compute gradient of the scalar output with respect to inputs 
    
    Args:
        inputs (torch.Tensor): torch tensor, requires_grad=True
        output (torch.Tensor): scalar output 
    
    Returns:
        torch.Tensor: gradients with respect to each input component 
    """
    assert inputs.requires_grad
    
    gradspred, = grad(output, inputs, grad_outputs=output.data.new(output.shape).fill_(1),
                   create_graph=True, retain_graph=True)
    
    return gradspred

# Gaussian Smearing 
def gaussian_smearing(distances, offset, widths, centered=False):
    """
    Perform gaussian smearing on interatomic distances.

    Args:
        distances (torch.Tensor): Variable holding the interatomic distances (B x N_at x N_nbh)
        offset (torch.Tensor): torch tensor of offsets
        centered (bool): If this flag is chosen, Gaussians are centered at the origin and the
                  offsets are used to provide their widths (used e.g. for angular functions).
                  Default is False.

    Returns:
        torch.Tensor: smeared distances (B x N_at x N_nbh x N_gauss)

    """
    if centered == False:
        # Compute width of Gaussians (using an overlap of 1 STDDEV)
        # widths = offset[1] - offset[0]
        coeff = -0.5 / torch.pow(widths, 2)
        # Use advanced indexing to compute the individual components
        diff = distances[:, :, :, None] - offset[None, None, None, :]
    else:
        # If Gaussians are centered, use offsets to compute widths
        coeff = -0.5 / torch.pow(offset, 2)
        # If centered Gaussians are requested, don't substract anything
        diff = distances[:, :, :, None]
    # Compute and return Gaussians
    gauss = torch.exp(coeff * torch.pow(diff, 2))
    return gauss


class GaussianSmearing(nn.Module):
    """
    Wrapper class of gaussian_smearing function. Places a predefined number of Gaussian functions within the
    specified limits.

    Args:
        start (float): Center of first Gaussian.
        stop (float): Center of last Gaussian.
        n_gaussians (int): Total number of Gaussian functions.
        centered (bool):  if this flag is chosen, Gaussians are centered at the origin and the
              offsets are used to provide their widths (used e.g. for angular functions).
              Default is False.
        trainable (bool): If set to True, widths and positions of Gaussians are adjusted during training. Default
              is False.
    """

    def __init__(self, start, stop, n_gaussians, centered=False, trainable=False):
        super(GaussianSmearing, self).__init__()
        offset = torch.linspace(start, stop, n_gaussians)
        widths = torch.FloatTensor((offset[1] - offset[0]) * torch.ones_like(offset))
        if trainable:
            self.width = nn.Parameter(widths)
            self.offsets = nn.Parameter(offset)
        else:
            self.register_buffer('width', widths)
            self.register_buffer('offsets', offset)
        self.centered = centered

    def forward(self, distances):
        """
        Args:
            distances (torch.Tensor): Tensor of interatomic distances.

        Returns:
            torch.Tensor: Tensor of convolved distances.

        """
        return gaussian_smearing(distances, self.offsets, self.width, centered=self.centered)

class Dense(nn.Linear):
    """ Applies a dense layer with activation: :math:`y = activation(Wx + b)`

    Args:
        in_features (int): number of input feature
        out_features (int): number of output features
        bias (bool): If set to False, the layer will not adapt the bias. (default: True)
        activation (callable): activation function (default: None)
        weight_init (callable): function that takes weight tensor and initializes (default: xavier)
        bias_init (callable): function that takes bias tensor and initializes (default: zeros initializer)
    """

    def __init__(self, in_features, out_features, bias=True, activation=None,
                 weight_init=xavier_uniform_, bias_init=zeros_initializer):
        self.weight_init = weight_init
        self.bias_init = bias_init
        self.activation = activation

        super(Dense, self).__init__(in_features, out_features, bias)

    def reset_parameters(self):
        """
        Reinitialize model parameters.
        """
        self.weight_init(self.weight)
        if self.bias is not None:
            self.bias_init(self.bias)

    def forward(self, inputs):
        """
        Args:
            inputs (dict of torch.Tensor): SchNetPack format dictionary of input tensors.

        Returns:
            torch.Tensor: Output of the dense layer.
        """
        y = super(Dense, self).forward(inputs)
        if self.activation:
            y = self.activation(y)

        return y

def shifted_softplus(x):
    """
    Shifted softplus activation function of the form:
    :math:`y = ln( e^{-x} + 1 ) - ln(2)`

    Args:
        x (torch.Tensor): Input tensor

    Returns:
        torch.Tensor: Shifted softplus applied to x

    """
    return F.softplus(x) - np.log(2.0)


class InteractionBlock(nn.Module):
    """ 
    Interaction Block with a distance filter based on Gaussian smearing 

    Args:
        n_atom_basis (int): number of atom features
        n_filters (int): filter dimensions
        n_gaussians (int): number of guassian basis
        
    """

    def __init__(self, n_atom_basis, n_filters, n_gaussians, cutoff_soft, trainable_gauss):
        super(InteractionBlock, self).__init__()
        self.smearing = GaussianSmearing(start=0.0, stop=cutoff_soft, n_gaussians=n_gaussians, trainable=trainable_gauss)

        # filter network 
        self.DistanceFilter1 = Dense(in_features= n_gaussians, out_features=n_filters, activation=shifted_softplus)
        self.DistanceFilter2 = Dense(in_features= n_filters, out_features=n_filters)

        # in2f
        self.AtomFilter = Dense(in_features=n_atom_basis, out_features=n_filters, bias=False)

        # f2_out
        self.Dense1 = Dense(in_features=n_filters, out_features= n_atom_basis, activation=shifted_softplus)

        #dense
        self.Dense2 = Dense(in_features=n_atom_basis, out_features= n_atom_basis, activation=None)
        
    def forward(self, r, e, A):
        #e = self.smearing(e.squeeze())
        e = self.smearing(e.reshape(-1, r.shape[1], r.shape[1]))
        W = self.DistanceFilter1(e)
        W = self.DistanceFilter2(W)
        
        #r = r[:, None, :, :].expand(-1, r.shape[1], -1, -1) # expand to filtered 
        #print(W.shape)
        r = self.AtomFilter(r)
        # adjacency matrix used as a mask
        A = A.unsqueeze(3).expand(-1, -1, -1, r.shape[2])
        W = W * A

        # filter 
        y = r[:, None, :, :].expand(-1, r.shape[1], -1, -1)# * A 
        y = y * W * A
        
        # y = y.sum(2)
        #y = y * A.sum(2).reciprocal().expand_as(y) # cutoff_network
        y = y.sum(2)
        y = self.Dense1(y)
        # y = y * A.sum(2).reciprocal().expand_as(y)
        y = self.Dense2(y)
        
        #y = y.sum(2) # sum pooling 
        
        return y

class Net(nn.Module):
    """ 
    Module to compute energy 

    Args:
        n_atom_basis (int): number of atom features
        n_filters (int): filter dimensions
        n_gaussians (int): number of guassian basis
        device (int): which gpu to use
        
    """

    def __init__(self, n_atom_basis, n_filters, n_gaussians, cutoff_soft, device, T, trainable_gauss, box_len=None):
        super(Net, self).__init__()
        
        self.graph_dis = GraphDis(Fr=1, Fe=1, cutoff=10.0, box_len = box_len, device=device)


        self.convolutions = nn.ModuleList([InteractionBlock(n_atom_basis=n_atom_basis,
                                             n_filters=n_filters, n_gaussians=n_gaussians, 
                                             cutoff_soft =cutoff_soft, trainable_gauss=trainable_gauss) for i in range(T)])

        #self.atomEmbed = Dense(in_features= 1, out_features=n_atom_basis, bias=False)
        self.atomEmbed = nn.Embedding(100, n_atom_basis, padding_idx=0)
        self.atomwise1 = Dense(in_features= n_atom_basis, out_features= int(n_atom_basis/2), activation=shifted_softplus)
        self.atomwise2 = Dense(in_features= int(n_atom_basis/2), out_features=1)
        
        
    def forward(self, r, xyz):
        
        r, e ,A = self.graph_dis(r= r, xyz=xyz)

        #r = self.atomEmbed(r).squeeze()
        r = self.atomEmbed(r).reshape(r.shape[0], r.shape[1], -1)

        for i, conv in enumerate(self.convolutions):
            r = r + conv(r=r, e=e, A=A)

        r = self.atomwise1(r)
        r = self.atomwise2(r)
        
        r = r.sum(1)#.squeeze()
        return r