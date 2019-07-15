import os
import json
import datetime
import time
import pickle
import numpy as np

import torch.optim as optim

import matplotlib.pyplot as plt
plt.switch_backend('agg')

from nff.nn.models import *
from nff.utils.scatter import *
from nff.md import * 
from nff.data.graphs import * 


class TrainWrapper:
    """A wrapper for training, validation, save and load models.
    
    Attributes:
        criterion (MSEloss): Description
        data (graph): Description
        device (TYPE): Description
        dir_loc (TYPE): Description
        energiesmae (float): Description
        predictedforces (list): Description
        targetforces (list): Description
        forcesmae (float): Description
        graph_batching (Boolean): If True, use graph batch input
        job_name (str): name of the job or experimeent
        mae (TYPE): Description
        model (TYPE): Description
        model_path (TYPE): Description
        N_batch (int): Description
        N_test (int): Description
        N_train (int): Description
        optimizer (TYPE): Description
        par (dict): a dictionary file for hyperparameters
        root (str): the root path for the saving training results 
        scheduler (Boolean): Description
        train_f_log (list): Description
        train_u_log (list): Description
        predictedenergies (list): Description
        targetenergies (list): Description
    """

    def __init__(self,par, device, job_name, graph_batching=True,
                 graph_data=None,  root="./", train_flag=False, shift=False):
        """Summary
        
        Args:
            par (TYPE): Description
            graph_data (TYPE): Description
            device (TYPE): Description
            job_name (TYPE): Description
            graph_batching (bool, optional): Description
            root (str, optional): Description
        """
        if graph_data == None and train_flag == True:
            raise ValueError("No graph data provided for training")
        if graph_data is not None and train_flag == False:
            raise ValueError("You import a graph dataset but dont want to train on the data, are you sure?")
        self.device = device
        self.job_name = job_name
        self.root = root 
        self.train_flag = train_flag
        self.par = par # needs to input parameters 
        self.check_parameters()

        if graph_data is not None:
            self.data = graph_data
            self.initialize_data()
            
        self.initialize_model()
        self.initialize_optim()
        
        if train_flag is False:
            print("need to load a pre-trained model")
        self.graph_batching = graph_batching
        
    def check_parameters(self):
        assert type(self.par["n_filters"]) == int, "Invalid filter dimension, it should be an integer"
        assert type(self.par["n_gaussians"]) == int, "Invalid number of gaussian basis, it should be an integer"
        assert type(self.par["optim"]) == float, "the learning rate is not an float"
        assert type(self.par["train_percentage"]) == float and self.par["optim"] < 1.0, "the training data percentage is invalid"
        assert type(self.par["T"]) == int, "number of convolutions have to an integer"
        assert type(self.par["batch_size"]) == int, "invalid batch size"
        assert type(self.par["cutoff"]) == float, "Invalid cutoff radius"
        assert type(self.par["max_epoch"]) == int, "max epoch should be an integer"
        assert type(self.par["trainable_gauss"]) == bool, "should be boolean value"
        assert type(self.par["rho"]) == float, "Rho should be float"
        assert type(self.par["eps"]) == float and self.par["eps"] <= 1.0, "Invalid convergence criterion"

    def initialize_model(self):

        if self.train_flag == True:

            print("setting up directories for saving training files")

            self.train_u_log = []
            self.train_f_log = []

            # create directoires if not exists 
            if not os.path.exists(self.root):
                os.makedirs(self.root)
                
            # obtain a time stamp 
            currentDT = datetime.datetime.now()

            date = str(currentDT).split()[0].split("-")[1:]
            self.dir_loc = self.root + self.job_name + "_" + "".join(date)
            
            if not os.path.exists(self.dir_loc):
                os.makedirs(self.dir_loc)

            with open(self.dir_loc + "/par.json", "w") as write_file:
                json.dump(self.par, write_file, indent=4)
        
        bondpar = self.par.get("bondpar", 50.0)
        box_vec = self.par.get("box_vec", None)

        self.model = BondNet(n_atom_basis = self.par["n_atom_basis"],
                             n_filters = self.par["n_filters"],
                             n_gaussians= self.par["n_gaussians"], 
                             cutoff_soft= self.par["cutoff"], 
                             trainable_gauss = self.par["trainable_gauss"],
                             T=self.par["T"],
                             device=self.device,
                             bondpar=bondpar,
                             box_len=box_vec).to(self.device)        
    
    def initialize_optim(self):
        self.optimizer = optim.Adam(list(self.model.parameters()), lr=self.par["optim"])
        
        if self.par["scheduler"]:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 
                                                                  'min', 
                                                                  min_lr=1.5e-7, 
                                                                  verbose=True, factor = 0.5, patience= 20,
                                                                  threshold=5e-5)
        self.criterion = torch.nn.MSELoss()
        self.mae = torch.nn.L1Loss()
         
    def initialize_data(self):
        
        self.N_batch = len(self.data.batches)
        self.N_train = int(self.par["train_percentage"] * self.N_batch)
        self.N_test = self.N_batch - self.N_train - 1 # ignore the last batch 
        
    def parse_batch(self, index, data=None):
        """Summary
        
        Args:
            index (int): index of the batch in GraphDataset
        
        Returns:
            TYPE: Description
        """
        if data == None:
            data = self.data

        a = data.batches[index].data["a"].to(self.device)

        r = data.batches[index].data["r"][:, [0]].to(self.device)

        f = data.batches[index].data["r"][:, 1:4].to(self.device)

        u = data.batches[index].data["y"].to(self.device)
        
        N = data.batches[index].data["N"]
        
        xyz = data.batches[index].data["xyz"].to(self.device)

        try: # try to get bond adjacency matrix 
            bond_adj = data.batches[index].data["bond_a"].to(self.device)
            bond_len = data.batches[index].data["bond_len"].to(self.device)
            return xyz, a, bond_adj, bond_len, r, f, u, N
        except:
            return xyz, a, r, f, u, N
    
    def train(self, N_epoch):
        """Summary
        
        Args:
            N_epoch (int): number of epoches to be trained 
        """

        self.start_time = time.time()

        for epoch in range(N_epoch):

            # check if max epoches are reached 
            if len(self.train_f_log) >= self.par["max_epoch"]:
                print("max epoches reached")
                break 

            train_energiesmae = 0.0
            train_forcesmae = 0.0
            
            for i in range(self.N_train):

                try:
                    xyz, a, bond_adj, bond_len, r, f, u, N = self.parse_batch(i)
                except:
                    xyz, a, r, f, u, N = self.parse_batch(i)
                    bond_adj = None
                    bond_len = None
                xyz.requires_grad = True

                # check if the input has graphs of various sizes 
                if len(set(N)) == 1:
                    graph_size_is_same = True
                else:
                    graph_size_is_same = False

                # Compute energies 
                if self.graph_batching:
                    U = self.model(r=r, bond_adj=bond_adj, bond_len=bond_len, xyz=xyz, a=a, N=N) 
                else:
                    assert graph_size_is_same # make sure all the graphs needs to have the same size
                    U = self.model(r=r.reshape(-1, N[0]), xyz=xyz.reshape(-1, N[0], 3), bond_adj=bond_adj, bond_len=bond_len)
                    
                f_pred = -compute_grad(inputs=xyz, output=U)

                # comput loss
                loss_force = self.criterion(f_pred, f)
                loss_u = self.criterion(U, u)
                loss = loss_force + self.par["rho"] * loss_u

                # update parameters
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # compute MAE
                train_energiesmae += self.mae(U, u) # compute MAE
                train_forcesmae += self.mae(f_pred, f)

            # averaging MAE

            train_u = train_energiesmae.item()/self.N_train
            train_force = train_forcesmae.item()/self.N_train

            self.train_u_log.append(train_u)
            self.train_f_log.append(train_force)
            
            # scheduler
            if self.par["scheduler"] == True:
                self.scheduler.step(train_force)
            else:
                pass

            # print loss
            print("epoch %d  U train: %.3f  force train %.3f" % (epoch, train_u, train_force))

            self.time_elapsed = time.time() - self.start_time

            # check convergence 
            current_lr = self.optimizer.param_groups[0]["lr"]
            if current_lr <= 1.5e-7:
                print("training converged")
                break

        #self.save_model()
        self.save_train_log()
            
    def validate(self, data=None, savefig=True):
        """Summary
        """
        self.predictedforces = []
        self.targetforces = []
        self.predictedenergies = []
        self.targetenergies = []

        if not os.path.exists(self.root + self.job_name):
            os.makedirs(self.root + self.job_name)

        # decide data 
        if data == None:
            data = self.data#.batches[self.N_train: self.N_train + self.N_test - 1]
            start_index = self.N_train - 1
            N_test = self.N_test
        else:
            start_index = 0 
            N_test = len(data.batches)

        try:
            list_species = set(data.batches[0].data["name"])
            list_species = [item for item in list_species if item is not None]
            species_trained = sorted(list_species)
        except:
            species_trained = ["nospecies"]

        for i in range(N_test):

            # parse_data
            try:
                xyz, a, bond_adj, bond_len, r, f, u, N = self.parse_batch(start_index + i, data=data)
            except:
                xyz, a, r, f, u, N = self.parse_batch(start_index + i, data=data)
                bond_adj = None
                bond_len = None

            xyz.requires_grad = True

            if self.graph_batching:
                u_pred = self.model(r=r, xyz=xyz, a=a, N=N, bond_adj=bond_adj, bond_len=bond_len) 
            else:
                u_pred = self.model(r=r.reshape(-1, N[0]), xyz=xyz.reshape(-1, N[0], 3), bond_adj=bond_adj, bond_len=bond_len)
                
            f_pred = -compute_grad(inputs=xyz, output=u_pred).reshape(-1)

            self.predictedforces.append(f_pred.detach().cpu().numpy())
            self.targetforces.append(f.reshape(-1).detach().cpu().numpy())
            
            self.predictedenergies.append(u_pred.detach().cpu().numpy())
            self.targetenergies.append(u.reshape(-1).detach().cpu().numpy())

        self.targetforces = np.concatenate( self.targetforces, axis=0 ).reshape(-1)
        self.predictedforces = np.concatenate( self.predictedforces, axis=0 ).reshape(-1)
        self.targetenergies = np.concatenate( self.targetenergies, axis=0 ).reshape(-1)
        self.predictedenergies = np.concatenate( self.predictedenergies, axis=0 ).reshape(-1)
        
        # compute force & energy MAE
        self.forcesmae = np.abs(self.predictedforces - self.targetforces).mean()
        self.energiesmae = np.abs(self.predictedenergies - self.targetenergies).mean()
        

        print("forcesmae", self.forcesmae, "kcal/mol A")
        print("energiesmae", self.energiesmae, "kcal/mol")
       
    def plot_results(self):
        fig, ax = plt.subplots(1, 2, figsize=(13,6))

        ax[0].set_title("Energies (validation)")
        ax[1].set_title("Forces (validation)")
    
        ax[1].scatter(self.targetforces,
                   self.predictedforces,
                   label="force MAE: " + str(self.forcesmae) + " kcal/mol A",
                   alpha=0.3,
                   s=6)

        ax[1].set_xlabel("test")
        ax[1].set_ylabel("prediction")
        ax[1].legend()

        ax[0].scatter(self.targetenergies, self.predictedenergies, label="energy MAE: " + str(self.energiesmae) + " kcal/mol",  alpha=0.3, s=6)
        ax[0].set_xlabel("test")
        ax2.set_ylabel("prediction")
        ax2.legend()
    
        now = datetime.datetime.now()
        timestamp = now.strftime('%Y%m%d-%H%M')
    
        f.suptitle(",".join(species_trained[:3])+"validations", fontsize=14)

        if savefig:
            plt.savefig(self.root + str(self.job_name)+"/" +"&".join(species_trained[:3]) + timestamp + "validation.jpg")

    def save_model(self, save_path=None):
        if save_path is None:
            self.model_path = os.path.join(self.dir_loc, "model.pt")
            torch.save(self.model.state_dict(), self.model_path)
        else:
            torch.save(self.model.state_dict(), save_path)

    def load_model(self, load_path):
        print("loading models from" + load_path)
        self.model.load_state_dict(torch.load(load_path))
    
    def save_train_log(self):
        if not os.path.exists(self.root):
            os.makedirs(self.root)
        # save the training log for energies and force 
        log = np.array([self.train_u_log, self.train_f_log]).transpose()
        np.savetxt(self.dir_loc + "/log.csv", log, delimiter=",")
        
    def load_train_log(self):
        log = np.loadtxt(self.dir_loc + "/log.csv", delimiter=",")
        return log

    def save_summary(self):
        # the final test loss, number of epochs trained

        self.validate() 

        train_state = dict()
        train_state["epoch_trained"] = len(self.train_f_log)
        train_state["test_forcesmae"] = self.forcesmae.item()
        train_state["test_energiesmae"] = self.energiesmae.item()
        train_state["time_per_epoch"] = self.time_elapsed / len(self.train_f_log)

        # dump json 
        with open(self.dir_loc+"/results.json", "w") as write_file:
            json.dump(train_state, write_file , indent=4)

        # dump test and predict energy and forces 
        val_energy = np.array([self.targetenergies, self.predictedenergies]).transpose()
        val_force = np.array([self.targetforces, self.predictedforces]).transpose()

        np.savetxt(self.dir_loc + "/val_energy.csv", val_energy, delimiter=",")
        np.savetxt(self.dir_loc + "/val_force.csv", val_force, delimiter=",")