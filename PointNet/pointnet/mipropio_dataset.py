from __future__ import print_function
import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
import sys
from tqdm import tqdm 
import json
from plyfile import PlyData, PlyElement
import matplotlib.pyplot as plt
from torch.autograd import Variable
#from pointnet.model import PointNetDenseCls
import pptk
import random

class ParisDataset(data.Dataset):
    
    def __init__(self, root, npoints=5000, datapath='split_de_paris10clases/reduced10k/normalized'):

        self.npoints = npoints
        self.root = root   #Root es donde esta el dataset
        self.datapath = datapath    #'split_de_paris10clases/reduced10k/normalized'  #si se pasa como param de fuera cambia
        self.available = [file for file in os.listdir(self.datapath)]
        
        
    def __getitem__(self, index):
        #chosen_file = 'Paris_' + str(index).zfill(3) + '.npy'
        #data = np.load(str(self.root + chosen_file)) #+'/'+
        chosen_file = self.available[index]
        data = np.load(os.path.join(self.datapath, chosen_file))
        #x = data[:, 0]
        #y = data[:, 1]
        #z = data[:, 2]
        #r = data[:, 3]
        c = data[:, 4]
        #l = data[:, 5]
        xyz = data[:,0:3]

        #print("estamos en dataloader")
        #v = pptk.viewer(xyz, c)
        point_set = torch.from_numpy(xyz.astype(np.float32))  #pointset por ahora es xyz, podría ser todo



        cls = torch.from_numpy(np.array([c]).astype(np.int64))
        return point_set, cls
    
    def __len__(self):
        dir_path = self.root# r'split_de_paris/'
        return len([entry for entry in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, entry))])


#DATASET ANTIGUO VERSION COMO SHAPENET
"""

class ParisDataset(data.Dataset):
    
    def __init__(self,
             root,
             npoints=2500):

        self.npoints = npoints
        self.root = root   #Root es donde esta el dataset

        self.datapath = 'split_de_paris10clases/reduced10k/'  #/reduced' #BIG WARNINGINGINIGNIGNIG CAMBIAR ESTO EN TODOS LADOS #cambiar los datapath a root cuando funcione
        self.available = [file for file in os.listdir(self.datapath)]
        
        
    def __getitem__(self, index):
        
        #chosen_file = 'Paris_' + str(index).zfill(3) + '.npy'
        #data = np.load(str(self.root + chosen_file)) #+'/'+
        
        chosen_file = self.available[index]
        data = np.load(os.path.join(self.datapath, chosen_file))
        
        #x = data[:, 0]
        #y = data[:, 1]
        #z = data[:, 2]
        #r = data[:, 3]
        c = data[:, 4]
        #l = data[:, 5]

        xyz = data[:,0:3]

        #v = pptk.viewer(xyz, r, c, l)

        point_set = torch.from_numpy(xyz.astype(np.float32))  #pointset por ahora es xyz, podría ser todo

        cls = torch.from_numpy(np.array([c]).astype(np.int64))
        return point_set, cls
    
    def __len__(self):
        dir_path = self.root# r'split_de_paris/'
        return len([entry for entry in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, entry))])
"""