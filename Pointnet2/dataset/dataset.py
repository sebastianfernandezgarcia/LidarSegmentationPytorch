import os
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.datasets import ShapeNet
import torch

class Aerolaser(Dataset):

    def __init__(self, train_dir=r'./aerolaser', test_dir=r'./aerolaser_validation/', train=True, transform=None, npoints=4096):
        #(self, root, npoints=5000, datapath='split_de_paris10clases/reduced10k/normalized'):
        self.npoints = npoints
        #self.root_dir = root_dir
        print(self.npoints)
        if train:
            self.root = train_dir #r'./aerolaser' #_test/ #root_dir   #root_dir
        if train == False:
            self.root = test_dir#r'./aerolaser_test/' #_test/  
        self.available = [file for file in os.listdir(self.root)] #root_dir
        
    def __getitem__(self, index):
        #chosen_file = 'Paris_' + str(index).zfill(3) + '.npy'
        #data = np.load(str(self.root + chosen_file)) #+'/'+
        chosen_file = self.available[index]
        data = np.load(os.path.join(self.root, chosen_file))
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
        cls = np.transpose(cls)
        
        choice = np.random.choice(point_set.shape[0], self.npoints, replace=True)
        points, labels = point_set[choice, :], cls[choice]

        points, labels = point_set, cls
        
        sample = {
            'points': points,  # torch.Tensor (n, 3)
            'labels': labels   # torch.Tensor (n,)
        }
        return sample
    
    def __len__(self):
        dir_path = self.root# r'split_de_paris/'
        return len([entry for entry in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, entry))])
    
    def num_classes(self):
        return 8 #self.dataset.num_classes
