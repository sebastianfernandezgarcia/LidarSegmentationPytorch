import pickle, time, warnings
import numpy as np

import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader, Sampler, BatchSampler

from utils.tools import Config as cfg
from utils.tools import DataProcessing as DP
import os
import random

class PointCloudsDataset(Dataset): 
      #def __init__(self, train_dir=r'./dataset_crudo_partido/train/', test_dir=r'./dataset_crudo_partido/validation/', train=True, transform=None, npoints=32768): #4096
    def __init__(self, train_dir=r'C:/Users/sfernandez/nueva_etapa/moviendoAlServidor/Pointnet2Aerolaser/nuevaparticion/train//train/procesados/', test_dir=r'C:/Users/sfernandez/nueva_etapa/moviendoAlServidor/Pointnet2Aerolaser/nuevaparticion/train/validation/procesados/', train=True, transform=None, npoints=4096): #4096
        #r'./dataset_final_pruebas_balanceo_2/train/train/'
        #(self, root, npoints=5000, datapath='split_de_paris10clases/reduced10k/normalized'):
        self.npoints = npoints
        #self.root_dir = root_dir
        #print(self.npoints)
        if train:
            self.root = train_dir #r'./aerolaser' #_test/ #root_dir   #root_dir
        if train == False:
            self.root = test_dir#r'./aerolaser_test/' #_test/  
        self.available = [file for file in os.listdir(self.root)] #root_dir

    """
    def __init__(self, dir, labels_available=True):
        self.paths = list(dir.glob(f'*.npy'))
        self.labels_available = labels_available
    """

    def __getitem__(self, idx):
        #path = self.paths[idx]

        #points, labels = self.load_npy(path)

                #chosen_file = 'Paris_' + str(index).zfill(3) + '.npy'
        #data = np.load(str(self.root + chosen_file)) #+'/'+
        chosen_file = self.available[idx]
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
        cls = cls - 1
        choice = np.random.choice(point_set.shape[0], self.npoints, replace=True)
        points, labels = point_set[choice, :], cls[choice]
        

        #equiparado de clase resta 1
        #cls = cls - 1
        #AHORA SE COGEN DIRECTAMENTE
        #points = point_set
        #labels = cls
        #points, labels = point_set, cls

        #points_tensor = torch.from_numpy(points).float()
        #labels_tensor = torch.from_numpy(labels).long()
        #print(points.shape)
        #print(labels.shape)
        return points, labels

    def __len__(self):
        dir_path = self.root# r'split_de_paris/'
        return len([entry for entry in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, entry))])
        #return len(self.paths)

    def load_npy(self, path):
        r"""
            load the point cloud and labels of the npy file located in path

            Args:
                path: str
                    path of the point cloud
                keep_zeros: bool (optional)
                    keep unclassified points
        """
        cloud_npy = np.load(path, mmap_mode='r').T
        points = cloud_npy[:,:-1] if self.labels_available else points

        if self.labels_available:
            labels = cloud_npy[:,-1]

            # balance training set
            points_list, labels_list = [], []
            for i in range(len(np.unique(labels))):
                try:
                    idx = np.random.choice(len(labels[labels==i]), 8000)
                    points_list.append(points[labels==i][idx])
                    labels_list.append(labels[labels==i][idx])
                except ValueError:
                    continue
            if points_list:
                points = np.stack(points_list)
                labels = np.stack(labels_list)
                labeled = labels>0
                points = points[labeled]
                labels = labels[labeled]

        return points, labels

class CloudsDataset(Dataset):
    def __init__(self, dir, data_type='npy'):
        self.path = dir
        self.paths = list(dir.glob(f'*.{data_type}'))
        self.size = len(self.paths)
        self.data_type = data_type
        self.input_trees = {'training': [], 'validation': []}
        self.input_colors = {'training': [], 'validation': []}
        self.input_labels = {'training': [], 'validation': []}
        self.input_names = {'training': [], 'validation': []}
        self.val_proj = []
        self.val_labels = []
        self.val_split = '1_'

        self.load_data()
        print('Size of training : ', len(self.input_colors['training']))
        print('Size of validation : ', len(self.input_colors['validation']))

    def load_data(self):
        for i, file_path in enumerate(self.paths):
            t0 = time.time()
            cloud_name = file_path.stem
            if self.val_split in cloud_name:
                cloud_split = 'validation'
            else:
                cloud_split = 'training'

            # Name of the input files
            kd_tree_file = self.path / '{:s}_KDTree.pkl'.format(cloud_name)
            sub_npy_file = self.path / '{:s}.npy'.format(cloud_name)

            data = np.load(sub_npy_file, mmap_mode='r').T

            sub_colors = data[:,3:6]
            sub_labels = data[:,-1].copy()

            # Read pkl with search tree
            with open(kd_tree_file, 'rb') as f:
                search_tree = pickle.load(f)

            # The points information is in tree.data
            self.input_trees[cloud_split].append(search_tree)
            self.input_colors[cloud_split].append(sub_colors)
            self.input_labels[cloud_split].append(sub_labels)
            self.input_names[cloud_split].append(cloud_name)

            size = sub_colors.shape[0] * 4 * 7
            print('{:s} {:.1f} MB loaded in {:.1f}s'.format(kd_tree_file.name, size * 1e-6, time.time() - t0))

        print('\nPreparing reprojected indices for testing')

        # Get validation and test reprojected indices

        for i, file_path in enumerate(self.paths):
            t0 = time.time()
            cloud_name = file_path.stem

            # Validation projection and labels
            if self.val_split in cloud_name:
                proj_file = self.path / '{:s}_proj.pkl'.format(cloud_name)
                with open(proj_file, 'rb') as f:
                    proj_idx, labels = pickle.load(f)

                self.val_proj += [proj_idx]
                self.val_labels += [labels]
                print('{:s} done in {:.1f}s'.format(cloud_name, time.time() - t0))

    def __getitem__(self, idx):
        pass

    def __len__(self):
        # Number of clouds
        return self.size


class ActiveLearningSampler(IterableDataset):

    def __init__(self, dataset, batch_size=6, split='training'):
        self.dataset = dataset
        self.split = split
        self.batch_size = batch_size
        self.possibility = {}
        self.min_possibility = {}

        if split == 'training':
            self.n_samples = cfg.train_steps
        else:
            self.n_samples = cfg.val_steps

        #Random initialisation for weights
        self.possibility[split] = []
        self.min_possibility[split] = []
        for i, tree in enumerate(self.dataset.input_colors[split]):
            self.possibility[split] += [np.random.rand(tree.data.shape[0]) * 1e-3]
            self.min_possibility[split] += [float(np.min(self.possibility[split][-1]))]

    def __iter__(self):
        return self.spatially_regular_gen()

    def __len__(self):
        return self.n_samples # not equal to the actual size of the dataset, but enable nice progress bars

    def spatially_regular_gen(self):
        # Choosing the least known point as center of a new cloud each time.

        for i in range(self.n_samples * self.batch_size):  # num_per_epoch
            # t0 = time.time()
            if cfg.sampling_type=='active_learning':
                # Generator loop

                # Choose a random cloud
                cloud_idx = int(np.argmin(self.min_possibility[self.split]))

                # choose the point with the minimum of possibility as query point
                point_ind = np.argmin(self.possibility[self.split][cloud_idx])

                # Get points from tree structure
                points = np.array(self.dataset.input_trees[self.split][cloud_idx].data, copy=False)

                # Center point of input region
                center_point = points[point_ind, :].reshape(1, -1)

                # Add noise to the center point
                noise = np.random.normal(scale=3.5 / 10, size=center_point.shape)
                pick_point = center_point + noise.astype(center_point.dtype)

                if len(points) < cfg.num_points:
                    queried_idx = self.dataset.input_trees[self.split][cloud_idx].query(pick_point, k=len(points))[1][0]
                else:
                    queried_idx = self.dataset.input_trees[self.split][cloud_idx].query(pick_point, k=cfg.num_points)[1][0]

                queried_idx = DP.shuffle_idx(queried_idx)
                # Collect points and colors
                queried_pc_xyz = points[queried_idx]
                queried_pc_xyz = queried_pc_xyz - pick_point
                queried_pc_colors = self.dataset.input_colors[self.split][cloud_idx][queried_idx]
                queried_pc_labels = self.dataset.input_labels[self.split][cloud_idx][queried_idx]

                dists = np.sum(np.square((points[queried_idx] - pick_point).astype(np.float32)), axis=1)
                delta = np.square(1 - dists / np.max(dists))
                self.possibility[self.split][cloud_idx][queried_idx] += delta
                self.min_possibility[self.split][cloud_idx] = float(np.min(self.possibility[self.split][cloud_idx]))

                if len(points) < cfg.num_points:
                    queried_pc_xyz, queried_pc_colors, queried_idx, queried_pc_labels = \
                        DP.data_aug(queried_pc_xyz, queried_pc_colors, queried_pc_labels, queried_idx, cfg.num_points)

            # Simple random choice of cloud and points in it
            elif cfg.sampling_type=='random':

                cloud_idx = np.random.choice(len(self.min_possibility[self.split]), 1)[0]
                points = np.array(self.dataset.input_trees[self.split][cloud_idx].data, copy=False)
                queried_idx = np.random.choice(len(self.dataset.input_trees[self.split][cloud_idx].data), cfg.num_points)
                queried_pc_xyz = points[queried_idx]
                queried_pc_colors = self.dataset.input_colors[self.split][cloud_idx][queried_idx]
                queried_pc_labels = self.dataset.input_labels[self.split][cloud_idx][queried_idx]

            queried_pc_xyz = torch.from_numpy(queried_pc_xyz).float()
            queried_pc_colors = torch.from_numpy(queried_pc_colors).float()
            queried_pc_labels = torch.from_numpy(queried_pc_labels).long()
            queried_idx = torch.from_numpy(queried_idx).float() # keep float here?
            cloud_idx = torch.from_numpy(np.array([cloud_idx], dtype=np.int32)).float()

            points = torch.cat( (queried_pc_xyz, queried_pc_colors), 1)

            yield points, queried_pc_labels


def data_loaders(dir, sampling_method='active_learning', **kwargs):
    if sampling_method == 'active_learning':
        dataset = CloudsDataset(dir / 'train')
        batch_size = kwargs.get('batch_size', 6)
        val_sampler = ActiveLearningSampler(
            dataset,
            batch_size=batch_size,
            split='validation'
        )
        train_sampler = ActiveLearningSampler(
            dataset,
            batch_size=batch_size,
            split='training'
        )
        return DataLoader(train_sampler, **kwargs), DataLoader(val_sampler, **kwargs)

    if sampling_method == 'naive':
        #train_dataset = PointCloudsDataset(dir / 'train') 
        print("Rango 0, 1")

        seed = 42  # Puedes elegir cualquier número entero como semilla

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        train_dataset = PointCloudsDataset(r'C:/Users/sfernandez/nueva_etapa/github2/LidarSegmentationPytorch/Datasets/ParisLille/train_10metros/train/procesados50000-0_1') #overfit50k-1/')#train/train/procesados50000-0_1/')    #('dataset_final/train/train/')  
        #>>>train_dataset = PointCloudsDataset(r'C:/Users/sfernandez/nueva_etapa/moviendoAlServidor/Pointnet2Aerolaser/nuevaparticion/train/train/procesados50-1_1/')     #('dataset_final/train/train/')      #('dataset_final_pruebas_balanceo_2/train/train/')
        #val_dataset = PointCloudsDataset(dir / 'validation') Datasets\Aerolaser\train\train\procesados1024-1_1
        #>>>>val_dataset = PointCloudsDataset(r'C:/Users/sfernandez/nueva_etapa/moviendoAlServidor/Pointnet2Aerolaser/nuevaparticion/train/validation/procesados50-1_1/')  #('dataset_final/train/validation/') #('dataset_final_pruebas_balanceo_2/train/validation/')
        val_dataset = PointCloudsDataset(r'C:/Users/sfernandez/nueva_etapa/github2/LidarSegmentationPytorch/Datasets/ParisLille/train_10metros/validation/procesados50000-0_1') #overfit50k-1/')#train/validation/procesados50000-0_1/') #('dataset_final/train/validation/') #
        test_dataset = PointCloudsDataset(r'C:/Users/sfernandez/nueva_etapa/github2/LidarSegmentationPytorch/Datasets/ParisLille/train_10metros/testFAKE/procesados50000-0_1') #overfit50k-1/')#test/procesados50000-0_1/') #('dataset_final/test/') #('dataset_final_pruebas_balanceo_2/test/')
        return DataLoader(train_dataset, shuffle=True, **kwargs), DataLoader(val_dataset, **kwargs), DataLoader(test_dataset, **kwargs)
        
    raise ValueError(f"Dataset sampling method '{sampling_method}' does not exist.")

import glob
def num_classes_metodo(train_dir):

    file_list = glob.glob(train_dir + "/*.npy")

    # Recorrer los archivos .npy
    unique_values = []

    for file_path in file_list:
        # Realizar operaciones con cada archivo .npy
        #print("Archivo:", file_path)
        data = np.load(os.path.join(file_path))
        c = data[:, 4]

        unique_c = np.unique(c)
        for value in unique_c:
            if not np.isin(value, unique_values):
                unique_values.append(value)
    print("El dataset pasado tiene ", len(unique_values), "clases")
    return len(unique_values)#10#8 #self.dataset.num_classes


if __name__ == '__main__':
    dataset = CloudsDataset('datasets/s3dis/subsampled/train')
    batch_sampler = ActiveLearningSampler(dataset)
    for data in batch_sampler:
        xyz, colors, labels, idx, cloud_idx = data
        print('Number of points:', len(xyz))
        print('Point position:', xyz[1])
        print('Color:', colors[1])
        print('Label:', labels[1])
        print('Index of point:', idx[1])
        print('Cloud index:', cloud_idx)
        break
