import numpy as np
import os
import time 
import random
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--split', type=str, choices = ('train', 'validation', 'test'), required=True, help='if train, dataset is splited, downsampled, normalized, if test same, but also gives numbers to reconstruct and paint dataset later')
parser.add_argument('--range', type=int, choices = (-1, 0), required=True, help='normaliza entre 0 y 1 o -1,1')
parser.add_argument('--puntos_finales', type=int, choices = (1024, 2048, 4096, 16384, 32768, 50000), required=True, help='puntos que tendrá cada fichero final')
opt = parser.parse_args()

def normalize_dataset(data):
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    r = data[:, 3]
    c = data[:, 4]

    x_normalized1 = (x-min(x))
    x_normalized = x_normalized1/max(x_normalized1)

    y_normalized1 = (y-min(y))
    y_normalized = y_normalized1/max(y_normalized1)

    z_normalized1 = (z-min(z))
    z_normalized = z_normalized1/100

    return return_normalized(x_normalized1, y_normalized1, x_normalized, y_normalized, z_normalized, r, c)

def normalize_dataset_1_1(data):
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    r = data[:, 3]
    c = data[:, 4]

    x_normalized1 = (x - min(x))
    x_normalized2 = x_normalized1 / max(x_normalized1)
    x_normalized = 2 * x_normalized2 - 1

    y_normalized1 = (y - min(y))
    y_normalized2 = y_normalized1 / max(y_normalized1)
    y_normalized = 2 * y_normalized2 - 1

    z_normalized1 = (z - min(z))
    z_normalized2 = z_normalized1 / 100
    z_normalized = 2 * z_normalized2 - 1

    return return_normalized(x_normalized1, y_normalized1, x_normalized, y_normalized, z_normalized, r, c)

def return_normalized(x_normalized1, y_normalized1, x_normalized, y_normalized, z_normalized, r, c):
    if(opt.split == 'test'):
        
        max_normalized_x_all_array = np.full(len(c), max(x_normalized1))
        max_normalized_y_all_array = np.full(len(c), max(y_normalized1))
        min_x_all_array = np.full(len(c), min(x))
        min_y_all_array = np.full(len(c), min(y))
        min_z_all_array = np.full(len(c), min(z))

        xyzrc_normalized = np.array([x_normalized,y_normalized,z_normalized,r,c,
                                    max_normalized_x_all_array, max_normalized_y_all_array, 
                                    min_x_all_array, min_y_all_array, min_z_all_array]).T #max(x_normalized), max(y_normalized), min(x), min(y), min(z)]).T
        
        #np.save(os.path.join(test_dataset_folder, filename_seg), xyzrc_normalized)
        return xyzrc_normalized
    
    if(opt.split == 'train' or opt.split == 'validation'):

        xyzrc_normalized = np.array([x_normalized,y_normalized,z_normalized,r,c]).T #max(x_normalized), max(y_normalized), min(x), min(y), min(z)]).T
        return xyzrc_normalized
        #np.save(os.path.join(train_dataset_folder, filename_seg), xyzrc_normalized)



if opt.split == 'train':
    print("Procesando Train")

    las_dir = 'train/train/solopartidos/' #origen
    las_dir = r'C:/Users/sfernandez/nueva_etapa/github/Datasets/Aerolaser/train/train/solopartidos/'
    las_dir = r'C:/Users/sfernandez/nueva_etapa/github2/LidarSegmentationPytorch/Datasets/Aerolaser20metros/train/train/'
    #Para guardar
    if(opt.range == 0):
        save_dir = 'train/train/procesados' + str(opt.puntos_finales) + '-' + str(opt.range) + '_' + str(opt.range+1) + '/'
    if(opt.range == -1):
        save_dir = 'train/train/procesados' + str(opt.puntos_finales) + str(opt.range) + '_' + str(opt.range+2) + '/'
    
if opt.split == 'validation':
    print("Procesando Validation")

    las_dir = 'train/validation/solopartidos/' #origen
    las_dir = r'C:/Users/sfernandez/nueva_etapa/github/Datasets/Aerolaser/train/validation/solopartidos/'
    las_dir = r'C:/Users/sfernandez/nueva_etapa/github2/LidarSegmentationPytorch/Datasets/Aerolaser20metros/train/validation/'

    if(opt.range == 0):
        save_dir = 'train/validation/procesados' + str(opt.puntos_finales) + '-' + str(opt.range) + '_' + str(opt.range+1) + '/'
    if(opt.range == -1):
        save_dir = 'train/validation/procesados' + str(opt.puntos_finales)  + str(opt.range) + '_' + str(opt.range+2) + '/'

if opt.split == 'test':
    print("Procesando Test")

    las_dir = 'test/solopartidos/' #origen
    las_dir = r'C:/Users/sfernandez/nueva_etapa/github/Datasets/Aerolaser/test/solopartidos/'
    las_dir = r'C:/Users/sfernandez/nueva_etapa/github2/LidarSegmentationPytorch/Datasets/Aerolaser20metros/test/'
    #las_dir = 'train/train/solopartidos/' #origen
    if(opt.range == 0):
        save_dir = 'test/procesados' + str(opt.puntos_finales) + '-' + str(opt.range) + '_' + str(opt.range+1) + '/'
    if(opt.range == -1):
        save_dir = 'test/procesados' + str(opt.puntos_finales) + str(opt.range) + '_' + str(opt.range+2) + '/'

datos_fragmentos = []

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for filename in tqdm(os.listdir(las_dir)):
    if filename.endswith('.npy'):
        #data = np.load(las_dir+filename)

        nubetemp = np.load(las_dir+filename)
        #print(nubetemp)
        #time.sleep(10)

        npoints = opt.puntos_finales
        choice = np.random.choice(nubetemp.shape[0], npoints, replace=True)
        randomNube = nubetemp[choice, :]#, c[choice]

        #####

        x = nubetemp[:, 0]
        y = nubetemp[:, 1]
        z = nubetemp[:, 2]

        voxel_size = 1  # Adjust this value based on your requirements
        voxel_indices = ((x / voxel_size).astype(int), (y / voxel_size).astype(int), (z / voxel_size).astype(int))

        voxel_dict = {}
        for i in range(nubetemp.shape[0]):
            voxel_index = voxel_indices[0][i], voxel_indices[1][i], voxel_indices[2][i]
            if voxel_index in voxel_dict:
                voxel_dict[voxel_index].append(nubetemp[i])
            else:
                voxel_dict[voxel_index] = [nubetemp[i]]

        fixed_num_points = opt.puntos_finales #16384  # Adjust this value based on your requirements
        num_points_per_voxel = int(np.ceil(fixed_num_points / len(voxel_dict)))

        downsampled_points = []
        voxel_indices = set(zip(*voxel_indices))

        num_points_per_voxel = int(np.ceil(fixed_num_points / len(voxel_indices)))

        for voxel_index in voxel_indices:
            voxel_points = voxel_dict.get(voxel_index, [])
            if len(voxel_points) <= num_points_per_voxel:
                downsampled_points.extend(voxel_points)
            else:
                random_indices = random.sample(range(len(voxel_points)), num_points_per_voxel)
                downsampled_points.extend(np.array(voxel_points)[random_indices])

        downsampled_points = np.array(downsampled_points[:fixed_num_points])
        

        #puntos_normalizados = normalize_dataset_1_1(downsampled_points)

        #np.save(save_dir + filename, puntos_normalizados)

     
        if len(downsampled_points)>16: #que haya mas de 16 puntos arbitrariamente
            
            if(opt.range == 0):
                puntos_normalizados = normalize_dataset(downsampled_points)
            if(opt.range == -1):
                puntos_normalizados = normalize_dataset_1_1(downsampled_points)

            #Guardar random para duplicar puntos 
            indices = np.random.choice(puntos_normalizados.shape[0], size=opt.puntos_finales, replace=True)
            resized = puntos_normalizados = puntos_normalizados[indices]
            #resized = puntos_normalizados
            np.save(save_dir + filename, resized)
    