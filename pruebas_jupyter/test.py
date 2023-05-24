import os
import numpy as np
import argparse
import laspy
import math
from tqdm import tqdm
import random 
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--segment_point_size', type=int, default=16384, help='resample points number') #Choose here number of point resample
parser.add_argument('--dataset_type', type=str, choices = ('train', 'test'), default='train', help='if train, dataset just splited, downsampled, normalized, if test same, but also gives numbers to reconstruct and paint dataset later')
parser.add_argument('--original_las', type=str, default=r'las/', help='train datasetfolder') #las_test
parser.add_argument('--output_dataset_folder', type=str, default=r'dataset_final/', help='train datasetfolder') #las_test
opt = parser.parse_args()

# Create the destination folders if they don't already exist
os.makedirs(opt.output_dataset_folder, exist_ok=True)

#Create test folder
test_dataset_folder = opt.output_dataset_folder+'test/'
os.makedirs(test_dataset_folder, exist_ok=True)

#Create Train folder
train_dataset_folder = opt.output_dataset_folder+'train/'
os.makedirs(train_dataset_folder, exist_ok=True)

def split_dataset():
    print("Procesando nubes...")
    for filename in tqdm(os.listdir(opt.original_las), desc ="Conjunto de Todas las nubes:"):
        f = os.path.join(opt.original_las, filename)
        if os.path.isfile(f):

            las = laspy.read(f)
            point_data = np.stack([las.x, las.y, las.z], axis=0)
            
            adapta_clases = las.classification
            adapta_clases = np.subtract(adapta_clases, 1)

            xyzic = np.stack([las.x, las.y, las.z, las.intensity, adapta_clases], axis=0).transpose((1, 0))
            xyz = np.stack([las.x, las.y, las.z], axis=0).transpose((1, 0))

            valid_rows = (xyzic[:, 4] >= 0) & (xyzic[:, 4] <= 7)

            xyzic = np.delete(xyzic, np.where(~valid_rows), axis=0)

            def borrar_ignorados(arr, valores_borrar, column_index):
                for value_to_delete in valores_borrar:
                    # Crear una máscara booleana para identificar las filas a eliminar
                    mask = arr[:, column_index] == value_to_delete
                    # Eliminar las filas que cumplen la condición
                    arr = np.delete(arr, np.where(mask), axis=0)
                    return arr

            nuevos_puntos = borrar_ignorados(xyzic, [10,15,21,27,255], 4) #0

            all_x = point_data[0]
            all_y = point_data[1]
            all_z = point_data[2]

            x_min = np.memmap.min(all_x)
            x_max = np.memmap.max(all_x)
            y_min = np.memmap.min(all_y)
            y_max = np.memmap.max(all_y)
            z_min = np.memmap.min(all_z)
            z_max = np.memmap.max(all_z)

            len_split = 51 #the length you want to have in each fragment of new division of pointcloud

            divide_x_in = math.floor((x_max-x_min) / len_split) #cogemos el de arriba para asegurar tener todo. quiza los trozos no son de 50x50 sino 49.8x49.9
            divide_y_in = math.floor((y_max-y_min) / len_split) 

            x_grid = np.linspace(x_min, x_max, divide_x_in + 1)
            y_grid = np.linspace(y_min, y_max, divide_y_in + 1)

            count = 0

            for i in tqdm(range(len(x_grid)-1), desc ="Nube actual"):
                for j in range(len(y_grid)-1):
                    if count <= len(y_grid)-1: 
                        mask_x = np.logical_and(nuevos_puntos[:,0] >= x_grid[i], nuevos_puntos[:,0] <= x_grid[i+1])
                    else: 
                        mask_x = np.logical_and(nuevos_puntos[:,0] >= x_grid[i], nuevos_puntos[:,0] < x_grid[i+1])
                    count += 1
                    
                    filtered_x = nuevos_puntos[mask_x]
                    
                    mask_y = np.logical_and(filtered_x[:,1] >= y_grid[j], filtered_x[:,1] < y_grid[j+1])
                    filtered_y = filtered_x[mask_y]
                    filename_seg = os.path.splitext(filename)[0]+f"_{str(count).zfill(3)}"
                    if(len(filtered_y)> opt.segment_point_size):
                        reduccion = np.random.choice(len(filtered_y), size=opt.segment_point_size, replace=True)
                        reducido = filtered_y[reduccion]
                        normalize_dataset(reducido, filename_seg)
                        #np.save(os.path.join(opt.output_dataset_folder, os.path.splitext(filename)[0]+f"_{str(count).zfill(3)}"), reducido)
                    else:
                        if(len(filtered_y)>0):
                            normalize_dataset(filtered_y, filename_seg)
                            #np.save(os.path.join(opt.output_dataset_folder, os.path.splitext(filename)[0]+f"_NO16384_{str(count).zfill(3)}"), filtered_y)
                        #else:
                            #print("Este trozo no se guardó porque tenía 0 puntos.")

def normalize_dataset(data, filename_seg):
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

    if(opt.dataset_type == 'test'):

        max_normalized_x_all_array = np.full(len(c), max(x_normalized1))
        max_normalized_y_all_array = np.full(len(c), max(y_normalized1))
        min_x_all_array = np.full(len(c), min(x))
        min_y_all_array = np.full(len(c), min(y))
        min_z_all_array = np.full(len(c), min(z))

        xyzrc_normalized = np.array([x_normalized,y_normalized,z_normalized,r,c,
                                     max_normalized_x_all_array, max_normalized_y_all_array, 
                                     min_x_all_array, min_y_all_array, min_z_all_array]).T #max(x_normalized), max(y_normalized), min(x), min(y), min(z)]).T
        
        np.save(os.path.join(test_dataset_folder, filename_seg), xyzrc_normalized)

    if(opt.dataset_type == 'train'):

        xyzrc_normalized = np.array([x_normalized,y_normalized,z_normalized,r,c]).T #max(x_normalized), max(y_normalized), min(x), min(y), min(z)]).T

        np.save(os.path.join(train_dataset_folder, filename_seg), xyzrc_normalized)

def train_valid_split():

    # Set the percentage of files to be moved to the validation folder
    validation_split = 0.2

    source_folder = train_dataset_folder

    # Create the destination folders if they don't already exist
    train_folder = train_dataset_folder + "train"
    validation_folder = train_dataset_folder + "validation"

    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(validation_folder, exist_ok=True)

    # Get the list of .npy files in the source folder
    file_list = [f for f in os.listdir(source_folder) if f.endswith('.npy')]

    # Calculate the number of files to be moved to the validation folder
    num_val_files = int(len(file_list) * validation_split)

    # Randomly select the files to be moved to the validation folder
    val_files = random.sample(file_list, num_val_files)

    # Move the files to the train and validation folders
    for file_name in file_list:
        if file_name in val_files:
            dest_folder = validation_folder
        else:
            dest_folder = train_folder
        shutil.move(os.path.join(source_folder, file_name), os.path.join(dest_folder, file_name))

if __name__ == "__main__":
    
    split_dataset()

    if(opt.dataset_type == 'train'):
        train_valid_split()

    print("Finished Spliting, Downscaling and Normalized Dataset. You can find the results at: ", opt.output_dataset_folder)
