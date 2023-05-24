import os
import numpy as np
import argparse
import laspy
import math
from tqdm import tqdm
import random 
import shutil
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

import open3d as o3d
import pyntcloud
import pandas as pd
#from pyntcloud import Voxelize
from scipy.spatial import cKDTree


parser = argparse.ArgumentParser()

"""
parser.add_argument('--segment_point_size', type=int, default=131072, help='resample points number') #Choose here number of point resample
parser.add_argument('--dataset_type', type=str, choices = ('train', 'test'), default='train', help='if train, dataset just splited, downsampled, normalized, if test same, but also gives numbers to reconstruct and paint dataset later')
parser.add_argument('--output_dataset_folder', type=str, default=r'nuevaparticion/test/', help='train datasetfolder') #las_test
"""

parser.add_argument('--original_las', type=str, default=r'raws/train/', help='train datasetfolder') #las_test
parser.add_argument('--dataset_type', type=str, choices = ('train', 'test'), default='train', help='if train, dataset just splited, downsampled, normalized, if test same, but also gives numbers to reconstruct and paint dataset later')
opt = parser.parse_args()




def split_dataset(output_folder):
    print("Procesando nubes...")

    # Ruta de la carpeta para guardar los segmentos
    #output_folder = opt.output_dataset_folder

    # Crear la carpeta si no existe
    #os.makedirs(output_folder, exist_ok=True)

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

            x_min = np.min(nuevos_puntos[:, 0])
            x_max = np.max(nuevos_puntos[:, 0])
            y_min = np.min(nuevos_puntos[:, 1])
            y_max = np.max(nuevos_puntos[:, 1])

            divide_x = np.ceil((x_max - x_min) / 51).astype(int)
            divide_y = np.ceil((y_max - y_min) / 51).astype(int)

            count = 0
            for i in range(divide_x):
                for j in range(divide_y):
                    # Definir límites de segmento
                    x_min_segment = x_min + i * 51
                    x_max_segment = x_min + (i + 1) * 51
                    y_min_segment = y_min + j * 51
                    y_max_segment = y_min + (j + 1) * 51

                    # Filtrar puntos dentro del segmento
                    segment_points = nuevos_puntos[
                        (nuevos_puntos[:, 0] >= x_min_segment)
                        & (nuevos_puntos[:, 0] < x_max_segment)
                        & (nuevos_puntos[:, 1] >= y_min_segment)
                        & (nuevos_puntos[:, 1] < y_max_segment)
                    ]
                    count += 1

                    filename_seg = os.path.splitext(filename)[0]+f"_{str(count).zfill(3)}"
                    #filename = f"segment_{i}_{j}.npy"

                    filepath = os.path.join(output_folder, filename_seg)
                    if len(segment_points)>1:
                        np.save(filepath, segment_points)


def train_valid_split(source_folder):

    # Set the percentage of files to be moved to the validation folder
    validation_split = 0.2

    #train_dataset_folder = opt.output_dataset_folder+'train/'
    #source_folder = opt.output_dataset_folder #train_dataset_folder
    
    # Create the destination folders if they don't already exist
    #train_folder = train_dataset_folder + "train"
    #validation_folder = train_dataset_folder + "validation"

    train_folder = source_folder + "/train"
    validation_folder = source_folder + "/validation"

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

    if(opt.dataset_type == 'train'):
        outputfolder = r'nuevaparticion/train/'
        #Create the destination folders if they don't already exist
        os.makedirs(outputfolder, exist_ok=True)

        split_dataset(outputfolder)
        train_valid_split(outputfolder)

    if opt.dataset_type == 'test':
        outputfolder = r'nuevaparticion/test/'
        #Create the destination folders if they don't already exist
        os.makedirs(outputfolder, exist_ok=True)
        split_dataset(outputfolder)

    print("Finished Spliting, Downscaling and Normalized Dataset. You can find the results at: ", outputfolder)
