#import argparse
#import os
from plyfile import PlyData, PlyElement
import numpy as np
from sklearn.decomposition import PCA
#import pptk

import os
import numpy as np
import argparse
#import laspy
import math
from tqdm import tqdm
import random 
import shutil
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

import open3d as o3d
#import pyntcloud
import pandas as pd
#from pyntcloud import Voxelize
from scipy.spatial import cKDTree
import time

parser = argparse.ArgumentParser()

"""
parser.add_argument('--segment_point_size', type=int, default=131072, help='resample points number') #Choose here number of point resample
parser.add_argument('--dataset_type', type=str, choices = ('train', 'test'), default='train', help='if train, dataset just splited, downsampled, normalized, if test same, but also gives numbers to reconstruct and paint dataset later')
parser.add_argument('--output_dataset_folder', type=str, default=r'nuevaparticion/test/', help='train datasetfolder') #las_test
"""

parser.add_argument('--original_las', type=str, default=r'C:/Users/sfernandez/nueva_etapa/github2/LidarSegmentationPytorch/Datasets/ParisLille/training_10_classes', help='train datasetfolder') #las_test
parser.add_argument('--dataset_type', type=str, choices = ('train', 'test'), default='train', help='if train, dataset just splited, downsampled, normalized, if test same, but also gives numbers to reconstruct and paint dataset later')
opt = parser.parse_args()


#opt.original_las

def split_dataset(output_folder):
    print("Procesando nubes...")

    # Ruta de la carpeta para guardar los segmentos
    #output_folder = opt.output_dataset_folder

    # Crear la carpeta si no existe
    #os.makedirs(output_folder, exist_ok=True)

    for filename in tqdm(os.listdir(opt.original_las), desc ="Conjunto de Todas las nubes:"):
        f = os.path.join(opt.original_las, filename)
        if os.path.isfile(f):

            #print(f)

            plydata = PlyData.read(f) #print(plydata)

            x = plydata["vertex"].data["x"].astype(np.float32)
            y = plydata["vertex"].data["y"].astype(np.float32)
            z = plydata["vertex"].data["z"].astype(np.float32)

            point_data = np.stack([x, y, z], axis=0)

            vertex_data = plydata["vertex"].data

            print(vertex_data)

            nombres_campos = vertex_data.dtype.names
            print(nombres_campos)

            """ TEST
            #TEST - ('x', 'y', 'z', 'x_origin', 'y_origin', 'z_origin', 'GPS_time', 'reflectance')
            x_origin = plydata["vertex"].data["x_origin"].astype(np.float32)
            y_origin = plydata["vertex"].data["y_origin"].astype(np.float32)
            z_origin = plydata["vertex"].data["z_origin"].astype(np.float32)


            nueva_x = x-x_origin

            nueva_y = y-y_origin

            nueva_z = z-z_origin
            reflectance = plydata["vertex"].data["reflectance"].astype(np.float32)
            xyzic = np.stack([nueva_x, nueva_y, nueva_z, reflectance], axis=0).transpose((1, 0))

            """

            """
            print(max(x)-min(x))
            print("----")
            print(max(y)-min(y))
            print("----")
            print(max(z)-min(z))
            print("----")
            """
            
            
            #ESTO SI HAY QUE RESTAR 1 A LAS CLASESS
            #adapta_clases = las.classification
            #adapta_clases = np.subtract(adapta_clases, 1)

            reflectance = plydata["vertex"].data["reflectance"].astype(np.float32)

            label = plydata["vertex"].data["class"].astype(np.float32) #label solo si es train
            
            #print(plydata["vertex"].data)

            #last_column = [row[-1] for row in plydata["vertex"].data]
            #unique_elements = set(last_column)
            #print(unique_elements)
            #true_label = plydata["vertex"].data["label"].astype(np.float32) #label solo si es train
            #reflectance = label = plydata["vertex"].data["reflectance"].astype(np.float32) #label solo si es train

            #xyz = np.array([x,y,z]).T
            
            xyzic = np.stack([x, y, z, reflectance, label], axis=0).transpose((1, 0))
            #print(xyzic.shape)

            #print(xyzic)

            def borrar_ignorados(arr, valores_borrar, column_index):
                for value_to_delete in valores_borrar:
                    # Crear una máscara booleana para identificar las filas a eliminar
                    mask = arr[:, column_index] == value_to_delete
                    # Eliminar las filas que cumplen la condición
                    arr = np.delete(arr, np.where(mask), axis=0)
                    return arr

            nuevos_puntos = borrar_ignorados(xyzic, [0], 4) #0 borrar lo sin clasificacion.


            data = nuevos_puntos #xyzic


            base_filename = os.path.basename(f)
            # Create a pandas DataFrame from the data for easier manipulation
            df = pd.DataFrame(data, columns=['x', 'y', 'z', 'reflectance', 'class'])

            # Define grid size
            grid_size = 20.0

            # Calculate grid indices
            df['grid_x'] = (df['x'] / grid_size).apply(math.floor)
            df['grid_y'] = (df['y'] / grid_size).apply(math.floor)

            # Group by grid indices
            grouped = df.groupby(['grid_x', 'grid_y'])

            # Create 'segments' directory if it doesn't exist
            os.makedirs('train', exist_ok=True)

            # Loop over each group
            for i, (name, group) in enumerate(grouped):
                # Drop grid indices columns from group
                group = group.drop(['grid_x', 'grid_y'], axis=1)

                # Convert group back to numpy array
                group_array = group.to_numpy()

                # Save to .npy file, appending group name to filename
                output_filename = os.path.join('train', f'{base_filename}_{i}.npy')
                if len(group_array) > 1:
                    #print(output_filename)
                    np.save(output_filename, group_array)
            #time.sleep(10)


            """
            ###################

            x_min = np.min(xyzic[:, 0])
            x_max = np.max(xyzic[:, 0])
            y_min = np.min(xyzic[:, 1])
            y_max = np.max(xyzic[:, 1])

            divide_x = np.ceil((x_max - x_min) / 20).astype(int)
            divide_y = np.ceil((y_max - y_min) / 20).astype(int)

            count = 0
            for i in range(divide_x):
                for j in range(divide_y):
                    # Definir límites de segmento
                    x_min_segment = x_min + i * 20
                    x_max_segment = x_min + (i + 1) * 20
                    y_min_segment = y_min + j * 20
                    y_max_segment = y_min + (j + 1) * 20

                    # Filtrar puntos dentro del segmento
                    segment_points = xyzic[
                        (xyzic[:, 0] >= x_min_segment)
                        & (xyzic[:, 0] < x_max_segment)
                        & (xyzic[:, 1] >= y_min_segment)
                        & (xyzic[:, 1] < y_max_segment)
                    ]
                    count += 1

                    filename_seg = os.path.splitext(filename)[0]+f"_{str(count).zfill(3)}"
                    #filename = f"segment_{i}_{j}.npy"

                    filepath = os.path.join(output_folder, filename_seg)
                    if len(segment_points)>1:
                        np.save(filepath, segment_points)
            """

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
        outputfolder = r'train/'
        #Create the destination folders if they don't already exist
        os.makedirs(outputfolder, exist_ok=True)

        split_dataset(outputfolder)
        train_valid_split(outputfolder)

    if opt.dataset_type == 'test':
        outputfolder = r'test/'
        #Create the destination folders if they don't already exist
        os.makedirs(outputfolder, exist_ok=True)
        split_dataset(outputfolder)

    print("Finished Spliting, Downscaling and Normalized Dataset. You can find the results at: ", outputfolder)
