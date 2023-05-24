import os
import numpy as np
import argparse
import laspy
import math
import open3d as o3d


#en el parser: dataset_type
#segment_point_size (points to reduce)

#los datasets.... train las folder   test las folder
#y de ahi crea train valid y test

#SPLITER Y DOWNSAMPLE


parser = argparse.ArgumentParser()
parser.add_argument('--segment_point_size', type=int, default=16384, help='resample points number') #Choose here number of point resample
parser.add_argument('--dataset_type', type=str, choices = ('train', 'test'), default='train', help='if train, dataset just splited, downsampled, normalized, if test same, but also gives numbers to reconstruct and paint dataset later')
parser.add_argument('--original_las', type=str, default=r'las/', help='train datasetfolder') #las_test
parser.add_argument('--output_dataset_folder', type=str, default=r'dataset_final/', help='train datasetfolder') #las_test
opt = parser.parse_args()







save_dir = "test_split_normalized"
points_to_reduce = 16384                ####esto como parser

for filename in os.listdir(opt.original_las):
    f = os.path.join(opt.original_las, filename)
    if os.path.isfile(f):
        las = laspy.read(f)
        
        point_data = np.stack([las.x, las.y, las.z], axis=0) #.transpose((1, 0))


        #clases_a_reducir = np.array(las.classification)

        # Identify indices of points with classification not in [1,8]
        #indices_to_remove = np.where(np.logical_or(clases_a_reducir < 1, clases_a_reducir > 8))[0]

        # Remove points with classification not in [1,8] from the arrays
        #nuevos_puntos = np.delete(nuevos_puntos, indices_to_remove, axis=0)
        #xyzic = np.delete(xyzic, indices_to_remove, axis=0)

        #print(np.unique(clases_a_reducir))
        #xyz = np.stack([las.x, las.y, las.z], axis=0).transpose((1, 0))

        adapta_clases = las.classification
        adapta_clases = np.subtract(adapta_clases, 1)

        xyzic = np.stack([las.x, las.y, las.z, las.intensity, adapta_clases], axis=0).transpose((1, 0))
        xyz = np.stack([las.x, las.y, las.z], axis=0).transpose((1, 0))

        #xyzic = np.stack([las.x, las.y, las.z, las.intensity, las.classification], axis=0).transpose((1, 0))
        #once i have created this array i want to delete all the values which are not beteween 1 and 8 of the 4th colum which is las.classification and its corresponding positions to all the rows

        valid_rows = (xyzic[:, 4] >= 0) & (xyzic[:, 4] <= 7)

        # delete rows where classification is not between 1 and 8
        xyzic = np.delete(xyzic, np.where(~valid_rows), axis=0)

        def borrar_ignorados(arr, valores_borrar, column_index):
            #column_index = 4
            # Especificar el valor que quieres eliminar
            #valores_borrar = [0,15,21,27]
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

        len_split = 51  #25000  #the length you want to have in each fragment of new division of pointcloud

        divide_x_in = math.floor((x_max-x_min) / len_split) #cogemos el de arriba para asegurar tener todo. quiza los trozos no son de 50x50 sino 49.8x49.9ç
        divide_y_in = math.floor((y_max-y_min) / len_split) #cogemos el de arriba para asegurar tener todo. quiza los trozos no son de 50x50 sino 49.8x49.9ç

        print(divide_x_in)
        print(divide_y_in)
    
        x_grid = np.linspace(x_min, x_max, divide_x_in + 1)
        y_grid = np.linspace(y_min, y_max, divide_y_in + 1)

        count = 0

        for i in range(len(x_grid)-1):
            for j in range(len(y_grid)-1):
                if count <= len(y_grid)-1: 
                    mask_x = np.logical_and(nuevos_puntos[:,0] >= x_grid[i], nuevos_puntos[:,0] <= x_grid[i+1])
                else: 
                    mask_x = np.logical_and(nuevos_puntos[:,0] >= x_grid[i], nuevos_puntos[:,0] < x_grid[i+1])
                count += 1
                
                filtered_x = nuevos_puntos[mask_x]
                
                mask_y = np.logical_and(filtered_x[:,1] >= y_grid[j], filtered_x[:,1] < y_grid[j+1])
                filtered_y = filtered_x[mask_y]

                if(len(filtered_y)> points_to_reduce):

                    reduccion = np.random.choice(len(filtered_y), size=points_to_reduce, replace=True)
                    reducido = filtered_y[reduccion]
                    np.save(os.path.join(save_dir, os.path.splitext(filename)[0]+f"_{str(count).zfill(3)}"), reducido)
                else:
                    if(len(filtered_y)>0):
                        print(filtered_y)
                        print(filtered_y.shape)
                        np.save(os.path.join(save_dir, os.path.splitext(filename)[0]+f"_NO16384_{str(count).zfill(3)}"), filtered_y)
                    else:
                        print("NO SE GUARDO POR VACIO-> ", os.path.join(save_dir, os.path.splitext(filename)[0]+f"_NO16384_{str(count).zfill(3)}"))















################
#############



#NORMALIZACION CON RESCATE DE NORMA. PARA TRAIN, NO HACERLO, PARA TEST SI.
#en parametros parser... dataset_type=train, test
#en el train hacer el split tambien? siiii 80% train 20%valid





######################
#dataset_folder = 'split_de_paris10clases/reduced25k/'
#save_dir = 'split_de_paris10clases/reduced25k/normalized/'

dataset_folder = 'test_split_normalized/'  #'procesadas_main/'
save_dir = 'test_split_normalized/normalized_withtransform' 

#save_dir2 = 'procesadas_main/normalized/transformedarrays' 

for filename in os.listdir(dataset_folder):

    f = os.path.join(dataset_folder, filename)

    if os.path.isfile(f):

        data = np.load(f)

        x = data[:, 0]
        y = data[:, 1]
        z = data[:, 2]
        r = data[:, 3]
        c = data[:, 4]
        #l = data[:, 5]

        range_to_normalize = (0,1)
        x_normalized1 = (x-min(x))
        x_normalized = x_normalized1/max(x_normalized1)

        y_normalized1 = (y-min(y))
        y_normalized = y_normalized1/max(y_normalized1)

        z_normalized1 = (z-min(z))
        z_normalized = z_normalized1/100

        #guardar -> max(x_normalized), max(y_normalized), min(x), min(), min(z)

        transformada = np.array([max(x_normalized1), max(y_normalized1), min(x), min(y), min(z)]).T


        #x = x_normalized * max(x_normalized) + min(x)
        #y = y_normalized * max(y_normalized) + min(y)
        #z = z_normalized * 100 + min(z)

        
        #AQUI GUARDAS LOS VALORES NEDESARIOS 1 ficheros por cada nube. Despues recorres igual y puedes invertir... puede ser interesante.
        #completamente hacer inversa, sunar y multiplicar y deberia tirar...


        #x_normalized = #normalize(x, range_to_normalize[0], range_to_normalize[1])
        #y_normalized = #normalize(y, range_to_normalize[0], range_to_normalize[1])
        #z_normalized = #normalize(z, range_to_normalize[0], range_to_normalize[1])

        #xyzrc_normalized = np.array([[x_normalized,y_normalized,z_normalized,r,c, transformada]]).T #max(x_normalized), max(y_normalized), min(x), min(y), min(z)]).T
        #print(xyzrc_normalized)
        print(c.shape)
        #print(max(x_normalized1).shape)

        max_normalized_x_all_array = np.full(len(c), max(x_normalized1))
        max_normalized_y_all_array = np.full(len(c), max(y_normalized1))
        min_x_all_array = np.full(len(c), min(x))
        min_y_all_array = np.full(len(c), min(y))
        min_z_all_array = np.full(len(c), min(z))

        #xyzrc_normalized = np.array([x_normalized,y_normalized,z_normalized,r,c,max(x_normalized1), max(y_normalized1), min(x), min(y), min(z)]).T #max(x_normalized), max(y_normalized), min(x), min(y), min(z)]).T
        #xyzrc_normalized = np.array([[x_normalized[i], y_normalized[i], z_normalized[i], r[i], c[i], transformada[i]] for i in range(len(x_normalized))])
        xyzrc_normalized = np.array([x_normalized,y_normalized,z_normalized,r,c,max_normalized_x_all_array, max_normalized_y_all_array, min_x_all_array, min_y_all_array, min_z_all_array]).T #max(x_normalized), max(y_normalized), min(x), min(y), min(z)]).T
        
        print(max(y)-min(y))
        np.save(os.path.join(save_dir, os.path.splitext(filename)[0]+f"_{str(range_to_normalize[0])}{'-'}{str(range_to_normalize[1])}"), xyzrc_normalized)













#####

##TRAIN TEST SPLITTER

################################################
import os
import shutil
import random

# Set the path of the source folder containing the .npy files
source_folder = r'procesadas_main\normalized'

# Set the paths of the destination folders for the split
train_folder = r'procesadas_main\normalized\train'
val_folder = r'procesadas_main\normalized\validation'

# Set the percentage of files to be moved to the validation folder
validation_split = 0.2

# Create the destination folders if they don't already exist
os.makedirs(train_folder, exist_ok=True)
os.makedirs(val_folder, exist_ok=True)

# Get the list of .npy files in the source folder
file_list = [f for f in os.listdir(source_folder) if f.endswith('.npy')]

# Calculate the number of files to be moved to the validation folder
num_val_files = int(len(file_list) * validation_split)

# Randomly select the files to be moved to the validation folder
val_files = random.sample(file_list, num_val_files)

# Move the files to the train and validation folders
for file_name in file_list:
    if file_name in val_files:
        dest_folder = val_folder
    else:
        dest_folder = train_folder
    shutil.move(os.path.join(source_folder, file_name), os.path.join(dest_folder, file_name))
