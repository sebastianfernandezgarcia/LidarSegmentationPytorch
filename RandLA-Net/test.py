import numpy as np
from pathlib import Path
import time

import torch
import torch.nn as nn

from data import data_loaders
from data_test import data_loaders_original

from model import RandLANet
from utils.ply import read_ply, write_ply

from vis.view_copy import view_points_labels
import tqdm

t0 = time.time()

#path = Path('datasets') / 's3dis' / 'subsampled' / 'test'
path = r'dataset_final' #r'dataset_final_pruebas_balanceo_2/'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print('Loading data...')
#loader, val_loader, test_loader = data_loaders(path, 'naive')

test_loader_original = data_loaders_original(path, 'naive')
#print(loader)

print('Loading model...')

d_in = 3
num_classes = 8 #14

model = RandLANet(d_in, num_classes, 16, 4, device)
model.load_state_dict(torch.load('runs\checkpoint_1000.pth')['model_state_dict'])  #'runs/2020-04-11_17:03/checkpoint_10.pth'
model.eval()

"""
points, labels = next(iter(loader))

print('Predicting labels...')
with torch.no_grad():
    points = points.to(device)
    labels = labels.to(device)
    scores = model(points)
    predictions = torch.max(scores, dim=-2).indices
    accuracy = (predictions == labels).float().mean() # TODO: compute mIoU usw.
    print('Accuracy:', accuracy.item())
    predictions = predictions.cpu().numpy()
"""

print('Predicting labels...')
total_accuracy = 0.0
total_samples = 0

all_pred_labels = np.empty((0,), dtype=int) # inicializar el array vacío
all_gt_labels = np.empty((0,),  dtype=int) # inicializar el array vacío
all_diff_labels =  np.empty((0,),  dtype=int) 

all_points = np.empty((0, 3))
all_original_points = np.empty((0, 3))

contador = 0
print(len(test_loader_original))
with torch.no_grad():

    for points, labels, original_points in test_loader_original:

        contador+=1
        points = points.to(device)
        labels = labels.to(device)
        scores = model(points)
        predictions = torch.max(scores, dim=-2).indices
        #accuracy = (predictions == labels).float().mean()
        
        #total_accuracy += accuracy.item() * points.size(0)
        total_samples += points.size(0)
        
        predictions = predictions.cpu().numpy().squeeze() #el esquieze es para que pase de 4096,1  a solo 4096
        #print(predictions)
        #print(predictions.shape)
        p_points = points.squeeze() #.permute(1, 0)

        #print("____________")
        #print(p_points.cpu().numpy())
        #print(p_points.cpu().numpy().shape)
        #time.sleep(50)

        all_pred_labels = np.concatenate((all_pred_labels, predictions), axis=0)
        all_points = np.concatenate((all_points, p_points.cpu().numpy()), axis=0)


        #print(original_points.squeeze())
        #print(original_points.squeeze().shape)
        #time.sleep(40)
        all_original_points = np.concatenate((all_original_points, original_points.squeeze()), axis=0)

        print(contador)

print("labels")
print(all_pred_labels)
print(all_pred_labels.shape)
print("trandofrmed points")
print("---")
print(all_points)
print(all_points.shape)
print("original points")
print("---")
print(all_original_points)
print(all_original_points.shape)

average_accuracy = total_accuracy / total_samples
print('Average Accuracy:', average_accuracy)


view_points_labels(all_original_points, all_pred_labels, all_pred_labels)



"""
#time.sleep(500)
#time.sleep(30)
####################################################################
print('Writing results...')
np.savetxt('output.txt', predictions, fmt='%d', delimiter='\n')


t1 = time.time()
# write point cloud with classes
print('Assigning labels to the point cloud...')
print(points)
print("--------------")
cloud = points.squeeze(0)[:,:3]

print(points[0])

print("------------")

print(cloud.shape)
print("------------")

print(predictions.shape)

view_points_labels(cloud, predictions, predictions) #all_original_points, all_pred_labels, all_gt_labels

time.sleep(39)
write_ply('MiniDijon9.ply', [cloud, predictions], ['x', 'y', 'z', 'class'])

print('Done. Time elapsed: {:.1f}s'.format(t1-t0))
"""