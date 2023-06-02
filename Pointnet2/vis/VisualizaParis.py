# Warning: import open3d may lead crash, try to import open3d first here
from view_copy import view_points_labels

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')  # add project root directory

from dataset.test_dataset import Aerolaser_Test
from model.pointnet2_part_seg import PointNet2PartSegmentNet
import torch_geometric.transforms as GT
import torch
import numpy as np
import argparse
import random
import time

##
parser = argparse.ArgumentParser()
#parser.add_argument('--dataset', type=str, choices=[r'../paris_faketest/procesados4096-0_1/', r'../aerolaser_test/'], required=True, help='esta puestoas las optiones al lado de parser')
parser.add_argument('--dataset', type=str, default=r'../paris_faketest/procesados4096-0_1/', help='Dataset que se carga')
parser.add_argument('--category', type=str, default='Airplane', help='select category')
parser.add_argument('--npoints', type=int, default=2500, help='resample points number')
parser.add_argument('--model', type=str, default='../checkpoint/checkpointParis.pt', help='model path') #cambiar el nombre de los pesos, pongo 19 porque voy a haver epoc 20
parser.add_argument('--sample_idx', type=int, default=0, help='select a sample to segment and view result')

opt = parser.parse_args()
print(opt)


## Load dataset
print('Construct dataset ..')
test_transform = GT.Compose([GT.NormalizeScale(),])

test_dataset = Aerolaser_Test(
    train_dir=opt.dataset,
    test_dir=opt.dataset,
    train=False,   #realmente estaba a false para visaluzar sobre test claramente..... pero pongo true para pintar torretas bonitas
    transform=test_transform,
    npoints=opt.npoints
)
num_classes = test_dataset.num_classes(opt.dataset)

print('test dataset size: ', len(test_dataset))
print('num_classes: ', num_classes)


# Load model
print('Construct model ..')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
dtype = torch.float

# net = PointNetPartSegmentNet(num_classes)
net = PointNet2PartSegmentNet(num_classes)

net.load_state_dict(torch.load(opt.model))
net = net.to(device, dtype)
net.eval()


##
def eval_sample(net, sample):
    '''
    sample: { 'points': tensor(n, 3), 'labels': tensor(n,) }
    return: (pred_label, gt_label) with labels shape (n,)
    '''
    net.eval()
    with torch.no_grad():
        # points: (n, 3)
        points, gt_label = sample['points'], sample['labels']
        n = points.shape[0]

        points = points.view(1, n, 3)  # make a batch
        points = points.transpose(1, 2).contiguous()
        points = points.to(device, dtype)

        pred = net(points)  # (batch_size, n, num_classes)
        pred_label = pred.max(2)[1]
        pred_label = pred_label.view(-1).cpu()  # (n,)
        
        #print("pred_label shape", pred_label.shape)              #OYE MUCHAAAA QUE CAMBIE ESTO PARA QUE TIRE REVISA ESTOOOOOOOOOOOOOOOO
        prueba = gt_label.ravel()
        #print("gt_label shape", prueba.shape)
        #time.sleep(60)
        assert pred_label.shape == prueba.shape
        return (pred_label, prueba)
    
    
        

def compute_mIoU(pred_label, gt_label):
    minl, maxl = np.min(gt_label), np.max(gt_label)
    ious = []
    for l in range(minl, maxl+1):
        I = np.sum(np.logical_and(pred_label == l, gt_label == l))
        U = np.sum(np.logical_or(pred_label == l, gt_label == l))
        if U == 0: iou = 1 
        else: iou = float(I) / U
        ious.append(iou)
    return np.mean(ious)


def label_diff(pred_label, gt_label):
    '''
    Assign 1 if different label, or 0 if same label  
    '''
    diff = pred_label - gt_label
    diff_mask = (diff != 0)

    diff_label = np.zeros((pred_label.shape[0]), dtype=np.int32)
    diff_label[diff_mask] = 1

    return diff_label
    

# Generar 20 valores aleatorios
valores_aleatorios = random.sample(range(100), 5)
#print(valores_aleatorios)
# Iterar sobre los valores aleatorios

#Todos SOlo final grande
###################RANDOMS EVALUANDO CADA SEGMENTO
#for i in valores_aleatorios:



all_pred_labels = np.empty((0,), dtype=int) # inicializar el array vacío
all_gt_labels = np.empty((0,),  dtype=int) # inicializar el array vacío
all_diff_labels =  np.empty((0,),  dtype=int) 

all_points = np.empty((0, 3))
all_original_points = np.empty((0, 3))

for i in range(len(test_dataset)):
    #print(i)

    #sample = test_dataset[opt.sample_idx]
    sample = test_dataset[i]

    import time
    #print(sample.shape)
    #time.sleep(20)
    #print('Eval test sample ..')
    pred_label, gt_label = eval_sample(net, sample)
    #print('Eval done ..')


    # Get sample result
    #print('Compute mIoU ..')
    points = sample['points'].numpy()

    original_points = sample['original_points']

    pred_labels = pred_label.numpy()



    gt_labels = gt_label.numpy()
    diff_labels = label_diff(pred_labels, gt_labels)


    all_points = np.concatenate((all_points, points), axis=0) 
    all_original_points = np.concatenate((all_original_points, original_points), axis=0)
    all_pred_labels = np.concatenate((all_pred_labels, pred_labels), axis=0)
    all_gt_labels = np.concatenate((all_gt_labels, gt_labels), axis=0)
    all_diff_labels = np.concatenate((all_diff_labels, diff_labels), axis=0) 

#Sobre todo el conjunto
print('mIoU: ', compute_mIoU(all_pred_labels, all_gt_labels))


    # View result

print('View gt labels ..')
view_points_labels(all_original_points, all_gt_labels, all_gt_labels, tipodataset=False)

print('View diff labels ..')
view_points_labels(all_original_points, all_diff_labels, all_gt_labels, diff=True, tipodataset=False)

print('View pred labels ..')
view_points_labels(all_original_points, all_pred_labels, all_gt_labels, tipodataset=False)





###################RANDOMS EVALUANDO CADA SEGMENTO
"""
for i in valores_aleatorios:
    print(i)

    #sample = test_dataset[opt.sample_idx]
    sample = test_dataset[i]
    print('Eval test sample ..')
    pred_label, gt_label = eval_sample(net, sample)
    print('Eval done ..')


    # Get sample result
    print('Compute mIoU ..')
    points = sample['points'].numpy()
    pred_labels = pred_label.numpy()
    gt_labels = gt_label.numpy()
    diff_labels = label_diff(pred_labels, gt_labels)

    print('mIoU: ', compute_mIoU(pred_labels, gt_labels))


    # View result

    # print('View gt labels ..')
    # view_points_labels(points, gt_labels)

    # print('View diff labels ..')
    # view_points_labels(points, diff_labels)

    print('View pred labels ..')
    view_points_labels(points, pred_labels)
"""
