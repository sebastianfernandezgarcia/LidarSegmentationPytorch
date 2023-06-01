from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
#from pointnet.dataset import ShapeNetDataset
from pointnet.mi_dataset import ParisDataset


from pointnet.model import PointNetDenseCls, feature_transform_regularizer
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix
from operator import truediv
import matplotlib.pyplot as plt
import random


#AHORA MISMO ES COPIA PEGA DEL FINALIZADO DE POINTNET SHAPE QUE YA HICE, MODIFICAR PARA QUE SE TRAGUE PARIS

#ya funcionaria para 5o10k creo el que tiene el split de train y test... pues quedaría adaptarlo con lo que se decida hacer 
#usando como referencia el train segmentarion.ipynb porque ahi furula perfecto

def main(opt):

    print('\n', opt, '\n')

    if(opt.process=='train'):
        train_test(opt)
        
    if(opt.process=='test'):
        test(opt)
    
def parse_opt(known=False):

    parser = argparse.ArgumentParser()

    parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
    parser.add_argument('--nepoch', type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--outf', type=str, default='pesos_segmentación/', help='output folder')
    parser.add_argument('--model', type=str, default='', help='model path')
    parser.add_argument('--dataset', type=str, required=True, help="dataset path")
    #parser.add_argument('--class_choice', type=str, default='Chair', help="class_choice")
    parser.add_argument('--feature_transform', action='store_true', help="use feature transform")
    parser.add_argument('--process', type=str, choices=['train', 'test'], default='train', help='choose if trains or test with pre-trained weights')
    parser.add_argument('--weights', type=str, default='seg/seg_model_Chair_9.pth', help='path to your trained weights for test')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()


    train_data = opt.dataset + 'train/'
    print("train data-> ",train_data)
    test_data = opt.dataset + 'test/'
    print("test data-> ",test_data)

    
    
    if (opt.process == 'test'):
        if (os.path.isfile(opt.weights) == False):
            print("\nWarning, your arguments weights might not exists or the path is wrong. Try to use a relative path.")
            print("\nYour entered path looks like: ", os.path.abspath(opt.weights))
            quit()

        if (opt.weights.endswith('.pth') == False):
            print("\nYou need a .pth file to evaluate.")
            print("\nYour entered file is: ", opt.weights)
            quit()
        
    return opt

def load_dataset(opt):
 


    #ACORDARSE PONER EN LA CARPETA QUE SE PASE LAS SUBCARPETAS DE TRAIN Y TEST
    train_data = opt.dataset + '/train/train/procesados4096-0_1/'
    print("train data-> ",train_data)
    test_data = opt.dataset + '/test/procesados4096-0_1/'
    print("test data-> ",test_data)

    print(opt.outf)
    carpeta = opt.outf

    if not os.path.exists(carpeta):
        os.makedirs(carpeta)
        print("Se ha creado la carpeta", carpeta)
    else:
        print("La carpeta", carpeta, "ya existe")
        
    print("\n----Train Dataset----")
    dataset = ParisDataset(
        root=train_data,
        datapath=train_data)  

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=int(opt.workers))

    print("\n----Test Dataset----")
    test_dataset = ParisDataset(
        root=test_data,
        datapath=test_data)
        #split='test',

    testdataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=int(opt.workers))
    
    num_classes = 8 #10 #VER COMO DETECTAR NUM CLASES DEL DATASET.. quiza fichero que lo especifique    dataset.num_seg_classes

    print ('\nTrain items: {} \nTest items: {} \nClasess: {}\n'.format(len(dataset), len(test_dataset), num_classes))

    try:
        os.makedirs(opt.outf)
    except OSError:
        pass

    return dataset, dataloader, test_dataset, testdataloader, num_classes

def evaluation(opt, testdataloader, num_classes, classifier):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if (opt.process == 'test'):
        print("Using pre-loaded weights")
        classifier.load_state_dict(torch.load(opt.weights))

        classifier.to(device) 

    else:
        print("Using the weights of the training just done")
    
    shape_ious = [] # benchmark mIOU
    todas_predicciones = np.array([])
    todas_reales = np.array([])

    #MIOU original
    
    for i,data in tqdm(enumerate(testdataloader, 0)):

        points, target = data
        points = points.transpose(2, 1)
        points, target = points.to(device), target.to(device)
        classifier = classifier.eval()
        pred, _, _ = classifier(points)
        pred_choice = pred.data.max(2)[1]
        pred_np = pred_choice.cpu().data.numpy()
        target_np = target.cpu().data.numpy() - 1
        
        todas_predicciones = np.concatenate([todas_predicciones, pred_np.ravel()]).astype(int)
        todas_reales = np.concatenate([todas_reales, target_np.ravel()]).astype(int)
        """
        for shape_idx in range(target_np.shape[0]):

            parts = range(num_classes)
            part_ious = []
            for part in parts:
                I = np.sum(np.logical_and(pred_np[shape_idx] == part, target_np[shape_idx] == part))
                U = np.sum(np.logical_or(pred_np[shape_idx] == part, target_np[shape_idx] == part))
                if U == 0:
                    iou = 1 #If the union of groundtruth and prediction points is empty, then count part IoU as 1
                else:
                    iou = I / float(U)
                part_ious.append(iou)
                
            shape_ious.append(np.mean(part_ious))
        """
    cm=confusion_matrix(todas_predicciones, todas_reales)

    print("----------")
    print(cm)
    print("----------")

    TP = np.diag(cm)
    FP = cm.sum(axis=0)-np.diag(cm)
    FN = cm.sum(axis=1)-np.diag(cm)  
    TN = cm.sum() - (FP + FN + TP)

    #print("TP->", TP)
    #print("FP->", FP)
    #print("FN->", FN)

    for iter in range(len(TP)):

        #Mean Intersection Over Union
        miou = TP[iter]/(TP[iter]+FP[iter]+FN[iter])
        #True Positive Rate
        TPR = TP[iter]/(TP[iter]+FN[iter])
        #False Positive Rate
        FPR = FP[iter]/(FP[iter]+TN[iter])
        #Overall accuracy
        ACC = (TP[iter]+TN[iter])/(TP[iter]+FP[iter]+FN[iter]+TN[iter])
        
        print ('mIOU from class: {} of Object {} is: {} '.format(iter, 'paris', miou)) #opt.class_choice
        print ('TPR from class: {} of Object {} is: {} '.format(iter, 'paris', TPR))
        print ('FPR from class: {} of Object {} is: {} '.format(iter, 'paris', FPR))
        print ('Overall ACC from class: {} of Object {} is: {} '.format(iter, 'paris', ACC))
        
        #Una forma de precision y recall
        precision = np.sum(TP[iter] / (TP[iter] + FP[iter]))
        recall = np.sum(TP[iter] / (TP[iter] + FN[iter]))
        print ('Precision from class: {} of Object {} is: {} '.format(iter, 'paris', precision))
        print ('Recall from class: {} of Object {} is: {} '.format(iter, 'paris', recall))

        print("----------")
        
    #Otra forma de hacer Precision y Recall
    tp = np.diag(cm)
    prec = list(map(truediv, tp, np.sum(cm, axis=0)))
    rec = list(map(truediv, tp, np.sum(cm, axis=1)))
    print ('Precision: {}\nRecall: {}'.format(prec, rec))
    print('\nmIOU for class {}: {}'.format('paris', np.mean(shape_ious)))

def train_test(opt):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    npoints = 10000

    opt.manualSeed = 42 # fix seed
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    dataset, dataloader, _, testdataloader, num_classes = load_dataset(opt)

    blue = lambda x: '\033[94m' + x + '\033[0m'
    
    classifier = PointNetDenseCls(k=num_classes, feature_transform=opt.feature_transform)

    if opt.model != '':
        classifier.load_state_dict(torch.load(os.path.abspath(opt.model)))

    optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    classifier.to(device) #.cuda()

    num_batch = len(dataset) / opt.batchSize

    for epoch in range(opt.nepoch):
        #scheduler.step()
        for i, data in enumerate(dataloader, 0):
            points, target = data
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            
            classifier = classifier.train()

            pred, trans, trans_feat = classifier(points) #points_normalized
            pred = pred.view(-1, num_classes)
            target = target.view(-1, 1)[:, 0] # - 1

            loss = F.nll_loss(pred, target)
            
            if opt.feature_transform:
                loss += feature_transform_regularizer(trans_feat) * 0.001
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            scheduler.step()

            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()


        

            print('[%d: %d/%d] train loss: %f accuracy: %f' % (epoch, i+1, num_batch, loss.item(), correct.item()/float(opt.batchSize * npoints)))

            """
            if i % 10 == 0:
                j, data = next(enumerate(testdataloader, 0))
                points, target = data
                points = points.transpose(2, 1)
                points, target = points.to(device) , target.to(device)  #.cuda()
                classifier = classifier.eval()
                pred, _, _ = classifier(points)
                pred = pred.view(-1, num_classes)
                target = target.view(-1, 1)[:, 0] - 1
                loss = F.nll_loss(pred, target)
                pred_choice = pred.data.max(1)[1]
                correct = pred_choice.eq(target.data).cpu().sum()
                print('[%d: %d/%d] %s loss: %f accuracy: %f' % (epoch, i, num_batch, blue('test'), loss.item(), correct.item()/float(opt.batchSize * npoints)))
            """
        torch.save(classifier.state_dict(), '%s/seg_model_%s_%d.pth' % (opt.outf, 'entrenopy', epoch))
    
    evaluation(opt, testdataloader, num_classes, classifier) #Evaluate the model with the recent finish training

def test(opt):

    opt.manualSeed = 42 #fix seed
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    _, _, _, testdataloader, num_classes = load_dataset(opt) 

    classifier = PointNetDenseCls(k=num_classes, feature_transform=opt.feature_transform)
    evaluation(opt, testdataloader, num_classes, classifier) #Evaluate the model with the weights you loaded

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
    