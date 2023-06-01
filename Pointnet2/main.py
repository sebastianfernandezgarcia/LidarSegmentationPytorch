'''
Modified from https://github.com/fxia22/pointnet.pytorch/blob/master/utils/train_segmentation.py
              https://github.com/dragonbook/pointnet2-pytorch
'''
import os
import random
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dataset.dataset import Aerolaser
from model.pointnet2_part_seg import PointNet2PartSegmentNet
import torch_geometric.transforms as GT
from sklearn.metrics import confusion_matrix
from confusion_matrix_plot import muestra_matriz_confusion
import logging
from pytorchtools import EarlyStopping
from tqdm import tqdm                                                                                                

## Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--npoints', type=int, default=16384, help='resample points number') #Choose here number of point resample (if needed)
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--nepoch', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='checkpoint', help='output folder')
parser.add_argument('--test_dataset', type=str, default=r'C:/Users/sfernandez/nueva_etapa/github2/LidarSegmentationPytorch/Datasets/Aerolaser/train/validation/', help='test datasetfolder') #r'./aerolaser_validation/'
parser.add_argument('--eval_test_dataset', type=str, default=r'C:/Users/sfernandez/nueva_etapa/github2/LidarSegmentationPytorch/Datasets/Aerolaser/test/', help='test datasetfolder')
parser.add_argument('--train_dataset', type=str, default=r'C:/Users/sfernandez/nueva_etapa/github2/LidarSegmentationPytorch/Datasets/Aerolaser/train/train/', help='train datasetfolder')
parser.add_argument('--batch_size', type=int, default=8, help='input batch size')   #Change here batchSize if needed
parser.add_argument('--patience', type=int, default=10, help='the patience the training earlystoping will have')   #Chane patience if needed
parser.add_argument('--num_workers', type=int, default=0, help='number of data loading workers')
parser.add_argument('--behaviour', type=str, choices = ('trainval', 'test'), default='trainval', help='what you want to do')
#if you set behaviour to test, make sure to give model, dataset folder... check batchSize..

opt = parser.parse_args()
print(opt)

def setSeeds():
    ## Random seed
    # opt.manual_seed = np.random.randint(1, 10000)  # fix seed
    # TODO: Still cannot get determinstic result
    opt.manual_seed = 123
    print('Random seed: ', opt.manual_seed)
    random.seed(opt.manual_seed)
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)
    torch.cuda.manual_seed(opt.manual_seed)

def DatasetandTrainingConfiguration(train_dataset_dir, test_dataset_dir, eval_test_dataset_dir):
    ## Dataset and transform
    print('Construct dataset ..')
    rot_max_angle = 15
    trans_max_distance = 0.01

    RotTransform = GT.Compose([GT.RandomRotate(rot_max_angle, 0), GT.RandomRotate(rot_max_angle, 1), GT.RandomRotate(rot_max_angle, 2)])
    TransTransform = GT.RandomJitter(trans_max_distance) #deprecated, usar RandomJitter

    train_transform = GT.Compose([GT.NormalizeScale(), RotTransform, TransTransform])
    test_transform = GT.Compose([GT.NormalizeScale(), ])

    dataset = Aerolaser(
        train_dir=train_dataset_dir, test_dir=test_dataset_dir, train=True, transform=train_transform, npoints=opt.npoints)   #train_transform
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    print(len(dataset))
    print(len(dataloader))
    import time 
    time.sleep(10)
    test_dataset = Aerolaser(
        train_dir=train_dataset_dir, test_dir=test_dataset_dir, train=False, transform=test_transform, npoints=opt.npoints) #test_transform
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    
    eval_test_dataset = Aerolaser(
        train_dir=eval_test_dataset_dir, test_dir=eval_test_dataset_dir, train=False, transform=test_transform, npoints=opt.npoints) #test_transform
    eval_test_dataloader = torch.utils.data.DataLoader(
        eval_test_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)


    #def __init__(self, train_dir=r'./aerolaser', test_dir=r'./aerolaser_test/', train=True, transform=None, npoints=2500):
    num_classes = dataset.num_classes()
    
    print('dataset size: ', len(dataset))
    print('test_dataset size(valid): ', len(test_dataset))
    print('eval_test_dataset size: ', len(eval_test_dataset))
    print('num_classes: ', num_classes)

    try:
        os.mkdir(opt.outf)
    except OSError:
        pass

    ## Model, criterion and optimizer
    print('Construct model ..')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    dtype = torch.float
    print('cudnn.enabled: ', torch.backends.cudnn.enabled)


    net = PointNet2PartSegmentNet(num_classes)

    if opt.model != '':
        net.load_state_dict(torch.load(opt.model))
    net = net.to(device, dtype)

    criterion = nn.NLLLoss()
    #print(str(net.parameters()))
    #import time
    #time.sleep(30)
    optimizer = optim.Adam(net.parameters())


    ## Train
    print('Training ..')
    blue = lambda x: '\033[94m' + x + '\033[0m'
    num_batch = len(dataset) // opt.batch_size
    #test_per_batches = opt.test_per_batches

    print('number of epoches: ', opt.nepoch)
    print('number of batches per epoch: ', num_batch)
    #print('run test per batches: ', test_per_batches)

    # Set up logging configuration
    logging.basicConfig(filename='EpochsMetrics.log', filemode='w', level=logging.DEBUG)

    return dataloader, test_dataloader, eval_test_dataloader, criterion, optimizer, blue, device, dtype, num_batch, num_classes, net

def Train(net, dataloader, device, dtype, optimizer, num_classes, num_batch, test_dataloader, patience):

    early_stopping = EarlyStopping(patience, verbose=True)

    for epoch in range(opt.nepoch):
        print('Epoch {}, total epoches {}'.format(epoch+1, opt.nepoch))

        net.train()

        for batch_idx, sample in enumerate(dataloader):

            points, labels = sample['points'], sample['labels']

            points = points.transpose(1, 2).contiguous()  # (batch_size, 3, n)
            points, labels = points.to(device, dtype), labels.to(device, torch.long)

            optimizer.zero_grad()

            pred = net(points)  # (batch_size, n, num_classes)
            
            print(pred.shape)
            print(pred)
            pred = pred.view(-1, num_classes)  # (batch_size * n, num_classes) 
            target = labels.view(-1, 1)[:, 0]

            loss = F.nll_loss(pred, target)
            loss.backward()

            optimizer.step()

            pred_label = pred.detach().max(1)[1] 
            correct = pred_label.eq(target.detach()).cpu().sum()
            total = pred_label.shape[0]

            print('[{}: {}/{}] train loss: {} accuracy: {}'.format(epoch, batch_idx, num_batch, loss.item(), float(correct.item())/total))

        val_loss = training_eval(epoch)
        early_stopping(val_loss, net) # check if the validation loss has stopped improving
        
        if early_stopping.early_stop:
            print("Early stopping")
            break

def training_eval(epoch):
    #Eval for each epoch
    net.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        val_loss = 0
        for i, data in enumerate(tqdm(test_dataloader, 0)): #added tqdm here to try to represent the proggress of the training eval
            points, labels = data['points'], data['labels']
            points = points.transpose(1, 2).contiguous()
            points, labels = points.to(device), labels.to(device, torch.long)
            pred = net(points)
            pred = pred.view(-1, num_classes)
            target = labels.view(-1, 1)[:, 0]
            val_loss += F.nll_loss(pred, target, reduction='sum').item()
            pred_label = pred.detach().max(1)[1]
            correct += pred_label.eq(target.detach()).cpu().sum().item()
            total += target.shape[0]
        val_loss /= total
        accuracy = correct / total

        print('[Epoch {}] Validation Loss: {:.4f} Validation Accuracy: {:.4f}'.format(epoch, val_loss, accuracy))
        logging.info('[Epoch {}] Validation Loss: {:.4f} Validation Accuracy: {:.4f}'.format(epoch, val_loss, accuracy))  

        return val_loss 
        #torch.save(net.state_dict(), '{}/seg_model_{}_{}.pth'.format(opt.outf, "aerolaser_pesos", epoch)) #model saves inside early_stipping if model is better that last epoch

def benchmark_final(net, test_dataloader, num_classes):
    #BenchMark (Confusion Matrix, mIOU)
    print("Starting Final Benchmark")
    net.eval()
    shape_ious = []
    total_pred = np.empty(0)
    total_target = np.empty(0)

    class_dict = {
        0: "terreno",
        1: "vegetación",
        2: "coche",
        3: "torre",
        4: "cable",
        5: "valla/muro",
        6: "farola",
        7: "edificio"
        }

    primero = True

    with torch.no_grad():
        for batch_idx, sample in enumerate(tqdm(test_dataloader)):

            points, labels = sample['points'], sample['labels']
            points = points.transpose(1, 2).contiguous()
            points = points.to(device, dtype)

            # start_t = time.time()
            pred = net(points) # (batch_size, n, num_classes)
            # print('batch inference forward time used: {} ms'.format(time.time() - start_t))

            pred_label = pred.max(2)[1]
            pred_label = pred_label.cpu().numpy()

            target_label = labels.numpy()
            target_label_reshaped = np.squeeze(target_label)

            if primero:
                total_pred = pred_label
                total_target = target_label_reshaped
                primero = False

            if primero==False:
                total_pred = np.concatenate((total_pred, pred_label))
                """
                print("-------------")
                print("-------------")
                print("-------------")
                print("-------------")
                print("-------------")

                print("Total_target", total_target)
                print("Total_target SHAPE", total_target.shape)

                print("target_label_reshaped", target_label_reshaped)
                print("target_label_reshaped SHAPE", target_label_reshaped.shape)

                print("-------------")
                print("-------------")
                print("-------------")
                print("-------------")
                print("-------------")
                """
                total_target = np.concatenate((total_target, target_label_reshaped))

            
            batch_size = target_label.shape[0]
            
            for shape_idx in range(batch_size):
                parts = range(num_classes)  # np.unique(target_label[shape_idx])
                part_ious = []
                for part in parts:
                    I = np.sum(np.logical_and(pred_label[shape_idx] == part, target_label[shape_idx] == part))
                    U = np.sum(np.logical_or(pred_label[shape_idx] == part, target_label[shape_idx] == part))
                    if U == 0: iou = 1
                    else: iou = float(I) / U
                    part_ious.append(iou)
                shape_ious.append(np.mean(part_ious))
            
        cf_matrix = confusion_matrix(total_pred.flatten(), total_target.flatten())
        
        cf_matrix_de_CHANO = np.array([[9857295, 172798, 30325, 2226, 451, 49382, 295, 36221],
              [472961, 3303423, 15047, 30569, 3168, 54343, 6088, 38344],
              [93268, 48979, 49882, 1949, 9, 38522, 119, 29285],
              [92, 1135, 1, 31563, 578, 13, 236, 97],
              [15, 159, 11, 3414, 77170, 3, 4, 269],
              [56580, 34622, 17693, 13884, 924, 47485, 1820, 27052],
              [893, 3166, 316, 711, 402, 930, 2784, 592],
              [939091, 287068, 17141, 92348, 11787, 101579, 1512, 2182312]])
        ##########################
        #cf_matrix = cf_matrix_de_CHANO                                              CAMBIAR PARA EVALUAR LA DE CHANO
        ############################
        print("\n ------------------------------ \n")
        print("Confusion Matrix:")
        print(cf_matrix)
        logging.info("\n ------------------------------ \n")
        logging.info("Confusion Matrix:")
        logging.info(cf_matrix)

        muestra_matriz_confusion(cf_matrix)
    
        
        print("\n ------------------------------ \n")
        print('mIOU for {}: {}'.format("Aerolaser Dataset", np.mean(shape_ious)))
        logging.info("\n ------------------------------ \n")
        logging.info('mIOU for {}: {}'.format("Aerolaser Dataset", np.mean(shape_ious)))
        
        print("\n ------------------------------ \n")

        print("---------------")
        print("Confussion")
        
        print("---------------")
        print("Confussion")
        
        print("---------------")
        print("Confussion")
        print(cf_matrix)
        
        print("---------------")
        print("Confussion")
        
        print("---------------")
        print("Confussion")

        acc, prec, rec, f1, miou = metrics(cf_matrix)

        print("Overall Acc: ", acc)
        print("Overall mIOU: ", miou)
        print("Overall Precision (All classes): ", prec)
        print("Overall Recall (All classes): ", rec)
        print("Overall F1 Score (All classes): ", f1)
        
        logging.info("\n ------------------------------ \n")
        logging.info('Overall Acc: {}'.format(acc))
        logging.info('Overall mIOU: {}'.format(miou))


        logging.info('Overall Precision (All classes): {}'.format(prec))  
        logging.info('Overall Recall (All classes): {}'.format(rec))  
        logging.info('Overall F1 Score (All classes): {}'.format(f1))  

        for current_class in range(len(prec)):

            print("---------------------------")


            print('Precision for class {}: {}'.format(class_dict[current_class], prec[current_class]))
            print('Recall  for class {}: {}'.format(class_dict[current_class], rec[current_class]))
            print('F1 Score for class {}: {}'.format(class_dict[current_class], f1[current_class]))
            
            logging.info('---------------------------')
            logging.info('Precision for class {}: {}'.format(class_dict[current_class], prec[current_class]))
            logging.info('Recall  for class {}: {}'.format(class_dict[current_class], rec[current_class]))
            logging.info('F1 Score for class {}: {}'.format(class_dict[current_class], f1[current_class]))

    print('Done.')

def metrics(cm):
    def accuracy(cm):
        return np.trace(cm) / np.sum(cm)

    
    def precision(cm):
        tp = np.diag(cm)
        fp = np.sum(cm, axis=0) - tp
        return tp / (tp + fp)

    def recall(cm):
        tp = np.diag(cm)
        fn = np.sum(cm, axis=1) - tp
        return tp / (tp + fn)

    def f1_score(cm):
        p = precision(cm)
        r = recall(cm)
        return 2 * (p * r) / (p + r)

    def iou(cm):
        ious = []
        for i in range(cm.shape[0]):
            tp = cm[i, i]
            fp = np.sum(cm[:, i]) - tp
            fn = np.sum(cm[i, :]) - tp
            iou = tp / (tp + fp + fn)
            ious.append(iou)
        return np.array(ious)

    def mean_iou(cm):
        ious = iou(cm)
        return np.mean(ious)

    acc = accuracy(cm)
    prec = precision(cm)
    rec = recall(cm)
    f1 = f1_score(cm)
    miou = mean_iou(cm)

    
    #print("Accuracy:", acc)
    #print("Precision:", prec)
    #print("Recall:", rec)
    #print("F1 score:", f1)
    #print("mIoU:", miou)
    return acc, prec, rec, f1, miou

if __name__ == "__main__":

    setSeeds()
    dataloader, test_dataloader, eval_test_dataloader, criterion, optimizer, blue, device, dtype, num_batch, num_classes, net = DatasetandTrainingConfiguration(opt.train_dataset, opt.test_dataset, opt.eval_test_dataset)

    if opt.behaviour == 'trainval':
        Train(net, dataloader, device, dtype, optimizer, num_classes, num_batch, test_dataloader, opt.patience)
        benchmark_final(net, test_dataloader, num_classes) #al validation final se estan pasando los datos e validacion.. puede que despues haya que hacer test claramente...

    if opt.behaviour == 'test':

        test_net = PointNet2PartSegmentNet(num_classes)
        test_net.load_state_dict(torch.load(opt.model))
        test_net = net.to(device, dtype)

        benchmark_final(test_net, eval_test_dataloader, num_classes) #test_dataloader
