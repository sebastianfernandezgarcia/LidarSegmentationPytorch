import numpy as np
from pathlib import Path
import time

import torch
import torch.nn as nn

#from data import data_loaders
from data_test import data_loaders_original

from model import RandLANet
from utils.ply import read_ply, write_ply

from vis.view_copy import view_points_labels
import tqdm
from sklearn.metrics import confusion_matrix
import logging

from confusion_matrix_plot import muestra_matriz_confusion

t0 = time.time()

#path = Path('datasets') / 's3dis' / 'subsampled' / 'test'
path = r'dataset_final' #r'dataset_final_pruebas_balanceo_2/'
path = r'C:/Users/sfernandez/nueva_etapa/github/Datasets/Aerolaser/train/train/procesados50000-0_1'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print('Loading data...')
#loader, val_loader, test_loader = data_loaders(path, 'naive')


test_loader_original = data_loaders_original(path, 'naive')
#print(loader)

# Set up logging configuration
logging.basicConfig(filename='MetricasRandLaNet.log', filemode='w', level=logging.DEBUG)

print('Loading model...')

d_in = 3
num_classes = 8 #14

model = RandLANet(d_in, num_classes, 16, 4, device)
model.load_state_dict(torch.load('runs\checkpoint_374_mejor_torre_todo.pth')['model_state_dict'])  #'runs/2020-04-11_17:03/checkpoint_10.pth'
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



def metrics(cm):
    """
    def accuracy_por_clase(cm):
        tp = np.diag(cm)
        fp = np.sum(cm, axis=0) - tp
        tn = np.sum(cm) - np.sum(cm, axis=0) - np.sum(cm, axis=1) + np.diag(cm)
        fn = np.sum(cm, axis=1) - np.diag(cm)

        acc = (tp+tn)/(tp+tn+fp+fn)

        return acc
        #Exactitud = (TP + TN) / (TP + TN + FP + FN)
    """
    def accuracy(cm):
        return np.trace(cm) / np.sum(cm)

    def accuracy_formula(cm):
        tp = np.diag(cm)
        fp = np.sum(cm, axis=0) - tp
        fn = np.sum(cm, axis=1) - tp
        tn = np.sum(cm) - np.sum(cm, axis=0) - np.sum(cm, axis=1) + np.diag(cm)
        return (tp + tn) / (tp + tn + fp + fn)
    
    
    def precision(cm):
        """
        num_classes = cm.shape[0]
        precisions = []

        for i in range(num_classes):
            TP = cm[i, i]
            FP = np.sum(cm[:, i]) - TP
            precision = TP / (TP + FP)
            precisions.append(precision)

        print("Precision for each class:")
        for i, precision in enumerate(precisions):
            print(f"Class {i}: {precision}")

        time.sleep(30)
        """
        
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
    
    #acc_per_class = accuracy_por_clase(cm)
    acc = accuracy(cm)
    precision_matrix = precision(cm)
    recall_matrix = recall(cm)
    f1 = f1_score(cm)
    miou = mean_iou(cm)
    acc_con_formula = accuracy_formula(cm)
    #print("Accuracy:", acc)
    #print("Precision:", prec)
    #print("Recall:", rec)
    #print("F1 score:", f1)
    #print("mIoU:", miou)
    return acc_con_formula, acc, precision_matrix, recall_matrix, f1, miou


#class_dict = {0: 'Suelo', 1: 'Edificio', 2: 'Señales', 3: 'Bolardo', 4: 'Papelera', 5: 'Barrier', 6: 'Peaton', 7: 'Coche', 8: 'Vegetación'}
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
#A EVALUAR
with torch.no_grad():

    for points, labels, original_points in test_loader_original:

        contador+=1
        points = points.to(device)

        #TRUE LABELS
        all_gt_labels = np.concatenate((all_gt_labels, labels.view(-1).numpy()), axis=0)
        
        labels = labels.to(device)

        #np_labels = labels.view(-1).numpy()
        #numpy_array = tensor.numpy()
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



    from collections import Counter
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report
    #y_true = [0, 1, 2, 1, 0, 1, 2, 0, 1, 2, 0, 0, 1, 2, 2, 1, 0, 2, 1, 0]
    #y_pred = [0, 1, 0, 1, 0, 2, 2, 0, 1, 2, 0, 1, 1, 2, 2, 0, 0, 2, 1, 0]


    y_true = all_gt_labels #total_target.flatten()
    y_pred = all_pred_labels #total_pred.flatten()
    



    print("----------------------")
    precision = precision_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)
    f1 = f1_score(y_true, y_pred, average=None)
    accuracy = accuracy_score(y_true, y_pred)

    print(f'Precision per class: {precision}')
    print(f'Recall per class: {recall}')
    print(f'F1 score per class: {f1}')
    print(f'Overall accuracy: {accuracy}')



    print("Report de Sklearn")
    logging.info("\n ------------------------------ \n")
    logging.info("Report de Sklearn:")
    logging.info(classification_report(y_true, y_pred))
    print(classification_report(y_true, y_pred))
    logging.info("\n ------------------------------ \n")
    # Inicializar un diccionario para almacenar los aciertos por clase
    class_accuracies = {}

    
    true_counts = Counter(y_true)

    # Calcular el accuracy por clase
    for class_label in true_counts.keys():
        true_positive = sum(1 for y_true, y_pred in zip(y_true, y_pred) if y_true == class_label and y_pred == class_label)
        total_examples = true_counts[class_label]
        class_accuracies[class_label] = true_positive / total_examples

    #print(class_accuracies)
    sorted_accuracy_values = [value for key, value in sorted(class_accuracies.items())]
    #print(sorted_accuracy_values)
    #time.sleep(20)


    cf_matrix = confusion_matrix(y_true, y_pred)
    
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

    
    #print("\n ------------------------------ \n")
    #print('mIOU for {}: {}'.format("Aerolaser Dataset", np.mean(shape_ious)))
    #logging.info("\n ------------------------------ \n")
    #logging.info('mIOU for {}: {}'.format("Aerolaser Dataset", np.mean(shape_ious)))
    
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

    #el accuracy por clase te lo traes de sorted_accuracy_values
    acc_con_formula, acc, prec, rec, f1, miou = metrics(cf_matrix)

    print("Overall Acc: ", acc)
    print("Overall mIOU: ", miou)
    print("Overall Accuracy (C/T) por clase: ", sorted_accuracy_values)
    print("Overall Accuracy (F/MC) por clase: ", acc_con_formula)
    print("Overall Precision (All classes): ", prec)
    print("Overall Recall (All classes): ", rec)
    print("Overall F1 Score (All classes): ", f1)
    
    logging.info("\n ------------------------------ \n")
    logging.info('Overall Acc: {}'.format(acc))
    logging.info('Overall mIOU: {}'.format(miou))

    logging.info('Overall Accuracy (C/T) por clase {}'.format(sorted_accuracy_values))
    logging.info('Overall Accuracy (F/MC) por clase {}'.format(acc_con_formula))
    logging.info('Overall Precision (All classes): {}'.format(prec))  
    logging.info('Overall Recall (All classes): {}'.format(rec))  
    logging.info('Overall F1 Score (All classes): {}'.format(f1))  


    for current_class in range(len(prec)):

        print("---------------------------")

        print('Accuracy (C/T) for class {}: {}'.format(class_dict[current_class], sorted_accuracy_values[current_class]))
        print('Accuracy (F/MC)for class {}: {}'.format(class_dict[current_class], acc_con_formula[current_class]))
        print('Precision for class {}: {}'.format(class_dict[current_class], prec[current_class]))
        print('Recall  for class {}: {}'.format(class_dict[current_class], rec[current_class]))
        print('F1 Score for class {}: {}'.format(class_dict[current_class], f1[current_class]))
        
        logging.info('---------------------------')
        logging.info('Accuracy (C/T) for class {}: {}'.format(class_dict[current_class], sorted_accuracy_values[current_class]))
        logging.info('Accuracy (F/MC) for class {}: {}'.format(class_dict[current_class], acc_con_formula[current_class]))
        logging.info('Precision for class {}: {}'.format(class_dict[current_class], prec[current_class]))
        logging.info('Recall  for class {}: {}'.format(class_dict[current_class], rec[current_class]))
        logging.info('F1 Score for class {}: {}'.format(class_dict[current_class], f1[current_class]))

    print('Done.')


#llamas a metrics.


view_points_labels(all_original_points, all_pred_labels, all_pred_labels)


"""
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
"""

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