
import torch
from collections import defaultdict
from sklearn.metrics import roc_curve, auc,roc_auc_score
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle
import torch.nn as nn
from collections import defaultdict

#1.acc 2.准确率 3.roc 4.度量监视器
#1
def calculate_accuracy(output, target):
    if output.shape[1]>1:
        output=torch.argmax(output, 1)
        target=torch.argmax(target,1)
        result=(target == output).sum(dim=0).float()/ output.size(0) 
    else:
#        result=-nn.BCELoss()(output.float(),target.float())
        result=((output>0.5).long()==target ).sum(dim=0).float()/ output.size(0) 
#        if (output>0.5).long()==0:
#            print('gggggggg')
    ###################???????????????????argmax  max
    return result

    

#2
def calculate_accuracy_score(output, target):
    outputgrad=output[:,0:2]
    targetgrad=target[:,0:2]
    if outputgrad.shape[1]>1:
        outputgrad=torch.argmax(outputgrad, 1)
        targetgrad=torch.argmax(targetgrad,1)
        result=(targetgrad == outputgrad).sum(dim=0).float()/ outputgrad.size(0) 
    else:

        result=((outputgrad>0.5).long()==targetgrad ).sum(dim=0).float()/ output.size(0) 

    return result
    
#    output = torch.sigmoid(output) >= 0.5
#    target = target == 1.0
#    return torch.true_divide((target == output).sum(dim=0), output.size(0)).item()

#3.
def plot_auc(y_test,y_score,savepath,average='micro'):

    fpr, tpr, thresholds = roc_curve(y_test, y_score)
    rocauc = auc(fpr, tpr)
    
    
    plt.figure(figsize=(6,6))
    plt.title('Validation ROC')
    plt.plot(fpr, tpr, 'b', label = 'Val AUC = %0.3f' % rocauc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(savepath)
    return rocauc

def f1_loss(y_true:torch.Tensor, y_pred:torch.Tensor, is_training=False):
    assert y_true.ndim == 1
    assert y_pred.ndim == 1 or y_pred.ndim == 2
   
    if y_pred.ndim == 2:
        y_pred = y_pred.argmax(dim=1)
       
   
    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)
   
    epsilon = 1e-7
   
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
   
    f1 = 2* (precision*recall) / (precision + recall + epsilon)
    f1.requires_grad = is_training
    print(f"  precision: {str(precision)}  recall: {str(recall)}  F1: {str(f1)}")
    return f1

def plot_auc(y_test,y_score,savepath,average='micro'):

    fpr, tpr, thresholds = roc_curve(y_test, y_score,pos_label=1)
    rocauc = auc(fpr, tpr)
    
    
    plt.figure(figsize=(6,6))
    plt.title('Validation ROC')
    plt.plot(fpr, tpr, 'b', label = 'Val AUC = %0.3f' % rocauc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(savepath)
    print("1")
    return rocauc

def cal_precision_and_recall(true_labels, pre_labels):
    # 计算f1值
    precision = defaultdict(float, 1)
    recall = defaultdict(float, 1)
    total = defaultdict(float, 1)
    for t_lab, p_lab in zip(true_labels, pre_labels):
        total[t_lab] += 1
        recall[p_lab] += 1
        if t_lab == p_lab:
            precision[t_lab] += 1
    
    for sub in range(precision.dict):
        pre = precision[sub] / recall[sub]
        rec =  precision[sub] / total[sub]
        F1 = (2 * pre * rec) / (pre + rec)
        print(f"{str(sub)}  precision: {str(pre)}  recall: {str(rec)}  F1: {str(F1)}")
#3
def P_R_F(true_labels, pre_labels, criterion,net):
    n_classes=2
    net.eval()
    test_loss = 0 
    target_num = torch.zeros((1, n_classes)) # n_classes为分类任务类别数量
    predict_num = torch.zeros((1, n_classes))
    acc_num = torch.zeros((1, n_classes))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for step, (inputs, targets) in zip(true_labels, pre_labels):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, _ = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            
            pre_mask = torch.zeros(outputs.size()).scatter_(1, predicted.cpu().view(-1, 1), 1.)
            predict_num += pre_mask.sum(0)  # 得到数据中每类的预测量
            tar_mask = torch.zeros(outputs.size()).scatter_(1, targets.data.cpu().view(-1, 1), 1.)
            target_num += tar_mask.sum(0)  # 得到数据中每类的数量
            acc_mask = pre_mask * tar_mask 
            acc_num += acc_mask.sum(0) # 得到各类别分类正确的样本数量

        recall = acc_num / target_num
        precision = acc_num / predict_num
        F1 = 2 * recall * precision / (recall + precision)
        accuracy = 100. * acc_num.sum(1) / target_num.sum(1)

        print('Test Acc {}, recal {}, precision {}, F1-score {}'.format(accuracy, recall, precision, F1))

    return accuracy

def multiclass_plot_auc(y_test,y_score,savepath,average='micro'):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes=y_score.shape[1]
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute micro-average ROC curve and ROC area（方法二）
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # Compute macro-average ROC curve and ROC area（方法一）
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    # Plot all ROC curves
    lw=2
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)
    
    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)
    
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))
    
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.savefig(savepath)
    return roc_auc[average]

#4
class MetricMonitor:
    def __init__(self, float_precision=3):
        self.float_precision = float_precision
        self.reset()

    def reset(self):
        self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})

    def update(self, metric_name, val):
        metric = self.metrics[metric_name]

        metric["val"] += val
        metric["count"] += 1
        metric["avg"] = metric["val"] / metric["count"]

    def __str__(self):
        return " | ".join(
            [
                "{metric_name}: {avg:.{float_precision}f}".format(
                    metric_name=metric_name, avg=metric["avg"], float_precision=self.float_precision
                )
                for (metric_name, metric) in self.metrics.items()
            ]
        )