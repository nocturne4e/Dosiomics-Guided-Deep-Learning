import torch.nn as nn
import copy
import random
import os
import numpy as np
from tqdm import tqdm
import torch
import torch.optim
from sklearn.metrics import roc_curve, auc,roc_auc_score
from Pre_processing import  get_dataloaders,ESODataset,backcrop
from Loss import FocalLoss,NTXentLossBinary,dice_loss
import pandas as pd
from helper import calculate_accuracy,MetricMonitor
from generate_model import getmodel


torch.backends.cudnn.enabled = True
def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
set_seed(9)
   

def train(train_loader, model, criterion, optimizer, epoch, params):
    probas_=[]
    states_=[]
    metric_monitor = MetricMonitor()
    model.train()
    stream = tqdm(train_loader)
    for i, (img, msk,patientid,patientstate) in enumerate(stream, start=1):
        target=patientstate.reshape(patientstate.shape[0],1)
        target=target.clone().detach().to(torch.int64)
        target = target.to(params["device"], non_blocking=True)              
        img = img.to(params["device"], non_blocking=True)
        msks =msk.to(params["device"], non_blocking=True)       
        output1,output2,zs = model(img)

        if params["numclass"]==1:

            loss_fn = FocalLoss().to(device)
            loss = loss_fn(output1.float(), target.float()) 
            accuracy = calculate_accuracy(output1.float(), target.float()) 
           
            loss_fxn=NTXentLossBinary(device,0.05).to(device)
            loss1=loss_fxn(zs[0],zs[1])
             
            loss2 = dice_loss(output2.float(), msks.float())          
    
            total_loss=loss+(loss1/100)+(loss2/20)

        else:
            accuracy = calculate_accuracy(output1, target)
            loss = criterion(output1,target.float())

        metric_monitor.update("loss", loss.item())
        metric_monitor.update("loss1", loss1.item())
        metric_monitor.update("loss2", loss2.item())
        metric_monitor.update("total_loss", total_loss.item())
        metric_monitor.update("Accuracy", accuracy.item())
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        stream.set_description(
            "Epoch: {epoch}. Train.   {metric_monitor}".format(epoch=epoch, metric_monitor=metric_monitor)
        )
        state=target.cpu().numpy().tolist()
        output1=torch.nn.Sigmoid()(output1)
        proba=output1.cpu().detach().numpy().tolist()
        probas_ = probas_+proba
        states_= states_+state

    fpr, tpr, thresholds = roc_curve(np.array(states_),np.array(probas_),pos_label=1)
    rocauc = auc(fpr, tpr)
    metric_monitor.update("train_AUC", rocauc )
    return metric_monitor


def validate(val_loader, model, criterion, epoch, params):
    probas_=[]
    states_=[]
    metric_monitor = MetricMonitor()
    model.eval()
    stream = tqdm(val_loader)
    with torch.no_grad():   
        for i, (img, msk,patientid,patientstate) in enumerate(stream, start=1):
            target=patientstate.reshape(patientstate.shape[0],1)
            target=target.clone().detach().to(torch.int64)
            target = target.to(params["device"], non_blocking=True)              
            img = img.to(params["device"], non_blocking=True)
            msks =msk.to(params["device"], non_blocking=True)       
            output1,output2,zs = model(img)
            
            if params["numclass"]==1:

                loss_fn = FocalLoss().to(device)
                loss = loss_fn(output1.float(), target.float()) 
                accuracy = calculate_accuracy(output1.float(), target.float()) 
            
                loss_fxn=NTXentLossBinary(device,0.05).to(device)
                loss1=loss_fxn(zs[0],zs[1])
                
                loss2 = dice_loss(output2.float(), msks.float())          
        
                total_loss=loss+(loss1/100)+(loss2/20)

            else:
                accuracy = calculate_accuracy(output1, target)
                loss = criterion(output1,target.float())

        metric_monitor.update("loss", loss.item())
        metric_monitor.update("loss1", loss1.item())
        metric_monitor.update("loss2", loss2.item())
        metric_monitor.update("total_loss", total_loss.item())
        metric_monitor.update("Accuracy", accuracy.item())
            
        stream.set_description(
            "Epoch: {epoch}. Validation. {metric_monitor}".format(epoch=epoch, metric_monitor=metric_monitor)
        )
        
        state=target.cpu().numpy().tolist()
        output1=torch.nn.Sigmoid()(output1)
        proba=output1.cpu().numpy().tolist()

        probas_ = probas_+proba
        states_= states_+state

    fpr, tpr, thresholds = roc_curve(np.array(states_),np.array(probas_),pos_label=1)
    rocauc = auc(fpr, tpr)
    metric_monitor.update("val_AUC", rocauc )  
    return metric_monitor

def test(test_loader, model, criterion, epoch, params):

    metric_monitor = MetricMonitor()
    model.eval()
    stream = tqdm(test_loader)
    probas_=[]
    states_=[]
    with torch.no_grad():   
        for i, (img, msk,patientid,patientstate) in enumerate(stream, start=1):
            target=patientstate.reshape(patientstate.shape[0],1)
            target=target.clone().detach().to(torch.int64)
            target = target.to(params["device"], non_blocking=True)              
            img = img.to(params["device"], non_blocking=True)
            msks =msk.to(params["device"], non_blocking=True)       
            output1,output2,zs = model(img)
            
            if params["numclass"]==1:

                loss_fn = FocalLoss().to(device)
                loss = loss_fn(output1.float(), target.float()) 
                accuracy = calculate_accuracy(output1.float(), target.float()) 
            
                loss_fxn=NTXentLossBinary(device,0.05).to(device)
                loss1=loss_fxn(zs[0],zs[1])
                
                loss2 = dice_loss(output2.float(), msks.float())          
        
                total_loss=loss+(loss1/100)+(loss2/20)

            else:
                accuracy = calculate_accuracy(output1, target)
                loss = criterion(output1,target.float())

        metric_monitor.update("loss", loss.item())
        metric_monitor.update("loss1", loss1.item())
        metric_monitor.update("loss2", loss2.item())
        metric_monitor.update("total_loss", total_loss.item())
        metric_monitor.update("Accuracy", accuracy.item())
        stream.set_description(
                "Epoch: {epoch}. test.      {metric_monitor}".format(epoch=epoch, metric_monitor=metric_monitor)
            )

        state=target.cpu().numpy().tolist()
        output1=torch.nn.Sigmoid()(output1)
        proba=output1.cpu().numpy().tolist()
        probas_ = probas_+proba
        states_= states_+state
    fpr, tpr, thresholds = roc_curve(states_, probas_)
    rocauc = auc(fpr, tpr)
    metric_monitor.update("test_AUC", rocauc )          
    return metric_monitor

 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
params = {
    "model": "resnet",
    "device": device,
    "lr": 0.01,
    "batch_size": 8,
    "num_workers": 0,
    "epochs":100,
    #"class_weight":torch.from_numpy(np.array([1,1])).float().to(device),
    "numclass":1
}


def traintestmodel(train_loader,val_loader,test_loader,params):

    model=getmodel("DLD_model",34,1,1,96)
    model = model.to(params["device"]) 
    
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma = 0.5, last_epoch = -1)
    criterion=nn.BCEWithLogitsLoss()

    best_accuracy=0
    data={}
    data['losstrain']=[]
    data['lossval']=[]
    data['acctrain']=[]
    data['accval']=[]
    data['acctest']=[]
    data['train_AUC']=[]
    data['val_AUC']=[]
    data['test_AUC']=[]

    best_model_wts=0
    for epoch in range(1, params["epochs"] + 1):
        train_metric=train(train_loader, model, criterion, optimizer, epoch, params)
        scheduler.step()
        data['losstrain'].append(train_metric.metrics['Loss']['avg'])
        data['acctrain'].append(train_metric.metrics['Accuracy']['avg'])         
        data['train_AUC'].append(train_metric.metrics['train_AUC']['avg'])

        val_metric=validate(val_loader, model, criterion, epoch, params)
        data['lossval'].append(val_metric.metrics['Loss']['avg'])
        data['accval'].append(val_metric.metrics['Accuracy']['avg'])
        data['val_AUC'].append(val_metric.metrics['val_AUC']['avg'])
        
        print("train:","{:.5f}".format(float(train_metric.metrics['train_AUC']['avg'])),\
              "val:","{:.5f}".format(float(val_metric.metrics['val_AUC']['avg'])))
                 
        if(val_metric.metrics['val_AUC']['avg']>best_accuracy):
            data['best_accuracy']=val_metric.metrics['val_AUC']['avg']
            best_accuracy=data['best_accuracy']
            best_model_wts = copy.deepcopy(model.state_dict())
            data['best_epoch']=epoch
            
    if best_model_wts!=0:
        str1=r'/media/physics/EXTERNAL_US/code/best_model/best_model_weight'
        torch.save(best_model_wts, str1)
        print("found")
    else:
        print("not found")    
  
    df = pd.DataFrame(data)
    df.to_excel(r'/media/physics/EXTERNAL_US/result/result.xlsx')


dataloaders, size = get_dataloaders(params['batch_size'])
train_loader=dataloaders['train']
val_loader=dataloaders['val']
test_loader=dataloaders['test']

traintestmodel( train_loader,val_loader,test_loader,params)
