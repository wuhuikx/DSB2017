import numpy as np
import os
import time
import random
import warnings
import json

import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.nn.functional import cross_entropy,sigmoid,binary_cross_entropy
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import shutil


def get_lr(epoch,args):
    assert epoch<=args.lr_stage2[-1]
    if args.lr==None:
        lrstage = np.sum(epoch>args.lr_stage2)
        lr = args.lr_preset2[lrstage]
    else:
        lr = args.lr
    return lr


class Sample(object):
    def __init__(self):
        self.id = []
        self.x = torch.from_numpy(np.zeros(1)).float()
        self.coord = torch.from_numpy(np.zeros(1))
        self.isnod = torch.from_numpy(np.zeros(1))
        self.score = []
        self.id_patient = []
        self.id_nodule = []
        
        self.pred = []
        self.loss = []
        

class Indicator(object):
    def __init__(self):
        self.tpn = 0
        self.fpn = 0
        self.fnn = 0
        self.tnn = 0
        self.gt_nod = 0
        self.pred_nod = 0
        self.pred_all = 0
        self.pred_correct = 0
        
        self.loss = 0
        self.missloss = 0
        
        self.mean_acc = 0
        self.mean_loss = 0
        self.recall = 0
        self.precision = 0


def check_sample(sample):            
    for i in range(len(sample.id_patient)):        
        print('---------sample.id_patient = ', sample.id_patient[i])
        print('score = ', sample.score[i])
        print('isnod = ', sample.isnod)
        print('x = ', sample.x.size)
        print('coord = ', sample.coord.size)
        
        crop = sample.x.numpy()[i]
        crop_pic = crop[0, 0, int(crop.shape[-1]/2), :, :]
        plt.imshow(crop_pic, cmap = 'gray')
        plt.show()

    
def train_casenet_shuffle(epoch,model,data_loader,optimizer,args, save_data, save_name):
    model.train()
    if args.freeze_batchnorm:    
        for m in model.modules():
            if isinstance(m, nn.BatchNorm3d):  # decide if m is the type of nn.BatchNorm3d
                m.eval()  #fix the parameter of m

    starttime = time.time()
    #lr = get_lr(epoch,args)
    #print('lr=',lr)
    #for param_group in optimizer.param_groups:
    #    param_group['lr'] = lr
    
    print('-----------sub-step 1---------') 
    pos_samples, neg_samples = get_model_input(data_loader)
    print('1 pos_samples = {0}\tneg_samples = {1}'.format(len(pos_samples), len(neg_samples)))
  
           
    print('-----------sub-step 2---------')
    is_hard_mining = True
    if is_hard_mining:  
        neg_samples = hard_mining(pos_samples, neg_samples, 1)
    print('2 pos_samples = {0}\tneg_samples = {1}'.format(len(pos_samples), len(neg_samples)))
    samples = neg_samples + pos_samples
     
    '''print('-----------sub-step 3---------')
    is_cat_sample = True
    if is_cat_sample:
        random.shuffle(neg_samples)
        random.shuffle(pos_samples) 
        
        neg_samples1 = []
        for i in range(0, len(neg_samples)-1, 2):
            sample = cat_samples(neg_samples[i], neg_samples[i+1])
            neg_samples1.append(sample)
        neg_samples = neg_samples1
    samples = pos_samples + neg_samples   
    #check_sample(neg_samples[0])'''
            
    print('-----------sub-step 3---------')
    is_cat_sample = True
    cat_two_neg = False
    cat_onepos_twoneg = False
    cat_twopos_twoneg = True
    if is_cat_sample:
        random.shuffle(neg_samples)
        random.shuffle(pos_samples)         
        samples = []
        
        if cat_two_neg:
            for i in range(0, len(neg_samples)-1, 2):
                #samp = [pos_samples[i], neg_samples[i], pos_samples[i+1], neg_samples[i+1]]
                samp = [neg_samples[i], neg_samples[i+1]]
                sample = cat_samples(samp)
                samples.append(sample)
            samples += pos_samples   
            
        elif cat_onepos_twoneg:    
            for i in range(0, len(pos_samples)):
                #samp = [pos_samples[i], neg_samples[i], pos_samples[i+1], neg_samples[i+1]]
                samp = [pos_samples[i], neg_samples[2*i], neg_samples[2*i+1]]
                sample = cat_samples(samp)
                samples.append(sample)
            #print('3-1 samples = {0}'.format(len(samples)))
            #print('3-2 pos_samples = {0}'.format(len(pos_samples)))
            
        elif cat_twopos_twoneg:
            for i in range(0, len(pos_samples)-1, 2):
                #samp = [pos_samples[i], neg_samples[i], pos_samples[i+1], neg_samples[i+1]]
                samp = [pos_samples[i], neg_samples[i], pos_samples[i + 1], neg_samples[i + 1]]
                sample = cat_samples(samp)
                samples.append(sample)
            

    #check_sample(neg_samples[0])
    #print('3-3 samples = {0}'.format(len(samples)))
    print('-----------sub-step 4---------')
     
    is_shuffle=True
    if is_shuffle:
        random.shuffle(samples)   
     
   
    flag = True
    while flag:
        flag = False        
        select_neg = False
        if select_neg:
            samples = select_incorrect(samples)
    
        print('-----------sub-step 5---------')
        results = []
        indicators = []
        for j in range(len(samples)):           
            #check_sample(samples[j])        
            
            xsize = samples[j].x.size()
            
            coord = Variable(samples[j].coord).cuda() #torch.cuda.FloatTensor of size 12x5x3x24x24x24 (GPU 0)]
            x = Variable(samples[j].x).cuda()  #[torch.cuda.FloatTensor of size 12x5x1x96x96x96 (GPU 0)]        
            isnod_j = Variable(samples[j].isnod).float().cuda()

            optimizer.zero_grad()
            nodulePred,casePred,casePred_each = model(x,coord)
            loss = binary_cross_entropy(casePred_each,isnod_j)

            samples[j].pred.append(casePred_each)
            samples[j].loss.append(loss)
            
            
            missMask = (casePred_each < 0.03).float()
            missLoss = -torch.sum(missMask * isnod_j * torch.log(casePred_each + 0.001)) / xsize[0] / xsize[1]
            loss = loss + missLoss            
            loss.backward() 
            optimizer.step()

            #print('missLoss.data.cpu().numpy().astype(float)=' , missLoss.data.cpu().numpy().astype(float))
            indic = statistics(casePred_each.data.cpu().numpy(), samples[j].isnod.numpy().astype(int), loss.data.cpu().numpy().astype(float), missLoss.data.cpu().numpy()[0].astype(float)) 
            indicators.append(indic)
            
           
        indic_res = compute_res(indicators)  
        endtime = time.time()
        print('>>>>>>>>>>>>Train, epoch %d, loss2 %.4f, acc %.4f, recall %.4f, precision %.4f, \n>>>>>>>>>>>>tpn %d, fpn %d, fnn %d, tnn %d,  time %3.2f'
              %(epoch, indic_res.mean_loss, indic_res.mean_acc, indic_res.recall, indic_res.precision, indic_res.tpn, indic_res.fpn, indic_res.fnn, indic_res.tnn, endtime-starttime))    
        save_json(save_data, indic_res, save_name)
  

    
def val_casenet(epoch,model,data_loader,args, save_data, save_name, save_name_csv):
    model.eval()
    starttime = time.time()
    
    
    results = []
    indicators = []

    for i,(x,coord,isnod,scores,patient_no, target) in enumerate(data_loader):  
        for j in range(len(isnod)):  
           
            xsize = x[j].size()            
            coord[j] = Variable(coord[j],volatile=True).cuda()
            x[j] = Variable(x[j],volatile=True).cuda()
            isnod_j = Variable(isnod[j]).float().cuda()            
            
            nodulePred,casePred,casePred_each = model(x[j],coord[j])
            loss = binary_cross_entropy(casePred_each,isnod_j) 
            missMask = (casePred_each < 0.03).float()
            missLoss = -torch.sum(missMask * isnod_j * torch.log(casePred_each + 0.001)) / xsize[0] / xsize[1]
            
            indic = statistics(casePred_each.data.cpu().numpy(), isnod[j].numpy().astype(int), loss.data.cpu().numpy().astype(float), missLoss.data.cpu().numpy()[0].astype(float)) 
            indicators.append(indic)
              
            outdata = casePred_each.data.cpu().numpy()
            pred = outdata > 0.5       
            outdata1 = outdata.tolist()[0]
            pred1 = pred.tolist()[0]
            
            for i in range(len(outdata1)): 
                #print('isnod[j].numpy().astype(int)=',isnod[j].numpy()[i][0].astype(int))
                result = {"pScore":0, "Lable":0, "pScore":0, "pID":0, "X":0, "Y":0, "Z":0, "D":0}
                result["pred"] = pred1[i]
                result["pScore"] = outdata1[i]
                result["Lable"] = isnod[j].numpy()[i][0].astype(int)
                               
                target1=target[j].numpy().astype(float)
                result["Z"] = target1[0][0]
                result["X"] = target1[0][1]
                result["Y"] = target1[0][2]
                result["D"] = target1[0][3]
                result["pID"] = patient_no[0]
                
                #print('result = ', result)
                results.append(result)
                         
            
    #print('indicators =', indicators)          
    indic_res = compute_res(indicators)   
    endtime = time.time()
    

    print('Valid, epoch %d, loss2 %.4f, acc %.4f, recall %.4f, precision %.4f, \ntpn %d, fpn %d, fnn %d, tnn %d, time %3.2f'
          %(epoch, indic_res.mean_loss, indic_res.mean_acc, indic_res.recall, indic_res.precision, indic_res.tpn, indic_res.fpn, indic_res.fnn, indic_res.tnn, endtime-starttime))
    save_json(save_data, indic_res, save_name)
    
    #print('results = ', results)
    print('save_name_csv=',save_name_csv)
    file = os.path.join('{0}'.format(save_name_csv))
    if os.path.exists(file):
        os.remove(file)
    fp = open(file, 'w')
    json.dump(results, fp, ensure_ascii=False)
    fp.close()

    
    
def test_casenet(model,testset):
    data_loader = DataLoader(
        testset,
        batch_size = 4,
        shuffle = False,
        num_workers = 32,
        pin_memory=True)
    #model = model.cuda()
    model.eval()
    predlist = []
    
    #     weight = torch.from_numpy(np.ones_like(y).float().cuda()
    for i,(x,coord) in enumerate(data_loader):

        coord = Variable(coord).cuda()
        x = Variable(x).cuda()
        nodulePred,casePred,_ = model(x,coord)
        predlist.append(casePred.data.cpu().numpy())
        #print([i,data_loader.dataset.split[i,1],casePred.data.cpu().numpy()])
    predlist = np.concatenate(predlist)
    return predlist    



def get_model_input(data_loader):
    
    samples_pos = []
    samples_neg = []         
    for i,(x,coord,isnod,scores,patient_no, target) in enumerate(data_loader):
        for j in range(len(x)):
            sample=Sample()
            sample.id_patient.append(patient_no[0])
            sample.id_nodule.append(j)
            sample.score.append(scores[j])
            
            sample.x = x[j] #torch.from_numpy(x[j]).float()  
            sample.coord = coord[j] #torch.from_numpy(coord[j])
            sample.isnod = isnod[j] #torch.from_numpy(isnod[j]) 

            if isnod[j].numpy()[0].astype(int) == 1:
                samples_pos.append(sample)
            else:
                samples_neg.append(sample)
    return samples_pos, samples_neg 


def hard_mining(pos_samples, neg_samples, num):
    random.shuffle(neg_samples)
    neg_samples=neg_samples[0:len(pos_samples)*num*2]
    topk = len(pos_samples) * num
    #print('neg_sample.score = ', neg_samples[0].score[0].numpy()[0])
    neg_samples.sort(key = lambda x: x.score[0].numpy()[0], reverse = True)
    neg_samples = neg_samples[:topk]
    return neg_samples


def select_incorrect(samples):
    
    topk = 0
    for i in range(len(samples)):
        for j in range(len(samples[i].isnod)):
            if not samples[i].isnod[j].numpy()[0] == samples[i].pred[j].numpy()[0]:
                topk += 1
                break
                
    topk = int(topk * 0.1)
    samples2 = samples
    samples2.sort(key = lambda x: x.loss[0].numpy()[0], reverse = True)
    samples += samples2[:topk]
    random.shuffle(samples) 
    return samples

        
    
def cat_samples2(sample1, sample2):
    sample = sample1
    sample.id_patient.append(sample2.id_patient)
    sample.id_nodule.append(sample2.id_nodule)
    sample.score.append(sample2.score)
    
    x1= sample.x.numpy()[0]
    x2= sample2.x.numpy()[0]       
   
    sample.x = np.array([x1, x2])
    sample.x = torch.from_numpy(sample.x).float() 
    
    coord1 = sample.coord.numpy()[0]
    coord2 = sample2.coord.numpy()[0]   
    sample.coord = np.array([coord1, coord2])
    sample.coord = torch.from_numpy(sample.coord).float()
    
    isnod1 = sample.isnod.numpy()[0]
    isnod2 = sample2.isnod.numpy()[0]
    sample.isnod = np.array([isnod1, isnod2])
    sample.isnod = torch.from_numpy(sample.isnod).float()   
                    
    return sample


def cat_samples(samples):    
    sample = Sample()
    sample.x = []
    sample.coord = []
    sample.isnod = []
    
    for i in samples:        
        sample.id_patient.append(i.id_patient)
        sample.id_nodule.append(i.id_nodule)
        sample.score.append(i.score)
            
        sample.x.append(i.x.numpy()[0])                
        sample.coord.append(i.coord.numpy()[0])                
        sample.isnod.append(i.isnod.numpy()[0])
            
    sample.x = torch.from_numpy(np.array(sample.x)).float() 
    sample.coord = torch.from_numpy(np.array(sample.coord)).float()
    sample.isnod = torch.from_numpy(np.array(sample.isnod)).float()   
    
    return sample



        
def statistics(outdata, isnod, loss, missloss):    
    indic = Indicator()    
    pred = outdata > 0.5
    for m in range(pred.shape[0]): 
        if pred[m][0] and isnod[m][0] == 1:
            indic.tpn += 1
        if pred[m][0] and isnod[m][0] == 0:
            indic.fpn += 1
        if not pred[m][0] and isnod[m][0] == 1:
            indic.fnn += 1
        if not pred[m][0] and isnod[m][0] == 0:
            indic.tnn += 1

        indic.loss = loss  
        indic.missloss = missloss
    return indic


def compute_res(indicators):
    indic = Indicator()
    for i in indicators:
        indic.tpn += i.tpn       
        indic.fpn += i.fpn
        indic.fnn += i.fnn
        indic.tnn += i.tnn        
        indic.loss += i.loss
        indic.missloss += i.missloss
        
    indic.mean_acc = float(indic.tpn + indic.tnn) / (indic.tpn + indic.fpn + indic.fnn + indic.tnn)
    indic.mean_loss = float(indic.loss) / (indic.tpn + indic.fpn + indic.fnn + indic.tnn)
     
    indic.recall = 0
    if not indic.tpn + indic.fnn == 0:
        indic.recall = float(indic.tpn) / (indic.tpn + indic.fnn)
    indic.precision = 0
    if not indic.tpn + indic.fpn == 0:
        indic.precision = float(indic.tpn) / (indic.tpn + indic.fpn)
    
    return indic


        
def save_json(save_data, indicators, save_name):
    
    save_data['tpn'].append(indicators.tpn)
    save_data['fpn'].append(indicators.fpn)
    save_data['fnn'].append(indicators.fnn)
    save_data['tnn'].append(indicators.tnn)
    save_data['loss'].append(indicators.mean_loss)
    save_data['missloss'].append(indicators.missloss)
    save_data['acc'].append(indicators.mean_acc)
    save_data['recall'].append(indicators.recall)
    save_data['precision'].append(indicators.precision)

    file = os.path.join('{0}'.format(save_name))
    fp = open(file, 'w')
    json.dump(save_data, fp, ensure_ascii=False)
    fp.close()
           
  
    