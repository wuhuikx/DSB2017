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

train_data = {
    'tpn':[],
    'fpn':[],
    'fnn':[],
    'tnn':[],
    'loss':[],
    'acc':[],
    'recall':[],
    'precision':[]
}

val_data = {
    'tpn':[],
    'fpn':[],
    'fnn':[],
    'tnn':[],
    'loss':[],
    'acc':[],
    'recall':[],
    'precision':[]
}

def save_json(file=None, j={}):
    fp = open(file, 'w')
    json.dump(j, fp, ensure_ascii=False)
    fp.close()

def get_lr(epoch,args):
    assert epoch<=args.lr_stage2[-1]
    if args.lr==None:
        lrstage = np.sum(epoch>args.lr_stage2)
        lr = args.lr_preset2[lrstage]
    else:
        lr = args.lr
    return lr

#def train_casenet(epoch,model,data_loader,optimizer,args):
#    model.train()
#    if args.freeze_batchnorm:    
#        for m in model.modules():
#            if isinstance(m, nn.BatchNorm3d):  # decide if m is the type of nn.BatchNorm3d
#                m.eval()  #fix the parameter of m

#    starttime = time.time()
#    lr = get_lr(epoch,args)
#    for param_group in optimizer.param_groups:
#        param_group['lr'] = lr

#    loss1Hist = []
#    loss2Hist = []
#    missHist = []
#    lossHist = []

#    accHist = []

#lenHist = []
#    tpn = 0
#    fpn = 0
#    fnn = 0
    
    
#    print('---------------train case-net----------------')
#     weight = torch.from_numpy(np.ones_like(y).float().cuda()
#     data_loader contain all the nodules in all the patient
#     for each patient, extract their nodules and x, coord, isnod, y
#    for i,(x,coord,isnod) in enumerate(data_loader):  
        #use function of __getitem__(self, idx,split=None) 
#        print('i=',i)
#        print('x=',x)
#        print('coord=',coord)
#        print('isnod=',isnod)
#        print('----------------------')
        
        # i: the index of patient
        # x: [torch.FloatTensor of size 1x6x1x96x96x96]  
        #    batch_size(1),nodule_number(6),number_of_output(1),input_cube_size(96*96*96)
        # coord:[torch.FloatTensor of size 1x6x3x24x24x24]
        # isnod: 'isnod=' 1 [torch.IntTensor of size 1x6]
        # y: 'y=', (0 ,.,.) = 1 [torch.LongTensor of size 1x1x1]
        #    has cancer or not
        
        # if debug, just use 5 patient information for training
#        if args.debug:
#            if i >4:
#                break
#        j=0
        #for j in range(len(x)):
#        coord[j] = Variable(coord[j]).cuda()
            
         
        #print('coord_cuda=',coord)
        #torch.cuda.FloatTensor of size 12x5x3x24x24x24 (GPU 0)]
#        x[j] = Variable(x[j]).cuda()  
        #[torch.cuda.FloatTensor of size 12x5x1x96x96x96 (GPU 0)]
        #print('x_cuda=',x)
        
#        xsize = x[j].size()
#        isnod_data=isnod.numpy()[:,0]
        #isnod = Variable(isnod).float().cuda()
        #ydata = y.numpy()[:,0]
        #y = Variable(y).float().cuda()
#       weight = 3*torch.ones(y.size()).float().cuda()
        #optimizer.zero_grad()
    
        #print('isnod_data=',isnod_data)
        #print('isnod=',isnod)
        #print('ydata=',ydata)
        #print('y=',y)
        
        #nodulePred,casePred,casePred_each = model(x,coord)  # use function forward
        #print('x=',x) #[torch.cuda.FloatTensor of size 1x6x1x96x96x96 ]
        #print('coord=',coord) #[torch.cuda.FloatTensor of size 1x6x3x24x24x24]
        #print('isnod=',isnod) #[torch.cuda.FloatTensor of size 1x6 (GPU 0)]
        #print('nodulePred=',nodulePred) 
                    #[torch.cuda.FloatTensor of size 1x6x207360=1x6x5x3x24x24x24]
        #print('casePred=',casePred) # casePred=0.8425 [torch.cuda.FloatTensor of size 1x1]
        #print('casePred_each',casePred_each) #'casePred_each' = 0.6281  0.6042  0.4607
                # 0.6222  0.4908  0.4925 [torch.cuda.FloatTensor of size 1x6 (GPU 0)]
        
      
        
        #loss2 = binary_cross_entropy(casePred,y[:,0])
        #missMask = (casePred_each<args.miss_thresh).float()
        #missLoss = -torch.sum(missMask*isnod*torch.log(casePred_each+0.001))/xsize[0]/xsize[1]
        #loss = loss2+args.miss_ratio*missLoss
        #loss=loss2
        #loss.backward()
        #torch.nn.utils.clip_grad_norm(model.parameters(), 1)

        #optimizer.step()
        #loss2Hist.append(loss2.data[0])
        #missHist.append(missLoss.data[0])
        #lenHist.append(len(x))
        #outdata = casePred.data.cpu().numpy()
       
        #pred = outdata>0.5
        #tpn += np.sum(1==pred[ydata==1])
        #fpn += np.sum(1==pred[ydata==0])
        #fnn += np.sum(0==pred[ydata==1])
        #acc = np.mean(ydata==pred)
        #accHist.append(acc)

        
#        loss2 = binary_cross_entropy(casePred,y[:,0])
#        missMask = (casePred_each<args.miss_thresh).float()
#        missLoss = -torch.sum(missMask*isnod*torch.log(casePred_each+0.001))/xsize[0]/xsize[1]
#        loss = loss2+args.miss_ratio*missLoss
#        loss=loss2
#        loss.backward()
#        torch.nn.utils.clip_grad_norm(model.parameters(), 1)

#        optimizer.step()
#        loss2Hist.append(loss2.data[0])
#        missHist.append(missLoss.data[0])
#        lenHist.append(len(x))
        
#        outdata2=casePred_each.data.cpu().numpy()
#        pred2=outdata2>0.5
#        tpn += np.sum(1==pred2[isnod_data==1])
#        fpn += np.sum(1==pred2[isnod_data==0])
#        fnn += np.sum(0==pred2[isnod_data==1])
#        acc = np.mean(isnod_data==pred2)
#        accHist.append(acc)
        
#        print('-------------')
     
    
#      for i in range(len(casePred)):
#              print('casePred+1',range(len(casePred)))
#             loss2 = binary_cross_entropy(casePred,y[:,0])
#             loss=loss2
#             loss.backward()
#             #torch.nn.utils.clip_grad_norm(model.parameters(), 1)

#             optimizer.step()
#             loss2Hist.append(loss2.data[0])
#             lenHist.append(len(x))
#             outdata = casePred_each[i].data.cpu().numpy()
       
#             pred = outdata>0.5
#             tpn += np.sum(1==pred&&isnod[0][i]==1)
#             fpn += np.sum(1==pred[ydata==0])
#             fnn += np.sum(0==pred[ydata==1])
#             acc = np.mean(ydata==pred)
#             accHist.append(acc)
    
    
#    endtime = time.time()
#    lenHist = np.array(lenHist)
#    loss2Hist = np.array(loss2Hist)
#    lossHist = np.array(lossHist)
#    accHist = np.array(accHist)
    
#    mean_loss2 = np.sum(loss2Hist*lenHist)/np.sum(lenHist)
#    mean_missloss = np.sum(missHist*lenHist)/np.sum(lenHist)
#    mean_acc = np.sum(accHist*lenHist)/np.sum(lenHist)
#    print('Train, epoch %d, loss2 %.4f, miss loss %.4f, acc %.4f, tpn %d, fpn %d, fnn %d, time %3.2f, lr % .5f '
#          %(epoch,mean_loss2,mean_missloss,mean_acc,tpn,fpn, fnn, endtime-starttime,lr))
def count_sample(epoch,model,data_loader,optimizer,args):
    # count of positive/negative
    count_pos=0
    count_neg=0
    for i,(x,coord,isnod) in enumerate(data_loader):
        for j in range(len(x)):
            isnod_j = Variable(isnod[j]).float().cuda()
            if isnod_j.data.cpu().numpy()[0].astype(int) == 1:
                count_pos+=1
            else:
                count_neg+=1
    
    count_all=count_pos + count_neg
    pos_rate=float(count_pos) / float(count_all)
    neg_rate=float(count_neg) / float(count_all)
    
    print('sample, all %d, positive %d, positive_rate %.4f, negative %d, negative_rate %.4f'
         %(count_all, count_pos, pos_rate, count_neg, neg_rate))
    
        
def train_casenet(epoch,model,data_loader,optimizer,args):
    model.train()
    if args.freeze_batchnorm:    
        for m in model.modules():
            if isinstance(m, nn.BatchNorm3d):  # decide if m is the type of nn.BatchNorm3d
                m.eval()  #fix the parameter of m

    starttime = time.time()
    lr = get_lr(epoch,args)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    loss1Hist = []
    loss2Hist = []
    missHist = []
    lossHist = []
    accHist = []
    lenHist = []
    tpn = 0
    fpn = 0
    fnn = 0
    tnn = 0
    
    count_all=0
    nodule_all = 0
    loss_all = 0
    count_true_nod=0
    count_pred_nod=0
    
    count_pos = 0
    count_neg = 0
#   weight = torch.from_numpy(np.ones_like(y).float().cuda()
    for i,(x,coord,isnod) in enumerate(data_loader):
        if args.debug:
            if i >4:
                break
        
        nodule_all += len(x)
        #lenHist.append(len(x))
        count = 0
        #print('--------len(x)--------=',len(x))
        if len(x) == 0:
            continue

        
        for j in range(len(x)):       
            coord[j] = Variable(coord[j]).cuda() #torch.cuda.FloatTensor of size 12x5x3x24x24x24 (GPU 0)]
            x[j] = Variable(x[j]).cuda()  #[torch.cuda.FloatTensor of size 12x5x1x96x96x96 (GPU 0)]
            #print('x=',x[j])
            #print('x=',x[j].size())

            xsize = x[j].size()
            #print('------------------------')
            #print('isnod[j]=', isnod[j])
 
            if isnod[j].numpy()[0]==1:
                count_pos+=1
            else:
                count_neg+=1
        
            isnod_j = Variable(isnod[j]).float().cuda()

            #ydata = y.numpy()[:,0]
            #y = Variable(y).float().cuda()
    #         weight = 3*torch.ones(y.size()).float().cuda()
            optimizer.zero_grad()
            #print('----------before model--------------')
            # bug: size to small
            nodulePred,casePred,casePred_each = model(x[j],coord[j])
            print('isnod_j=',isnod_j)
            print('casePred_each=',casePred_each.data.cpu().numpy()[0].astype(float))


            #loss2 = binary_cross_entropy(casePred,y[:,0])
            loss2 = binary_cross_entropy(casePred_each,isnod_j)
            # print('loss=',loss2)
            #missMask = (casePred_each<args.miss_thresh).float()
            #missLoss = -torch.sum(missMask*isnod*torch.log(casePred_each+0.001))/xsize[0]/xsize[1]
            #loss = loss2+args.miss_ratio*missLoss
            loss=loss2
            #print('loss=',loss)
            loss_all += loss.data.cpu().numpy()[0].astype(float)
            loss.backward()
            #torch.nn.utils.clip_grad_norm(model.parameters(), 1)

            optimizer.step()
            loss2Hist.append(loss2.data[0])
            #missHist.append(missLoss.data[0])
            outdata = casePred_each.data.cpu().numpy()
            pred = outdata>0.5

            #print('pred=',pred)
            #print('isnod[j]=',isnod_j.data.cpu().numpy())

            if pred[0][0] and isnod_j.data.cpu().numpy()[0].astype(int) == 1:
                tpn += 1
            if pred[0][0] and isnod_j.data.cpu().numpy()[0] == 0:
                fpn += 1
            if not pred[0][0] and isnod_j.data.cpu().numpy()[0].astype(int) == 1:
                fnn += 1
            
            if pred[0][0].astype(int) == isnod_j.data.cpu().numpy()[0].astype(int):
                count += 1
                count_all += 1
                
            if isnod_j.data.cpu().numpy()[0].astype(int) == 1:
                count_true_nod += 1
            if pred[0][0].astype(int) == 1:
                count_pred_nod += 1
            
        
        #tpn += np.sum(1==pred[ydata==1])
        #fpn += np.sum(1==pred[ydata==0])
        #fnn += np.sum(0==pred[ydata==1])
        #acc = np.mean(ydata==pred)
        acc= float(count) / float(len(x))
        accHist.append(acc)
    
    #print('loss_all=',loss_all)
    #print('nodule_all=',nodule_all)
    #print('loss_all_type',type(loss_all))
    #print('nodule_all_type=',type(nodule_all))
    mean_acc = float(count_all) / float(nodule_all)
    mean_loss2 = float(loss_all) / float(nodule_all)
    if count_true_nod == 0:
        recall = 1
    else:
        recall = float(tpn) / count_true_nod
    if count_pred_nod == 0:
        precision = 0
    else:
        precision = float(tpn) / count_pred_nod
    
    
    endtime = time.time()
    #lenHist = np.array(lenHist)
    #loss2Hist = np.array(loss2Hist)
    #lossHist = np.array(lossHist)
    #accHist = np.array(accHist)
    
    #mean_loss2 = np.sum(loss2Hist*lenHist)/np.sum(lenHist)
    #mean_missloss = np.sum(missHist*lenHist)/np.sum(lenHist)
    #mean_acc = np.sum(accHist*lenHist)/np.sum(lenHist)
    #print('Train, epoch %d, loss2 %.4f, miss loss %.4f, acc %.4f, tpn %d, fpn %d, fnn %d, time %3.2f, lr % .5f '
     #     %(epoch,mean_loss2,mean_missloss,mean_acc,tpn,fpn, fnn, endtime-starttime,lr))
    print('Train, epoch %d, loss2 %.4f, acc %.4f, recall %.4f, precision %.4f, tpn %d, fpn %d, fnn %d, total pos %d, total neg %d, time %3.2f, lr % .5f '
          %(epoch,mean_loss2,mean_acc,recall,precision,tpn,fpn, fnn, count_pos, count_neg, endtime-starttime,lr))
    
   
    #### zhaifly
    train_data['tpn'].append(tpn)
    train_data['fpn'].append(fpn)
    train_data['fnn'].append(fnn)
    
    train_data['loss'].append(mean_loss2)
    train_data['acc'].append(mean_acc)
    train_data['recall'].append(recall)
    train_data['precision'].append(precision)

    save_json('./casenet_train_loss.json', train_data)
    ### zhaifly

class Sample(object):
    def __init__(self):
        self.id = 0
        self.x = torch.from_numpy(np.zeros(1)).float()
        self.coord = torch.from_numpy(np.zeros(1)).float()
        self.isnod = torch.from_numpy(np.zeros(1)).float()
        self.score = 0.0
        self.id_patient = ''
        self.id_nodule = 0
        
            

def train_casenet_shuffle(epoch,model,data_loader,optimizer,args,save_name):
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

    loss1Hist = []
    loss2Hist = []
    missHist = []
    lossHist = []
    accHist = []
    lenHist = []
    tpn = 0
    fpn = 0
    fnn = 0
    tnn = 0
    
    count_all=0
    nodule_all = 0
    loss_all = 0
    count_true_nod=0
    count_pred_nod=0
    
    count_pos = 0
    count_neg = 0
#   weight = torch.from_numpy(np.ones_like(y).float().cuda()
    samples=[]
    print('-----------sub-step 1---------') 
    for i,(x,coord,isnod,scores,patient_no) in enumerate(data_loader):
        for j in range(len(x)):
            sample=Sample()
            sample.id_patient = patient_no[0]
            sample.id_nodule = j
            sample.x = x[j]
            sample.coord = coord[j]
            sample.isnod = isnod[j]
            sample.score = scores[j].numpy()[0]
            samples.append(sample)
           

            '''np.save(os.path.join('./samples/{0}/sample_x_{1}.npy'.format(patient_no[0],j)),sample.x.numpy())
            np.save(os.path.join('./samples/{0}/sample_label_{1}.npy'.format(patient_no[0],j)),sample.isnod.numpy())'''
            #if sample.id_patient == '0482c444ac838adc5aa00d1064c976c1':
            #    print(sample.coord)
            
            #print('i={0}\tj={1}\tsample_isnod='.format(i,j,sample.isnod.numpy()[0]))
            
            #print(patient_no)
            '''if patient_no[0]=='7917af5df3fe5577f912c5168c6307e3':
                print('isnod1=',sample.isnod.numpy()[0].astype(int))
                if j ==4:
                    np.save('./samples/sample_x.npy',sample.x.numpy())
                    print('isnod2=',sample.isnod.numpy()[0])
                    print('--------------------------------')
            '''

            
           
    print('-----------sub-step 2---------')
    pos_samples = []
    neg_samples = []
    neg_scores = []
    for sample in samples:
        if sample.isnod.numpy()[0].astype(int):
            pos_samples.append(sample)
        elif not sample.isnod.numpy()[0]:
            neg_samples.append(sample)
            neg_scores.append(sample.score)
    #print('pos_num={0}\tneg_num={1}'.format(len(pos_samples),len(neg_samples)))
    
    print('-----------sub-step 3---------')
    random.shuffle(neg_samples)
    neg_samples=neg_samples[0:len(pos_samples)*4]
    if hard_mining:
        topk = len(pos_samples) * 2
        neg_samples.sort(key = lambda x: x.score, reverse = True)
        #print(neg_samples)
        neg_samples = neg_samples[:topk]
    

    #---------------check smaples---------------#    
    '''
    import shutil
    ischeck_sample = True
    if ischeck_sample:
        sample_dir = './samples/pos'
        if os.path.exists(sample_dir):
            shutil.rmtree(sample_dir)
        os.mkdir(sample_dir)
        i = 1
        for one_sample in pos_samples:
            sample = one_sample.x.numpy()           
            #for z in range(sample.shape[-1]):
            z=sample.shape[-1]
            one_pic = sample[0,0,0,int(z/2),:,:]
            np.save(os.path.join(sample_dir,'{0}_{1}.npy'.format(i,z+1)), one_pic)
            i += 1
        
        sample_dir = './samples/neg'
        if os.path.exists(sample_dir):
            shutil.rmtree(sample_dir)
        os.mkdir(sample_dir)
        i = 1
        for one_sample in neg_samples:
            sample = one_sample.x.numpy()
            #for z in range(sample.shape[-1]):
            z=sample.shape[-1]
            one_pic = sample[0,0,0,int(z/2),:,:]
            np.save(os.path.join(sample_dir,'{0}_{1}.npy'.format(i,z+1)), one_pic)
            i += 1
    
    return'''
            
    
    #samples = pos_samples + neg_samples
    is_shuffle=True
    pos_samples.extend(neg_samples)
    samples = pos_samples
    if is_shuffle:
        random.shuffle(samples)    
    print('-----------sub-step 4---------')
        
    #lenHist.append(len(x))
    count = 0
    count_all = 0
    #print('--------len(x)--------=',len(x))
 

    '''for sample in samples:
        np.save(os.path.join('./samples/{0}/sample_shuffle_{1}.npy'.format(sample.id_patient,sample.id_nodule)),sample.x.numpy())
        np.save(os.path.join('./samples/{0}/sample_shuffle_label_{1}.npy'.format(sample.id_patient,sample.id_nodule)),sample.isnod.numpy())
        #if sample.id_patient == '7917af5df3fe5577f912c5168c6307e3' and sample.id_nodule == 4:
        #print(os.path.join('./samples/{0}/sample_shuffle_{1}.npy'.format(sample.id_patient,sample.id_nodule)))
    '''
    
    #print('----------end----')
        
    count_pos=0
    count_neg=0
    for j in range(len(samples)):       
        coord = Variable(samples[j].coord).cuda() #torch.cuda.FloatTensor of size 12x5x3x24x24x24 (GPU 0)]
        x = Variable(samples[j].x).cuda()  #[torch.cuda.FloatTensor of size 12x5x1x96x96x96 (GPU 0)]
            #print('x=',x[j])
            #print('x=',x[j].size())

        xsize = x.size()
        #print('------------------------')
        #print('isnod[j]=', isnod[j])
 
        if samples[j].isnod.numpy()[0]==1:
            count_pos+=1
        else:
            count_neg+=1
        
        isnod_j = Variable(samples[j].isnod).float().cuda()
        #ydata = y.numpy()[:,0]
        #y = Variable(y).float().cuda()
    #         weight = 3*torch.ones(y.size()).float().cuda()
        optimizer.zero_grad()
        #print('----------before model--------------')
         # bug: size to small
        nodulePred,casePred,casePred_each = model(x,coord)
        #print('isnod_j=',isnod_j)
        #print('casePred_each=',casePred_each.data.cpu().numpy()[0].astype(float))


            #loss2 = binary_cross_entropy(casePred,y[:,0])
        loss2 = binary_cross_entropy(casePred_each,isnod_j)
            # print('loss=',loss2)
            #missMask = (casePred_each<args.miss_thresh).float()
            #missLoss = -torch.sum(missMask*isnod*torch.log(casePred_each+0.001))/xsize[0]/xsize[1]
            #loss = loss2+args.miss_ratio*missLoss
        loss=loss2
            #print('loss=',loss)
        loss_all += loss.data.cpu().numpy()[0].astype(float)
        loss.backward()
            #torch.nn.utils.clip_grad_norm(model.parameters(), 1)

        optimizer.step()
        #missHist.append(missLoss.data[0])
        outdata = casePred_each.data.cpu().numpy()
        pred = outdata>0.5
        
        #print('2_fc2=',model.module.state_dict()['fc2.weight']) 
        
            #print('pred=',pred)
            #print('isnod[j]=',isnod_j.data.cpu().numpy())

        if pred[0][0] and isnod_j.data.cpu().numpy()[0].astype(int) == 1:
            tpn += 1
        if pred[0][0] and isnod_j.data.cpu().numpy()[0] == 0:
            fpn += 1
        if not pred[0][0] and isnod_j.data.cpu().numpy()[0].astype(int) == 1:
            fnn += 1
            
        if pred[0][0].astype(int) == isnod_j.data.cpu().numpy()[0].astype(int):
            count += 1
            count_all += 1
                
        if isnod_j.data.cpu().numpy()[0].astype(int) == 1:
            count_true_nod += 1
        if pred[0][0].astype(int) == 1:
            count_pred_nod += 1
            
        
        #tpn += np.sum(1==pred[ydata==1])
        #fpn += np.sum(1==pred[ydata==0])
        #fnn += np.sum(0==pred[ydata==1])
        #acc = np.mean(ydata==pred)
        acc= float(count) / float(len(x))
        accHist.append(acc)
    
    #print('loss_all=',loss_all)
    #print('nodule_all=',nodule_all)
    #print('loss_all_type',type(loss_all))
    #print('nodule_all_type=',type(nodule_all))
    nodule_all = len(samples)
    mean_acc = float(count_all) / float(nodule_all)
    mean_loss2 = float(loss_all) / float(nodule_all)
    if count_true_nod == 0:
        recall = 1
    else:
        recall = float(tpn) / count_true_nod
    if count_pred_nod == 0:
        precision = 0
    else:
        precision = float(tpn) / count_pred_nod
    
    
    endtime = time.time()
    tnn = nodule_all - fpn - fnn -tpn
    #lenHist = np.array(lenHist)
    #loss2Hist = np.array(loss2Hist)
    #lossHist = np.array(lossHist)
    #accHist = np.array(accHist)
    
    #mean_loss2 = np.sum(loss2Hist*lenHist)/np.sum(lenHist)
    #mean_missloss = np.sum(missHist*lenHist)/np.sum(lenHist)
    #mean_acc = np.sum(accHist*lenHist)/np.sum(lenHist)
    #print('Train, epoch %d, loss2 %.4f, miss loss %.4f, acc %.4f, tpn %d, fpn %d, fnn %d, time %3.2f, lr % .5f '
     #     %(epoch,mean_loss2,mean_missloss,mean_acc,tpn,fpn, fnn, endtime-starttime,lr))
    print('>>>>>>>>>>>>Train, epoch %d, loss2 %.4f, acc %.4f, recall %.4f, precision %.4f, \n>>>>>>>>>>>>tpn %d, fpn %d, fnn %d, tnn %d, total pos %d, total neg %d, time %3.2f, lr % .5f '
          %(epoch,mean_loss2,mean_acc,recall,precision,tpn,fpn, fnn, tnn, count_pos, count_neg, endtime-starttime,lr))
    
   
    #### zhaifly
    train_data['tpn'].append(tpn)
    train_data['fpn'].append(fpn)
    train_data['fnn'].append(fnn)
    train_data['tnn'].append(tnn)
    train_data['loss'].append(mean_loss2)
    train_data['acc'].append(mean_acc)
    train_data['recall'].append(recall)
    train_data['precision'].append(precision)

    save_json(os.path.join('./json/{0}.json'.format(save_name)), train_data)
    ### zhaifly
    
def hard_mining(neg_output, neg_loss, num_hard):
    neg_output1=[]
    neg_loss1=[]
    
    for i in range(num_hard):
        #print('max(neg_output)=',max(neg_output))
        idx=neg_output.index(max(neg_output))
        neg_output1.append(neg_output[idx])
        neg_loss1.append(neg_loss[idx])
        del neg_output[idx]
        del neg_loss[idx]

    #_, idcs = torch.topk(neg_output, min(num_hard, len(neg_output)))
    #neg_output = torch.index_select(neg_output, 0, idcs)
    #neg_labels = torch.index_select(neg_loss, 0, idcs)
    return neg_output1, neg_loss1

def train_casenet_patient(epoch,model,data_loader,optimizer,args):
    model.train()
    if args.freeze_batchnorm:    
        for m in model.modules():
            if isinstance(m, nn.BatchNorm3d):  # decide if m is the type of nn.BatchNorm3d
                m.eval()  #fix the parameter of m

    starttime = time.time()
    lr = get_lr(epoch,args)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    loss1Hist = []
    loss2Hist = []
    missHist = []
    lossHist = []
    accHist = []
    lenHist = []
    tpn = 0
    fpn = 0
    fnn = 0
    tnn = 0
    
    count_all=0
    nodule_all = 0
    loss_all = 0
    count_true_nod=0
    count_pred_nod=0
    
    count_pos = 0
    count_neg = 0
#   weight = torch.from_numpy(np.ones_like(y).float().cuda()
  
    for i,(x,coord,isnod) in enumerate(data_loader):
        #print('x=',len(x))
        
        pos = 0
        neg = 0
        for node in isnod:
            print(node.numpy()[0])
            if node.numpy()[0] == 1:
                pos +=1
            elif node.numpy()[0] == 0:

                neg +=1
        
        #print('pos={0}\tneg={1}\n'.format(pos, neg))
        continue
                
        
        if args.debug:
            if i >4:
                break
        
        nodule_all += len(x)
        #lenHist.append(len(x))
        count = 0
        #print('--------len(x)--------=',len(x))
        if len(x) == 0:
            continue

        
        '''loss3=[]
        predOut=np.zeros((len(x)))
        #print('predOut=',predOut)
        isnodLabel=np.zeros((len(x)))
        for j in range(len(x)):
            coord[j] = Variable(coord[j]).cuda() #torch.cuda.FloatTensor of size 12x5x3x24x24x24 (GPU 0)]
            x[j] = Variable(x[j]).cuda()  #[torch.cuda.FloatTensor of size 12x5x1x96x96x96 (GPU 0)]           
            xsize = x[j].size()    
            isnod_j = Variable(isnod[j]).float().cuda()
            isnodLabel[j]=isnod_j.data.cpu().numpy()[0].astype(float)
            if isnod[j].numpy()[0]==1:
                count_pos+=1
            else:
                count_neg+=1
                
            nodulePred,casePred,casePred_each = model(x[j],coord[j])
            predOut[j]=casePred_each.data.cpu().numpy()[0].astype(float)
            #predOut[j]=casePred_each
            
            loss2 = binary_cross_entropy(casePred_each,isnod_j)
            loss3.append(loss2)
        '''
        
        pos_loss=[]
        neg_loss=[]
        pos_pred=[]
        neg_pred=[]
        
        predOut=np.zeros((len(x)))
        #print('predOut=',predOut)
        isnodLabel=np.zeros((len(x)))
        for j in range(len(x)):
            coord[j] = Variable(coord[j]).cuda() #torch.cuda.FloatTensor of size 12x5x3x24x24x24 (GPU 0)]
            x[j] = Variable(x[j]).cuda()  #[torch.cuda.FloatTensor of size 12x5x1x96x96x96 (GPU 0)]           
            xsize = x[j].size()    
            isnod_j = Variable(isnod[j]).float().cuda()
            isnodLabel[j]=isnod_j.data.cpu().numpy()[0].astype(float)
            #print('isnod[j]=',isnod_j.data.cpu().numpy()[0].astype(int))
            if isnod[j].numpy()[0]==1:
                #print('------------1----------')
                count_pos+=1
                nodulePred,casePred,casePred_each = model(x[j],coord[j])            
                loss = binary_cross_entropy(casePred_each,isnod_j)
                pos_pred.append(casePred_each.data.cpu().numpy()[0].astype(float))
                pos_loss.append(loss)                
            else:
                #print('------------2----------')
                count_neg+=1
                nodulePred,casePred,casePred_each = model(x[j],coord[j])            
                loss = binary_cross_entropy(casePred_each,isnod_j)
                #print('neg_pred_each=',casePred_each)
                neg_pred.append(casePred_each.data.cpu().numpy()[0].astype(float))
                neg_loss.append(loss)  
                
            #nodulePred,casePred,casePred_each = model(x[j],coord[j])
            #predOut[j]=casePred_each.data.cpu().numpy()[0].astype(float)
            #predOut[j]=casePred_each
            
            #loss2 = binary_cross_entropy(casePred_each,isnod_j)
            #loss3.append(loss2)
        
        num_hard=10
        train=True
        batch_size=1
        if len(neg_pred) > num_hard and num_hard > 0 and train:
            #print('neg_pred=',neg_pred)
            neg_pred, neg_loss = hard_mining(neg_pred, neg_loss, num_hard * batch_size)
        
        
        loss4=0
        for j in range(len(pos_loss)):
            loss4 += pos_loss[j]
        for j in range(len(neg_loss)):
            loss4 += neg_loss[j]
        if loss4 > 0:
            optimizer.zero_grad()
            loss4.backward()
            optimizer.step()    
            
        print('pos_count=',len(pos_loss))
        print('neg_count=',len(neg_loss))
        #print('-----------------')
        
        
        '''
        print('predOut=',predOut)
        print('isnodLabel=',isnodLabel)
        
        pred=Variable(torch.from_numpy(predOut).float()).float().cuda()
        label=Variable(torch.from_numpy(isnodLabel).float()).float().cuda()     
        print('pred=',pred)
        print('label=',label)
        
        optimizer.zero_grad()
        #nodulePred,casePred,casePred_each = model(x[0],coord[0])
        loss2 = binary_cross_entropy(pred,label)
        #loss2=binary_cross_entropy(casePred_each,Variable(isnod[0]).float().cuda())
        print('loss=',loss2)
        loss=loss2
        loss_all += loss.data.cpu().numpy()[0].astype(float)
        loss.backward()
        #torch.nn.utils.clip_grad_norm(model.parameters(), 1)
        '''
        
        '''
        for j in range(len(loss3)):
            optimizer.zero_grad()
            loss3[j].backward()
            optimizer.step()
         
        print('----------end---------')
        '''
        
        '''loss4=loss3[0]
        for j in range(1,len(loss3)):
            loss4 += loss3[j]
        optimizer.zero_grad()
        loss4.backward()
        optimizer.step()    
        print('----------end---------')
        '''
        
        #loss2Hist.append(loss2.data[0])
        #missHist.append(missLoss.data[0])
        outdata = casePred_each.data.cpu().numpy()
        pred = outdata>0.5

            #print('pred=',pred)
            #print('isnod[j]=',isnod_j.data.cpu().numpy())

        if pred[0][0] and isnod_j.data.cpu().numpy()[0].astype(int) == 1:
            tpn += 1
        if pred[0][0] and isnod_j.data.cpu().numpy()[0] == 0:
            fpn += 1
        if not pred[0][0] and isnod_j.data.cpu().numpy()[0].astype(int) == 1:
            fnn += 1

        if pred[0][0].astype(int) == isnod_j.data.cpu().numpy()[0].astype(int):
            count += 1
            count_all += 1

        if isnod_j.data.cpu().numpy()[0].astype(int) == 1:
            count_true_nod += 1
        if pred[0][0].astype(int) == 1:
            count_pred_nod += 1
            
        
        #tpn += np.sum(1==pred[ydata==1])
        #fpn += np.sum(1==pred[ydata==0])
        #fnn += np.sum(0==pred[ydata==1])
        #acc = np.mean(ydata==pred)
        acc= float(count) / float(len(x))
        accHist.append(acc)
    
    #print('loss_all=',loss_all)
    #print('nodule_all=',nodule_all)
    #print('loss_all_type',type(loss_all))
    #print('nodule_all_type=',type(nodule_all))
    mean_acc = float(count_all) / float(nodule_all)
    mean_loss2 = float(loss_all) / float(nodule_all)
    if count_true_nod == 0:
        recall = 1
    else:
        recall = float(tpn) / count_true_nod
    if count_pred_nod == 0:
        precision = 0
    else:
        precision = float(tpn) / count_pred_nod
    
    
    endtime = time.time()
    #lenHist = np.array(lenHist)
    #loss2Hist = np.array(loss2Hist)
    #lossHist = np.array(lossHist)
    #accHist = np.array(accHist)
    
    #mean_loss2 = np.sum(loss2Hist*lenHist)/np.sum(lenHist)
    #mean_missloss = np.sum(missHist*lenHist)/np.sum(lenHist)
    #mean_acc = np.sum(accHist*lenHist)/np.sum(lenHist)
    #print('Train, epoch %d, loss2 %.4f, miss loss %.4f, acc %.4f, tpn %d, fpn %d, fnn %d, time %3.2f, lr % .5f '
     #     %(epoch,mean_loss2,mean_missloss,mean_acc,tpn,fpn, fnn, endtime-starttime,lr))
    print('Train, epoch %d, loss2 %.4f, acc %.4f, recall %.4f, precision %.4f, tpn %d, fpn %d, fnn %d, total pos %d, total neg %d, time %3.2f, lr % .5f '
          %(epoch,mean_loss2,mean_acc,recall,precision,tpn,fpn, fnn, count_pos, count_neg, endtime-starttime,lr))
    
   
    #### zhaifly
    train_data['tpn'].append(tpn)
    train_data['fpn'].append(fpn)
    train_data['fnn'].append(fnn)
    
    
    train_data['loss'].append(mean_loss2)
    train_data['acc'].append(mean_acc)
    train_data['recall'].append(recall)
    train_data['precision'].append(precision)

    save_json('./casenet_train_loss.json', train_data)
    ### zhaifly

    
def val_casenet(epoch,model,data_loader,args,save_name):
    model.eval()
    starttime = time.time()
    loss1Hist = []
    loss2Hist = []
    lossHist = []
    missHist = []
    accHist = []
    lenHist = []
    tpn = 0
    fpn = 0
    fnn = 0
    tnn = 0

    count_all=0
    nodule_all=0
    loss_all=0    
    count_true_nod=0
    count_pred_nod=0

    
    for i,(x,coord,isnod,scores,patient_no) in enumerate(data_loader):       
        
        nodule_all +=len(x)
        count=0
        for j in range(len(x)):     
            
            #if not os.path.exists(os.path.join('./samples/val_set/{0}'.format(patient_no[0]))):
            #    os.mkdir(os.path.join('./samples/val_set/{0}'.format(patient_no[0])))                    
            np.save(os.path.join('./val_set/{0}/3x_{1}.npy'.format(patient_no[0],j)),x[j].numpy())
            np.save(os.path.join('./val_set/{0}/3label_{1}.npy'.format(patient_no[0],j)),isnod[j].numpy())
            
            coord[j] = Variable(coord[j],volatile=True).cuda()
            x[j] = Variable(x[j],volatile=True).cuda()
            xsize = x[j].size()
            isnod_j = Variable(isnod[j]).float().cuda()
            
            
            nodulePred,casePred,casePred_each = model(x[j],coord[j])

            loss2 = binary_cross_entropy(casePred_each,isnod_j)
            loss_all+=loss2.data.cpu().numpy()[0].astype(float)
            
            #missMask = (casePred_each<args.miss_thresh).float()
            #missLoss = -torch.sum(missMask*isnod*torch.log(casePred_each+0.001))/xsize[0]/xsize[1]

        #loss2 = binary_cross_entropy(sigmoid(casePred),y[:,0])
        #loss2Hist.append(loss2.data[0])
        #missHist.append(missLoss.data[0])
        #lenHist.append(len(x))
        outdata = casePred_each.data.cpu().numpy()
        #print([i,data_loader.dataset.split[i,1],sigmoid(casePred).data.cpu().numpy()])
        pred = outdata>0.5
        if pred[0][0] and isnod_j.data.cpu().numpy()[0].astype(int) == 1:
            tpn += 1
        if pred[0][0] and isnod_j.data.cpu().numpy()[0] == 0:
            fpn += 1
        if not pred[0][0] and isnod_j.data.cpu().numpy()[0].astype(int) == 1:
            fnn += 1
        if not pred[0][0] and isnod_j.data.cpu().numpy()[0].astype(int) == 0:
            tnn += 1
            
        if pred[0][0].astype(int) == isnod_j.data.cpu().numpy()[0].astype(int):
            count += 1
            count_all += 1
        if isnod_j.data.cpu().numpy()[0].astype(int) == 1:
            count_true_nod += 1
        if pred[0][0].astype(int) == 1:
            count_pred_nod += 1
                
        #tpn += np.sum(1==pred[ydata==1])
        #fpn += np.sum(1==pred[ydata==0])
        #fnn += np.sum(0==pred[ydata==1])
        #acc = np.mean(ydata==pred)
        #accHist.append(acc)
    endtime = time.time()
    #tnn = nodule_all - tpn - fpn - fnn
    #lenHist = np.array(lenHist)
    #loss2Hist = np.array(loss2Hist)
    #accHist = np.array(accHist)
    #mean_loss2 = np.sum(loss2Hist*lenHist)/np.sum(lenHist)
    #mean_missloss = np.sum(missHist*lenHist)/np.sum(lenHist)
    #mean_acc = np.sum(accHist*lenHist)/np.sum(lenHist)
    
    
    #acc= float(count) / float(len(x))
    mean_acc = 0
    mean_loss2 = 0
    if not nodule_all == 0:
        mean_acc = float(count_all) / float(nodule_all)
        mean_loss2 = float(loss_all) / float(nodule_all)

    if count_true_nod == 0:
        recall = 1
    else:
        recall = float(tpn) / count_true_nod
    if count_pred_nod == 0:
        precision =0
    else:
        precision = float(tpn) / count_pred_nod
    
    
    print('Valid, epoch %d, loss2 %.4f, acc %.4f, recall %.4f, precision %.4f, \ntpn %d, fpn %d, fnn %d, tnn %d, time %3.2f'
          %(epoch,mean_loss2,mean_acc,recall,precision,tpn,fpn, fnn, tnn, endtime-starttime))

    #### zhaifly
    val_data['tpn'].append(tpn)
    val_data['fpn'].append(fpn)
    val_data['fnn'].append(fnn)
    val_data['tnn'].append(tnn)
    
    val_data['loss'].append(mean_loss2)
    val_data['acc'].append(mean_acc)
    val_data['recall'].append(recall)
    val_data['precision'].append(precision)

    #save_json('./casenet_val_loss.json', val_data)
    save_json(os.path.join('./json/{0}.json'.format(save_name)),val_data)
    ### zhaifly
    
    
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
