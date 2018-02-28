import numpy as np
import os
import time
import random
import warnings

import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.nn.functional import cross_entropy,sigmoid,binary_cross_entropy
from torch.utils.data import DataLoader
from layers import Loss
import json

def get_lr(epoch,args):
    assert epoch<=args.lr_stage2[-1]
    if args.lr==None:
        lrstage = np.sum(epoch>args.lr_stage2)
        lr = args.lr_preset2[lrstage]
    else:
        lr = args.lr
    return lr

def hard_mining(neg_output, neg_labels, num_hard):
    _, idcs = torch.topk(neg_output, min(num_hard, len(neg_output)))
    #print('idcs =', idcs)
    neg_output = torch.index_select(neg_output, 0, idcs)
    neg_labels = torch.index_select(neg_labels, 0, idcs)
    
    return neg_output, neg_labels

'''def random_select(neg_output, neg_labels, num_hard):
    _, idcs = torch.topk(neg_output, min(num_hard, len(neg_output)))
    neg_output = torch.index_select(neg_output, 0, idcs)
    neg_labels = torch.index_select(neg_labels, 0, idcs)
    
    return neg_output, neg_labels'''

def train_casenet(epoch,model,data_loader,optimizer,args, save_data, save_name):
    model.train()
    '''if args.freeze_batchnorm:    
        for m in model.modules():
            if isinstance(m, nn.BatchNorm3d):
                m.eval()'''

    starttime = time.time()
    lr = get_lr(epoch,args)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    num_hard = 2
    loss = Loss(num_hard)
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
#     weight = torch.from_numpy(np.ones_like(y).float().cuda()
    for i, (data, target, coord) in enumerate(data_loader):
        #print('patient_id =', patient_id)
        if args.debug:
            if i >4:
                break
        data = Variable(data.cuda(async = True)) 
        #data:torch.cuda.FloatTensor of size 4x1x128x128x128 (GPU 0)
        # 4(batch-size)  1(channel)  128x128x128(input patch size)
        
        target = Variable(target.cuda(async = True))
        #target:torch.cuda.FloatTensor of size 4x32x32x32x3x5 (GPU 0)
        # 4:batch-size;32*32*32:output cube;3:anchors;5:(0,1,x,y,z)
        
        coord = Variable(coord.cuda(async = True))
        #coord:torch.cuda.FloatTensor of size 4x3x32x32x32 (GPU 0)
        # 4:batch-size;3:(x,y,z);32*32*32:output cube;
         
        
        optimizer.zero_grad()
        #print('x =', x.size())
        #print('coord = ', coord.size())
        nodulePred,casePred,casePred_each = model(data, coord, target)
        #print('casePred_each =', casePred_each)
        #print('isnod =', isnod)
        #print('scores =', scores)
        len_x = len(x)
        del x, coord, nodulePred, casePred
       
        loss_output = loss(output, target)
        
        
        outdata = casePred_each
        
        pos_idcs = isnod[0,:] > 0.5
        pos_output = outdata[0, :][pos_idcs]
        pos_labels = isnod[0, :][pos_idcs]
        

        neg_idcs = isnod[0,:] < 0.5
        neg_output = outdata[0, :][neg_idcs]
        neg_labels = isnod[0, :][neg_idcs]
        
       
        pos_size = pos_labels.data.cpu().numpy().shape[0]
        neg_size = neg_labels.data.cpu().numpy().shape[0] 
  
        #print('neg_output = {0}'.format(neg_output))
        if pos_size > 0 and neg_size > pos_size * 2:
            num_hard = pos_size * 2
            batch_size = 1
            neg_output, neg_labels = hard_mining(neg_output, neg_labels, num_hard * batch_size)
        if pos_size + neg_size > 10:
            num_hard = neg_size / 2
            batch_size = 1
            neg_output, neg_labels = hard_mining(neg_output, neg_labels, num_hard * batch_size)

        
        #print('pos_output = {0}'.format(pos_output))
        #print('neg_output = {0}'.format(neg_output))
        if pos_size > 0 and neg_size > 0:
            loss = 0.5 * binary_cross_entropy(
                pos_output, pos_labels) + 0.5 * binary_cross_entropy(
                neg_output, neg_labels)
        elif neg_size > 0:
            loss = 0.5 * binary_cross_entropy(
                neg_output, neg_labels)
        elif pos_size > 0:
            loss = 0.5 * binary_cross_entropy(
                pos_output, pos_labels)
        else:
            continue
            
        del neg_output, pos_output, neg_labels, pos_labels
        #print('loss =', loss)    
        #loss2 = binary_cross_entropy(casePred_each,isnod)
        #loss = loss2
        #missMask = (casePred_each<args.miss_thresh).float()
        #missLoss = -torch.sum(missMask*isnod*torch.log(casePred_each+0.001))/xsize[0]/xsize[1]
        #loss = loss2+args.miss_ratio*missLoss
        loss.backward()
        #torch.nn.utils.clip_grad_norm(model.parameters(), 1)
        #print('---------step2-------------')
        
        optimizer.step()
        loss2Hist.append(loss.data[0])
        #missHist.append(missLoss.data[0])
        lenHist.append(len_x)
        
        outdata = casePred_each.data.cpu().numpy()
        pred = outdata>0.5
        
        tpn += np.sum(1==pred[ydata==1])
        fpn += np.sum(1==pred[ydata==0])
        fnn += np.sum(0==pred[ydata==1])
        tnn += np.sum(0==pred[ydata==0])
        acc = np.mean(ydata==pred)
        accHist.append(acc)
        
    endtime = time.time()
    lenHist = np.array(lenHist)
    loss2Hist = np.array(loss2Hist)
    #lossHist = np.array(lossHist)
    accHist = np.array(accHist)
    
    mean_loss2 = np.sum(loss2Hist*lenHist)/np.sum(lenHist)
    #mean_missloss = np.sum(missHist*lenHist)/np.sum(lenHist)
    mean_acc = np.sum(accHist*lenHist)/np.sum(lenHist)
    print('Train, epoch %d, loss2 %.4f, acc %.4f, tpn %d, fpn %d, fnn %d, tnn %d, time %3.2f, lr %3.6f'
          %(epoch,mean_loss2,mean_acc,tpn,fpn, fnn, tnn, endtime-starttime, lr))
    
    save_data['tpn'].append(tpn)
    save_data['fpn'].append(fpn)
    save_data['fnn'].append(fnn)
    save_data['tnn'].append(tnn)
    save_data['loss'].append(mean_loss2)
    save_data['acc'].append(mean_acc)
    
    file = os.path.join('{0}'.format(save_name))
    fp = open(file, 'w')
    json.dump(save_data, fp, ensure_ascii=False)
    fp.close()    
    print
    
    
def val_casenet(epoch,model,data_loader,args, save_data, save_name, save_name_csv):
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
    results =[]
    #print('len(data_loader) = ', len(data_loader))
    #print('args.debug=',args.debug)
    for i,(x,coord,isnod,pbb,patient_id) in enumerate(data_loader):
        if args.debug:
            if i >4:
                break

        coord = Variable(coord,volatile=True).cuda()
        x = Variable(x,volatile=True).cuda()
        xsize = x.size()
        ydata = isnod.numpy()[:,0]
        #y = Variable(y).float().cuda()
        isnod1 = isnod
        isnod = Variable(isnod).float().cuda()

        nodulePred,casePred,casePred_each = model(x,coord)
        
        loss2 = binary_cross_entropy(casePred_each,isnod)
        #missMask = (casePred_each<args.miss_thresh).float()
        #missLoss = -torch.sum(missMask*isnod*torch.log(casePred_each+0.001))/xsize[0]/xsize[1]

        #loss2 = binary_cross_entropy(sigmoid(casePred),y[:,0])
        loss2Hist.append(loss2.data[0])
        #missHist.append(missLoss.data[0])
        lenHist.append(len(x))
        outdata = casePred_each.data.cpu().numpy()
        #print([i,data_loader.dataset.split[i,1],sigmoid(casePred).data.cpu().numpy()])
        
        
        pred = outdata>0.5        
        outdata1 = outdata.tolist()[0]
        pred1 = pred.tolist()[0]  
        #print('outdata=',outdata)
        #print('pred=',pred)
        #print('outdata1=',outdata1)
        #print('pred1=',pred1)
        #print('isnod1=',isnod)
        
        #print('pbb=', pbb)
        for j in range(len(outdata1)): 
            result = {"pred":0, "Lable":0, "pScore":0, "pScore1":0, "pID":0, "X":0, "Y":0, "Z":0, "D":0}
            result["pred"] = pred1[j]
            result["pScore"] = outdata1[j]
            result["pScore1"] = pbb[0][j][0]
            result["Lable"] = isnod1.numpy()[0][j].astype(int)
            
            #print('pbb[0][j][1] =', pbb[0][j][1])
            #print('pbb[0,j,1] =', pbb[0,j,1])
            result["Z"] = pbb[0][j][1]
            result["X"] = pbb[0][j][2]
            result["Y"] = pbb[0][j][3]
            result["D"] = pbb[0][j][4]
            result["pID"] = patient_id[0]
            results.append(result)            
            #print('result =', result)
                
        tpn += np.sum(1==pred[ydata==1])
        fpn += np.sum(1==pred[ydata==0])
        fnn += np.sum(0==pred[ydata==1])
        tnn += np.sum(0==pred[ydata==1])
        
        acc = np.mean(ydata==pred)
        accHist.append(acc)
    endtime = time.time()
    lenHist = np.array(lenHist)
    loss2Hist = np.array(loss2Hist)
    accHist = np.array(accHist)
    mean_loss2 = np.sum(loss2Hist*lenHist)/np.sum(lenHist)
    #mean_missloss = np.sum(missHist*lenHist)/np.sum(lenHist)
    mean_acc = np.sum(accHist*lenHist)/np.sum(lenHist)
    print('Valid, epoch %d, loss2 %.4f, acc %.4f, tpn %d, fpn %d, fnn %d,  time %3.2f'
          %(epoch, mean_loss2, mean_acc, tpn, fpn, fnn, endtime-starttime))
    
    save_data['tpn'].append(tpn)
    save_data['fpn'].append(fpn)
    save_data['fnn'].append(fnn)
    save_data['tnn'].append(tnn)
    save_data['loss'].append(mean_loss2)
    save_data['acc'].append(mean_acc)
    file = os.path.join('{0}'.format(save_name))
    fp = open(file, 'w')
    json.dump(save_data, fp, ensure_ascii=False)
    fp.close()
    
    
    file = os.path.join('{0}'.format(save_name_csv))
    if os.path.exists(file):
        os.remove(file)
    fp = open(file, 'w')
    json.dump(results, fp, ensure_ascii=False)
    fp.close()
    
    print

    
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
        if args.debug:
            if i >4:
                break

        coord = Variable(coord).cuda()
        x = Variable(x).cuda()
        nodulePred,casePred,_ = model(x,coord)
        predlist.append(casePred.data.cpu().numpy())
        #print([i,data_loader.dataset.split[i,1],casePred.data.cpu().numpy()])
    predlist = np.concatenate(predlist)
    return predlist    
