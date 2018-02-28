import os
import time
import numpy as np
import json

import torch
from torch import nn
from torch.nn import DataParallel
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch import optim
from torch.autograd import Variable

from layers import acc



def save_json(file=None, j={}):
    fp = open(file, 'w')
    json.dump(j, fp, ensure_ascii=False)
    fp.close()

def get_lr(epoch,args):
    assert epoch<=args.lr_stage[-1]
    if args.lr==None:
        lrstage = np.sum(epoch>args.lr_stage)
        lr = args.lr_preset[lrstage]
    else:
        lr = args.lr
    return lr

def train_nodulenet(data_loader, net, loss, epoch, optimizer, args, train_data, save_name):
    start_time = time.time()
    net.train()
    if args.freeze_batchnorm:
        for m in net.modules():
            if isinstance(m, nn.BatchNorm3d):
                m.eval()

    lr = get_lr(epoch,args)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    #print('data_loader=',len(data_loader))
    metrics = []
    for i, (data, target, coord) in enumerate(data_loader):
        if args.debug:
            if i >4:
                break
        data = Variable(data.cuda(async = True))
        target = Variable(target.cuda(async = True))
        coord = Variable(coord.cuda(async = True))
        
        
        _,output = net(data, coord)
        loss_output = loss(output, target)
        
        #print('----------------detect-loss----------------')
        #print('data.size()=',data.size())
        #print('output.size()',output.size())
        #print('target.size()',target.size())
        #print('loss_output=',loss_output)
        #print('loss_output[0]=',loss_output[0])
        
                
        #print('target=',target[0,0,0,:,:])
        #print('output=',output[0,0,0,:,:])
        
        optimizer.zero_grad()
        loss_output[0].backward()
        #torch.nn.utils.clip_grad_norm(net.parameters(), 1)
        optimizer.step()

        loss_output[0] = loss_output[0].data[0]
        metrics.append(loss_output)
        #print('positive_num={0}\tnegative_num={1}\n---------------'.format(metrics[-1][7], metrics[-1][8]))
        #print('----------detect_pos=',metrics[-1][7])
        #print('----------detect_neg=',metrics[-1][8])

    end_time = time.time()

    metrics = np.asarray(metrics, np.float32)
    #print('metrics=',metrics)
    #print('----------------end-detect-loss----------------')
    
    #print('Epoch %03d (lr %.5f)' % (epoch, lr))
    print('Train:      tpr %3.2f, tnr %3.2f, total pos %d, total neg %d, time %3.2f, lr %3.6f' % (
        100.0 * np.sum(metrics[:, 6]) / np.sum(metrics[:, 7]),
        100.0 * np.sum(metrics[:, 8]) / np.sum(metrics[:, 9]),
        np.sum(metrics[:, 7]),
        np.sum(metrics[:, 9]),
        end_time - start_time, lr))
        
    print('loss %2.4f, classify loss %2.4f, regress loss %2.4f, %2.4f, %2.4f, %2.4f' % (
        np.mean(metrics[:, 0]),
        np.mean(metrics[:, 1]),
        np.mean(metrics[:, 2]),
        np.mean(metrics[:, 3]),
        np.mean(metrics[:, 4]),
        np.mean(metrics[:, 5])))

    train_data['tpn'].append(float(np.sum(metrics[:, 6])))
    train_data['tnn'].append(float(np.sum(metrics[:, 8])))
    train_data['fnn'].append(float(np.sum(metrics[:, 7]) - np.sum(metrics[:, 6])))
    train_data['fpn'].append(float(np.sum(metrics[:, 9]) - np.sum(metrics[:, 8])))
    train_data['loss'].append(float(np.mean(metrics[:, 0])))
    train_data['classify_loss'].append(float(np.mean(metrics[:, 1])))
    train_data['regress_loss1'].append(float(np.mean(metrics[:, 2])))
    train_data['regress_loss2'].append(float(np.mean(metrics[:, 3])))
    train_data['regress_loss3'].append(float(np.mean(metrics[:, 4])))
    train_data['regress_loss4'].append(float(np.mean(metrics[:, 5])))
    
    file = os.path.join('{0}'.format(save_name))
    fp = open(file, 'w')
    json.dump(train_data, fp, ensure_ascii=False)
    fp.close()
    print

    
def validate_nodulenet(data_loader, net, loss, train_data, save_name):
    start_time = time.time()
    net.eval()

    metrics = []
    for i, (data, target, coord) in enumerate(data_loader):
        data = Variable(data.cuda(async = True), volatile = True)
        target = Variable(target.cuda(async = True), volatile = True)
        coord = Variable(coord.cuda(async = True), volatile = True)

        _,output = net(data, coord)
        loss_output = loss(output, target, train = False)

        loss_output[0] = loss_output[0].data[0]
        metrics.append(loss_output)    
    end_time = time.time()

    metrics = np.asarray(metrics, np.float32)
    print('Validation: tpr %3.2f, tnr %3.8f, total pos %d, total neg %d, time %3.2f' % (
        100.0 * np.sum(metrics[:, 6]) / np.sum(metrics[:, 7]),
        100.0 * np.sum(metrics[:, 8]) / np.sum(metrics[:, 9]),
        np.sum(metrics[:, 7]),
        np.sum(metrics[:, 9]),
        end_time - start_time))
    print('loss %2.4f, classify loss %2.4f, regress loss %2.4f, %2.4f, %2.4f, %2.4f' % (
        np.mean(metrics[:, 0]),
        np.mean(metrics[:, 1]),
        np.mean(metrics[:, 2]),
        np.mean(metrics[:, 3]),
        np.mean(metrics[:, 4]),
        np.mean(metrics[:, 5])))
   
    
    train_data['tpn'].append(float(np.sum(metrics[:, 6])))
    train_data['tnn'].append(float(np.sum(metrics[:, 8])))
    train_data['fnn'].append(float(np.sum(metrics[:, 7]) - np.sum(metrics[:, 6])))
    train_data['fpn'].append(float(np.sum(metrics[:, 9]) - np.sum(metrics[:, 8])))
    train_data['loss'].append(float(np.mean(metrics[:, 0])))
    train_data['classify_loss'].append(float(np.mean(metrics[:, 1])))
    train_data['regress_loss1'].append(float(np.mean(metrics[:, 2])))
    train_data['regress_loss2'].append(float(np.mean(metrics[:, 3])))
    train_data['regress_loss3'].append(float(np.mean(metrics[:, 4])))
    train_data['regress_loss4'].append(float(np.mean(metrics[:, 5])))    
    
    file = os.path.join('{0}'.format(save_name))
    fp = open(file, 'w')
    json.dump(train_data, fp, ensure_ascii=False)
    fp.close()
    print

def test_nodulenet(data_loader, net, get_pbb, save_dir, config, n_per_run):
    start_time = time.time()
    save_dir = os.path.join(save_dir,'bbox')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    net.eval()
    namelist = []
    split_comber = data_loader.dataset.split_comber
    for i_name, (data, target, coord, nzhw) in enumerate(data_loader):
        s = time.time()
        target = [np.asarray(t, np.float32) for t in target]
        lbb = target[0]
        nzhw = nzhw[0]
        name = data_loader.dataset.filenames[i_name].split('-')[0].split('/')[-1]
        data = data[0][0]
        coord = coord[0][0]
        isfeat = False
        if 'output_feature' in config:
            if config['output_feature']:
                isfeat = True
        # print(data.size())
        splitlist = range(0,len(data)+1,n_per_run)
        if splitlist[-1]!=len(data):
            splitlist.append(len(data))
        outputlist = []
        featurelist = []

        for i in range(len(splitlist)-1):
            input = Variable(data[splitlist[i]:splitlist[i+1]], volatile = True).cuda()
            inputcoord = Variable(coord[splitlist[i]:splitlist[i+1]], volatile = True).cuda()
            _,output = net(input,inputcoord)
            outputlist.append(output.data.cpu().numpy())
        output = np.concatenate(outputlist,0)
        output = split_comber.combine(output,nzhw=nzhw)
        thresh = -3
        pbb,mask = get_pbb(output,thresh,ismask=True)
        #tp,fp,fn,_ = acc(pbb,lbb,0,0.1,0.1)
        #print([len(tp),len(fp),len(fn)])
        #print([i_name,name])
        e = time.time()
        np.save(os.path.join(save_dir, name+'_pbb.npy'), pbb)
        np.save(os.path.join(save_dir, name+'_lbb.npy'), lbb)
    np.save(os.path.join(save_dir, 'namelist.npy'), namelist)
    end_time = time.time()


    print('elapsed time is %3.2f seconds' % (end_time - start_time))
    print
    
