import argparse
import os
import time
import numpy as np
import data
from importlib import import_module
import shutil
from utils import *
import sys
sys.path.append('../')
from split_combine import SplitComb

import torch
from torch.nn import DataParallel
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch import optim
from torch.autograd import Variable
from config_training import config as config_training

from layers import acc
import json

parser = argparse.ArgumentParser(description='PyTorch DataBowl3 Detector')
parser.add_argument('--model', '-m', metavar='MODEL', default='base',
                    help='model')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--save-freq', default='10', type=int, metavar='S',
                    help='save frequency')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--save-dir', default='', type=str, metavar='SAVE',
                    help='directory to save checkpoint (default: none)')
parser.add_argument('--test', default=0, type=int, metavar='TEST',
                    help='1 do test evaluation, 0 not')
parser.add_argument('--split', default=8, type=int, metavar='SPLIT',
                    help='In the test phase, split the image to 8 parts')
parser.add_argument('--gpu', default='all', type=str, metavar='N',
                    help='use gpu')
parser.add_argument('--n_test', default=1, type=int, metavar='N',
                    help='number of gpu for test')
parser.add_argument('--subset', default=0, type=int, metavar='N',
                    help='the idx of the subset')
parser.add_argument('--debug', default=False, type=bool, metavar='N',
                    help='if debug or not')

def main():
    global args
    args = parser.parse_args()
    # args=Namespace(batch_size=32, epochs=100, gpu='all', lr=0.01, model='res18', momentum=0.9, n_test=8, resume='', save_dir='res18', save_freq=10, split=8, start_epoch=0, test=0, weight_decay=0.0001,workers=32)
    # print(args)
    
    torch.manual_seed(0)
    torch.cuda.set_device(0)

    model = import_module(args.model)
    config, net, loss, get_pbb = model.get_model()
    start_epoch = args.start_epoch
    save_dir = args.save_dir
    # results/res18
         
            
    #print('net=', net)
    ####### 1 training from checkpoint or fine-tuning
    if args.resume:  
        checkpoint = torch.load(args.resume)
        if start_epoch == 0:
            start_epoch = checkpoint['epoch'] + 1
        if not save_dir:
            save_dir = checkpoint['save_dir']
        else:
            save_dir = os.path.join('results',save_dir)
        net.load_state_dict(checkpoint['state_dict'])
    else:  # training from the beginning
        if start_epoch == 0:
            start_epoch = 1
        if not save_dir:
            exp_id = time.strftime('%Y%m%d-%H%M%S', time.localtime())
            save_dir = os.path.join('results', args.model + '-' + exp_id)
        else:
            save_dir = os.path.join('results',save_dir)
    
    ####### 2 set path
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)    
    logfile = os.path.join(save_dir,'log')
    if args.test!=1:
        sys.stdout = Logger(logfile)
        pyfiles = [f for f in os.listdir('./') if f.endswith('.py')]
        for f in pyfiles:
            shutil.copy(f,os.path.join(save_dir,f))
    n_gpu = setgpu(args.gpu)
    args.n_gpu = n_gpu
    net = net.cuda()
    loss = loss.cuda()
    cudnn.benchmark = True
    net = DataParallel(net)
    datadir = config_training['preprocess_result_path']
    
    '''
    ## check the weights and bias
    state_dict = net.module.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].cpu()
        if (key=='conv3dwuhui.0.weight'):
            print(key)
            res = state_dict[key]
            print('key =', res[0, 0:128, 0, 0, 0])'''
    
    print('--------------------step 1------------------')
    # for testing
    
    
   
    if args.test == 2:
        
        save_dir = '/home/wuhui/3dcnn/result-wh/nodnet/input96/{0}'.format(args.subset)
        '''
        print(save_dir)
        bbox_path = save_dir + '/bbox'
        if os.path.exists(bbox_path):
            shutil.rmtree(bbox_path)
            os.mkdir(bbox_path)'''
        
        margin = 32
        sidelen = 144
    
        luna_all_name = np.load('/home/wuhui/3dcnn/result-wh/inputdata/luna_val_{0}.npy'.format(args.subset))
        split_comber = SplitComb(sidelen,config['max_stride'],config['stride'],margin,config['pad_value'])
        dataset = data.DataBowl3Detector(
            datadir,
            '/home/wuhui/3dcnn/result-wh/inputdata/luna_val_{0}.npy'.format(args.subset),
            config,
            phase='test',
            split_comber=split_comber)
        test_loader = DataLoader(
            dataset,
            batch_size = 1,
            shuffle = False,
            num_workers = args.workers,
            collate_fn = data.collate,
            pin_memory=False)
        test(test_loader, net, get_pbb, save_dir,config)
        return
        
    
    save_dir = '/home/wuhui/3dcnn/result-wh/nodnet/input96/{0}'.format(args.subset)
    if not os.path.exists(save_dir):
           os.mkdir(save_dir)
    net_path = save_dir + '/net'
    json_path = save_dir + '/json'
    bbox_path = save_dir + '/bbox'
    if not os.path.exists(net_path):
        os.mkdir(net_path)
    if not os.path.exists(json_path):
        os.mkdir(json_path)
    if not os.path.exists(bbox_path):
        os.mkdir(bbox_path)
    
    
    #args.test = 0
    #if args.test == 1:
    '''margin = 32
    sidelen = 144
    
        #luna_val = np.load('../result-wh/inputdata/luna_val.npy')
        #luna_val = [idc.split('_')[0] for idc in luna_val]
        #np.save('../result-wh/inputdata/luna_val_name.npy', luna_val)
        
    split_comber = SplitComb(sidelen,config['max_stride'],config['stride'],margin,config['pad_value'])
    dataset = data.DataBowl3Detector(
            datadir,
            '../result-wh/inputdata/luna_val_name.npy',
            config,
            phase='test',
            split_comber=split_comber)
    test_loader = DataLoader(
            dataset,
            batch_size = 1,
            shuffle = False,
            num_workers = args.workers,
            collate_fn = data.collate,
            pin_memory=False)
        #test(test_loader, net, get_pbb, save_dir,config)
        #return'''

    #net = DataParallel(net)
    ######## 3 load training data
    print('--------------------step 2------------------')       
    dataset = data.DataBowl3Detector(
        datadir,
        '/home/wuhui/3dcnn/result-wh/inputdata/luna_train_{0}.npy'.format(args.subset),
        config,
        phase = 'train')
    train_loader = DataLoader(
        dataset,
        batch_size = args.batch_size,
        shuffle = True,
        num_workers = args.workers,
        pin_memory=True)

    ######## 4 load validate data
    dataset = data.DataBowl3Detector(
        datadir,
        '/home/wuhui/3dcnn/result-wh/inputdata/luna_val_{0}.npy'.format(args.subset),
        config,
        phase = 'val')
    val_loader = DataLoader(
        dataset,
        batch_size = args.batch_size,
        shuffle = False,
        num_workers = args.workers,
        pin_memory=True)

    ######## 5 set optimizer
    optimizer = torch.optim.SGD(
        net.parameters(),
        args.lr,
        momentum = 0.9,
        weight_decay = args.weight_decay)
    
    def get_lr(epoch):
        if epoch <= args.epochs * 0.5:
            lr = args.lr
        elif epoch <= args.epochs * 0.8:
            lr = 0.1 * args.lr
        else:
            lr = 0.01 * args.lr
        return lr
    
            
    train_data, val_data_val, val_data_all = save_model()         
    
    ######### 6 train and validate network
    for epoch in range(start_epoch, args.epochs + 1): 
        
        print('------------train------------')
        save_name = os.path.join('{0}/train_json_{1}.json'.format(json_path, epoch))
        train(train_loader, net, loss, epoch, optimizer, get_lr, args.save_freq, net_path, train_data, save_name, args)
        
        print('------------val------------')
        save_name = os.path.join('{0}/val_json_{1}.json'.format(json_path, epoch))
        validate(val_loader, net, loss, val_data_val, save_name, args)
        
        #print('------------test------------')
        #test(test_loader, net, get_pbb, save_dir,config)
        
def save_model():
    train_data = {
        'tpn':[],
        'fpn':[],
        'fnn':[],
        'tnn':[],
        'loss':[],
        'classify_loss':[],
        'regress_loss1':[],
        'regress_loss2':[],
        'regress_loss3':[],
        'regress_loss4':[]
        }


    val_data_val = {
        'tpn':[],
        'fpn':[],
        'fnn':[],
        'tnn':[],
        'loss':[],
        'classify_loss':[],
        'regress_loss1':[],
        'regress_loss2':[],
        'regress_loss3':[],
        'regress_loss4':[]
        }

    val_data_all = {
        'tpn':[],
        'fpn':[],
        'fnn':[],
        'tnn':[],
        'loss':[],
        'missloss':[],
        'acc':[],
        'recall':[],
        'precision':[]
        }
    return train_data, val_data_val, val_data_all
         
def train(data_loader, net, loss, epoch, optimizer, get_lr, save_freq, save_dir, train_data, save_name, args):
    start_time = time.time()
    
    net.train()
    lr = get_lr(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    
    #print(data_loader)
    metrics = []
    # data_loader: <torch.utils.data.dataloader.DataLoader object at 0x7f6633dfe8d0>
    for i, (data, target, coord, patient_id) in enumerate(data_loader):

        data = Variable(data.cuda(async = True))        
        target = Variable(target.cuda(async = True))        
        coord = Variable(coord.cuda(async = True))

        output = net(data, coord)      
        loss_output = loss(output, target)
        
        optimizer.zero_grad()
        loss_output[0].backward()
        optimizer.step()

        loss_output[0] = loss_output[0].data[0]
        metrics.append(loss_output)
        
        #print('end')
        
    end_time = time.time()   
    
    if epoch % args.save_freq == 0:            
        state_dict = net.module.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].cpu()
            
        torch.save({
            'epoch': epoch,
            'save_dir': save_dir,
            'state_dict': state_dict,
            'args': args},
            os.path.join(save_dir, '%03d.ckpt' % epoch))

    

        
    metrics = np.asarray(metrics, np.float32)
    print('Epoch %03d (lr %.5f)' % (epoch, lr))
    print('Train:      tpr %3.2f, tnr %3.2f, total pos %d, total neg %d, time %3.2f, lr %3.7f' % (
        100.0 * np.sum(metrics[:, 6]) / np.sum(metrics[:, 7]),
        100.0 * np.sum(metrics[:, 8]) / np.sum(metrics[:, 9]),
        np.sum(metrics[:, 7]),
        np.sum(metrics[:, 9]),
        end_time - start_time,
        lr))
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
    
         
        
        

def validate(data_loader, net, loss, train_data, save_name, args):
    start_time = time.time()
    
    net.eval()

    metrics = []
    for i, (data, target, coord, patient_id) in enumerate(data_loader):
            
        data = Variable(data.cuda(async = True), volatile = True)
        target = Variable(target.cuda(async = True), volatile = True)
        coord = Variable(coord.cuda(async = True), volatile = True)

        output = net(data, coord)
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
    print
    

def test(data_loader, net, get_pbb, save_dir, config):
    start_time = time.time()
    
    margin = 32
    sidelen = 144
    
    save_dir = os.path.join(save_dir,'bbox')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
   
    net.eval()
    namelist = []
    split_comber = data_loader.dataset.split_comber
    
    
    for i_name, (data, target, coord, nzhw) in enumerate(data_loader):
        print('i_name=', i_name)
        
        
        s = time.time()
        target = [np.asarray(t, np.float32) for t in target]
        lbb = target[0]
        nzhw = nzhw[0]       
        
        name = data_loader.dataset.filenames[i_name].split('/')[-1].split('_')[0]
        data = data[0][0]     
        coord = coord[0][0]   
       
        isfeat = False
        if 'output_feature' in config:
            if config['output_feature']:
                isfeat = True
        
        n_per_run = args.n_test       
        splitlist = range(0,len(data)+1,n_per_run)       
        
        if splitlist[-1]!=len(data):
            splitlist.append(len(data))
        outputlist = []
        featurelist = []

        isfeat = False
        for i in range(len(splitlist)-1):
            inputdata = Variable(data[splitlist[i]:splitlist[i+1]], volatile = True).cuda() 
            inputcoord = Variable(coord[splitlist[i]:splitlist[i+1]], volatile = True).cuda()
            
            if isfeat:
                output,feature = net(inputdata,inputcoord)
                featurelist.append(feature.data.cpu().numpy())
            else:
                output = net(inputdata,inputcoord)   
            outputlist.append(output.data.cpu().numpy())
            
        output = np.concatenate(outputlist,0)      
        output = split_comber.combine(output,nzhw=nzhw)
        
        if isfeat:
            feature = np.concatenate(featurelist,0).transpose([0,2,3,4,1])[:,:,:,:,:,np.newaxis]
            feature = split_comber.combine(feature,sidelen)[...,0]

        thresh = -3
        pbb,mask = get_pbb(output,thresh,ismask=True)
        
        if isfeat:
            feature_selected = feature[mask[0],mask[1],mask[2]]
            np.save(os.path.join(save_dir, name+'_feature.npy'), feature_selected)
        
        e = time.time()
        print('batch time is %3.2f seconds' % (e - s))
        print
    
        np.save(os.path.join(save_dir, name+'_pbb.npy'), pbb)
        np.save(os.path.join(save_dir, name+'_lbb.npy'), lbb)
    np.save(os.path.join(save_dir, 'namelist.npy'), namelist)
    end_time = time.time()

    print('elapsed time is %3.2f seconds' % (end_time - start_time))
    print
    

def singletest(data,net,config,splitfun,combinefun,n_per_run,margin = 64,isfeat=False):
    z, h, w = data.size(2), data.size(3), data.size(4)
    print(data.size())
    data = splitfun(data,config['max_stride'],margin)
    data = Variable(data.cuda(async = True), volatile = True,requires_grad=False)
    splitlist = range(0,args.split+1,n_per_run)
    outputlist = []
    featurelist = []
    for i in range(len(splitlist)-1):
        if isfeat:
            output,feature = net(data[splitlist[i]:splitlist[i+1]])
            featurelist.append(feature)
        else:
            output = net(data[splitlist[i]:splitlist[i+1]])
        output = output.data.cpu().numpy()
        outputlist.append(output)
        
    output = np.concatenate(outputlist,0)
    output = combinefun(output, z / config['stride'], h / config['stride'], w / config['stride'])
    if isfeat:
        feature = np.concatenate(featurelist,0).transpose([0,2,3,4,1])
        feature = combinefun(feature, z / config['stride'], h / config['stride'], w / config['stride'])
        return output,feature
    else:
        return output
    
if __name__ == '__main__':
    main()

