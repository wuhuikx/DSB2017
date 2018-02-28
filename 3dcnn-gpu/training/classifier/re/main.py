import argparse
import os
import time
import numpy as np
from importlib import import_module
import shutil
import sys
from split_combine import SplitComb

import torch
from torch.nn import DataParallel
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch import optim
from torch.autograd import Variable

from layers import acc
from trainval_detector import *
from trainval_classifier_roi import *
from data_detector import DataBowl3Detector
from data_classifier_wuhui1 import DataBowl3Classifier

from utils import *

parser = argparse.ArgumentParser(description='PyTorch DataBowl3 Detector')
parser.add_argument('--model1', '-m1', metavar='MODEL', default='base',
                    help='model')
parser.add_argument('--model2', '-m2', metavar='MODEL', default='base',
                    help='model')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=None, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('-b2', '--batch-size2', default=3, type=int,
                    metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('--lr', '--learning-rate', default=None, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--save-freq', default='5', type=int, metavar='S',
                    help='save frequency')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--save-dir', default='', type=str, metavar='SAVE',
                    help='directory to save checkpoint (default: none)')
parser.add_argument('--test1', default=0, type=int, metavar='TEST',
                    help='do detection test')
parser.add_argument('--test2', default=0, type=int, metavar='TEST',
                    help='do classifier test')
parser.add_argument('--test3', default=0, type=int, metavar='TEST',
                    help='do classifier test')
parser.add_argument('--split', default=8, type=int, metavar='SPLIT',
                    help='In the test phase, split the image to 8 parts')
parser.add_argument('--gpu', default='all', type=str, metavar='N',
                    help='use gpu')
parser.add_argument('--n_test', default=8, type=int, metavar='N',
                    help='number of gpu for test')
parser.add_argument('--debug', default=0, type=int, metavar='TEST',
                    help='debug mode')
parser.add_argument('--freeze_batchnorm', default=0, type=int, metavar='TEST',
                    help='freeze the batchnorm when training')
 

def main():
    
    global args
    args = parser.parse_args()
    
    print('-------------begin program---------------')
    ##################### 1 load 'nodule-detection-model' #################
    nodmodel = import_module(args.model1)
    config1, nod_net, loss, get_pbb = nodmodel.get_model()
    args.lr_stage = config1['lr_stage']  # lr_stage=[ 50 100 140 160]
    args.lr_preset = config1['lr'] #lr_preset=[0.01, 0.001, 0.0001, 1e-05]
    save_dir = args.save_dir
    # print('node_net:', id(nod_net))
    
    ##################### 2 load 'cancer-prediction-model' ###############
    casemodel = import_module(args.model2)   
    config2 = casemodel.config
    args.lr_stage2 = config2['lr_stage']
    args.lr_preset2 = config2['lr']
    topk = config2['topk']
    case_net = casemodel.CaseNet(topk = topk,nodulenet=nod_net)

    
    args.miss_ratio = config2['miss_ratio']
    args.miss_thresh = config2['miss_thresh']
    if args.debug:
        args.save_dir = 'debug'    
    
    
   ################ 3 initialize 'case_net' based on nod_net #############
    start_epoch = args.start_epoch
    if args.resume:
        print('-----------step 1---------')
        checkpoint = torch.load(args.resume)
        if start_epoch == 0:
            start_epoch = checkpoint['epoch'] + 1
        if not save_dir:
            save_dir = checkpoint['save_dir']
        else:
            save_dir = os.path.join('results',save_dir)
        case_net.load_state_dict(checkpoint['state_dict'])
    else:
        print('-----------step 2---------')
        if start_epoch == 0:
            start_epoch = 1
        if not save_dir:
            exp_id = time.strftime('%Y%m%d-%H%M%S', time.localtime())
            save_dir = os.path.join('results', args.model1 + '-' + exp_id)
        else:
            save_dir = os.path.join('results',save_dir)
    if args.epochs == None:
        end_epoch = args.lr_stage2[-1]
    else:
        end_epoch = args.epochs
        
   
    ############## set path ##############
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    logfile = os.path.join(save_dir,'log')
    if args.test1!=1 and args.test2!=1 :
        sys.stdout = Logger(logfile)
        pyfiles = [f for f in os.listdir('./') if f.endswith('.py')]
        for f in pyfiles:
            shutil.copy(f,os.path.join(save_dir,f))
            
    
    #####################################
    torch.cuda.set_device(0)
    #nod_net = nod_net.cuda()
    case_net = case_net.cuda()
    loss = loss.cuda()
    cudnn.benchmark = True
    if not args.debug:
        case_net = DataParallel(case_net)
        nod_net = DataParallel(nod_net)
 
    
    ########### for testing ##############
    if args.test1 == 1:
        testsplit = np.load('full.npy')
        dataset = DataBowl3Classifier(testsplit, config2, phase = 'test')
        predlist = test_casenet(case_net,dataset).T
        anstable = np.concatenate([[testsplit],predlist],0).T
        df = pandas.DataFrame(anstable)
        df.columns={'id','cancer'}
        df.to_csv('allstage1.csv',index=False)
        return
    
    if args.test2 ==1:
        testsplit = np.load('test.npy')
        dataset = DataBowl3Classifier(testsplit, config2, phase = 'test')
        predlist = test_casenet(case_net,dataset).T
        anstable = np.concatenate([[testsplit],predlist],0).T
        df = pandas.DataFrame(anstable)
        df.columns={'id','cancer'}
        df.to_csv('quick',index=False)
        return
    
    if args.test3 == 1:
        testsplit3 = np.load('stage2.npy')
        dataset = DataBowl3Classifier(testsplit3,config2,phase = 'test')
        predlist = test_casenet(case_net,dataset).T
        anstable = np.concatenate([[testsplit3],predlist],0).T
        df = pandas.DataFrame(anstable)
        df.columns={'id','cancer'}
        df.to_csv('stage2_ans.csv',index=False)
        return
   
   
    ############ 4 load data for the 'nodule-detection-net' ##################    
    trainsplit = np.load('../result-wh/inputdata/luna_train_name.npy')
    valsplit = np.load('../result-wh/inputdata/luna_val_name.npy')
    testsplit = np.load('../result-wh/inputdata/luna_val_name.npy')    
    dataset = DataBowl3Detector(trainsplit,config1,phase = 'train')
    train_loader_nod = DataLoader(dataset,batch_size = args.batch_size,
        shuffle = True,num_workers = args.workers,pin_memory=True)

    dataset = DataBowl3Detector(valsplit,config1,phase = 'val')
    val_loader_nod = DataLoader(dataset,batch_size = args.batch_size,
        shuffle = False,num_workers = args.workers,pin_memory=True)

    optimizer = torch.optim.SGD(nod_net.parameters(),
        args.lr,momentum = 0.9,weight_decay = args.weight_decay)
    
    
    ########### 5 load data for the 'cancer-prediction-model' ################

    trainsplit = np.load('../result-wh/inputdata/luna_train.npy') # all patient name information  kaggleluna_full.npy
    valsplit = np.load('../result-wh/inputdata/luna_val.npy')   
    #print('trainsplit = ', trainsplit.shape)
    dataset = DataBowl3Classifier(trainsplit,config2,phase = 'train')
    
    #print('dataset =', len(dataset))
    train_loader_case = DataLoader(dataset,batch_size = args.batch_size2,shuffle = True,num_workers = args.workers,pin_memory=True)  
    
    #print('train_loader_case =', len(train_loader_case))
    print('----------end 1----------')
    
    dataset = DataBowl3Classifier(valsplit,config2,phase = 'val')
    #print('dataset =', len(dataset))
    val_loader_case = DataLoader(dataset,batch_size = 1,
        shuffle = False,num_workers = args.workers,pin_memory=True)    
    #print('val_loader_case =', len(val_loader_case))
    print('----------end 2----------')
    
    dataset = DataBowl3Classifier(trainsplit,config2,phase = 'val')
    #print('dataset =', len(dataset))
    all_loader_case = DataLoader(dataset,batch_size = 1,
        shuffle = False,num_workers = args.workers,pin_memory=True)   
    #print('val_loader_case =', len(all_loader_case))
    print('----------end 3----------')

    #print('case_net.parameters=',case_net.parameters)
    optimizer2 = torch.optim.SGD(case_net.parameters(),
        args.lr,momentum = 0.9,weight_decay = args.weight_decay)

    '''optimizer2 = torch.optim.SGD([{'params': case_net.module.NoduleNet.parameters(), 'lr': 0.0001},
                                  {'params': case_net.module.fc1.parameters(), 'lr': 0.0001},
                                  {'params': case_net.module.fc2.parameters(), 'lr': 0.0001}],
                                 0.0,
                                 momentum = 0.9,
                                 weight_decay = args.weight_decay)'''
    print('----------end 4----------')

    ############################# 6 training ##################################
    # the nod_net in the case_net and the detection noed_net is the same object
    #-----------------------use flip data for training-----------------------#
    # (1) first train case_net (the included nod_net is already trained)
    # (2) then train the nod_net (fine-tuning)
    # (3) train the case_net
    # (4) train node_net once then train case_net once, step by step
    #----------use flip/resize/swap/rotation data for fine-tuning------------#
    # (1) use the trained case_net (including nod_net + prediction)
    # (2) the same training process as above
    #args.save_dir = '../result-wh/casenet/2017_09_04'
    print('save_dir = ', args.save_dir)
    model_path = args.save_dir + '/model'
    net_path = args.save_dir + '/net3'
    log_path = args.save_dir + '/log'   
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)  
    print('model_path = ', model_path)
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    if not os.path.exists(net_path):
        os.mkdir(net_path)
    if not os.path.exists(log_path):
        os.mkdir(log_path)



    case_train_data, case_val_data_val, case_val_data_all, nod_train_data, nod_val_data = save_model()
    
    for epoch in range(start_epoch, end_epoch + 1):
        '''for i in [60, 80, 100, 160]:   
            if epoch == i:
                for param_group in optimizer2.param_groups:
                    if param_group['lr'] !=0:
                        param_group['lr'] /= 10'''
                    
        if epoch ==start_epoch:
            print('--------1 train_casenet-------')
            #print('args1=',args)
            #lr = args.lr
            debug = args.debug
            args.debug = True
            
            print('----------end 5----------')
            #print('1_fc2=',case_net.module.state_dict()['fc2.weight']) 
            save_name = './{0}/casenet_train_0.json'.format(model_path)
            train_casenet(epoch,case_net,train_loader_nod,optimizer2,args,case_train_data,save_name)
            
            #print('----------end 6----------')
            #print('2_fc2=',case_net.module.state_dict()['fc2.weight']) 
            #train_casenet_patient(epoch,case_net,train_loader_case,optimizer2,args)
            #args.lr = lr
            #args.debug = debug

        if epoch<args.lr_stage[-1]:
            print('--------2 train_nodulenet+validate_nodulenet-------')
            save_name = './{0}/nodnet_train_{1}.json'.format(model_path,epoch) 
            train_nodulenet(train_loader_nod, nod_net, loss, epoch, optimizer, args, nod_train_data, save_name)
            save_name = './{0}/nodnet_val_{1}.json'.format(model_path,epoch)
            validate_nodulenet(val_loader_nod, nod_net, loss, nod_val_data, save_name)   
            
        if epoch>config2['startepoch']:
            print('--------3 train_casenet+val_casenet-------')
            save_name = './{0}/casenet_train_{1}.json'.format(model_path,epoch)
            train_casenet(epoch,case_net,train_loader_case,optimizer2,args, case_train_data, save_name)
            
            save_name = './{0}/casenet_val_val_{1}.json'.format(model_path,epoch)   
            save_name_csv = './{0}/casenet_val_csv_val_{1}.json'.format(model_path,epoch)  
            val_casenet(epoch,case_net,val_loader_case,args, case_val_data_val, save_name, save_name_csv)
            
            save_name = './{0}/casenet_val_all_{1}.json'.format(model_path,epoch)
            save_name_csv = './{0}/casenet_val_csv_all_{1}.json'.format(model_path,epoch)
            val_casenet(epoch,case_net,all_loader_case, args, case_val_data_all, save_name, save_name_csv)
           
        
        #save model 
        args.save_freq = 1
        if epoch % args.save_freq == 0:            
            state_dict = case_net.module.state_dict()
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].cpu()

            torch.save({
                'epoch': epoch,
                'save_dir': save_dir,
                'state_dict': state_dict,
                'args': args},
                os.path.join(net_path, '%03d.ckpt' % epoch))
            

            
def save_model():
    train_data = {
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

    val_data_val = {
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
    
    nod_train_data = {
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
    
    nod_val_data = {
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
    return train_data, val_data_val, val_data_all, nod_train_data, nod_val_data
'''            
def load_net(args):

   
    ##################### 1 load 'nodule-detection-model' #################
    nodmodel = import_module(args.model1)
    config1, nod_net, loss, get_pbb = nodmodel.get_model()
    args.lr_stage = config1['lr_stage']  # lr_stage=[ 50 100 140 160]
    args.lr_preset = config1['lr'] #lr_preset=[0.01, 0.001, 0.0001, 1e-05]
    save_dir = args.save_dir
    # print('node_net:', id(nod_net))
    
    ##################### 2 load 'cancer-prediction-model' ###############
    casemodel = import_module(args.model2)   
    config2 = casemodel.config
    args.lr_stage2 = config2['lr_stage']
    args.lr_preset2 = config2['lr']
    #print('args.lr_stage2=',args.lr_stage2)
    #print('args.lr_preset2=',args.lr_preset2)
    topk = config2['topk']
    case_net = casemodel.CaseNet(topk = topk,nodulenet=nod_net)
    #case_net = casemodel.CaseNet(nodulenet=nod_net)
    
    args.miss_ratio = config2['miss_ratio']
    args.miss_thresh = config2['miss_thresh']
    if args.debug:
        args.save_dir = 'debug'    
    
    
   ################ 3 initialize 'case_net' based on nod_net #############
    start_epoch = args.start_epoch
    if args.resume:
        print('-----------step 1---------')
        checkpoint = torch.load(args.resume)
        if start_epoch == 0:
            start_epoch = checkpoint['epoch'] + 1
        if not save_dir:
            save_dir = checkpoint['save_dir']
        else:
            save_dir = os.path.join('results',save_dir)
        case_net.load_state_dict(checkpoint['state_dict'])
    else:
        print('-----------step 2---------')
        if start_epoch == 0:
            start_epoch = 1
        if not save_dir:
            exp_id = time.strftime('%Y%m%d-%H%M%S', time.localtime())
            save_dir = os.path.join('results', args.model1 + '-' + exp_id)
        else:
            save_dir = os.path.join('results',save_dir)
    if args.epochs == None:
        end_epoch = args.lr_stage2[-1]
    else:
        end_epoch = args.epochs
        
     #####################################
    torch.cuda.set_device(0)
    #nod_net = nod_net.cuda()
    case_net = case_net.cuda()
    loss = loss.cuda()
    cudnn.benchmark = True
    if not args.debug:
        case_net = DataParallel(case_net)
        nod_net = DataParallel(nod_net)
 
        

def set_path():
    ############## set path ##############
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    logfile = os.path.join(save_dir,'log')
    if args.test1!=1 and args.test2!=1 :
        sys.stdout = Logger(logfile)
        pyfiles = [f for f in os.listdir('./') if f.endswith('.py')]
        for f in pyfiles:
            shutil.copy(f,os.path.join(save_dir,f))
            
            
def test_func():
    ########### for testing ##############
    if args.test1 == 1:
        testsplit = np.load('full.npy')
        dataset = DataBowl3Classifier(testsplit, config2, phase = 'test')
        predlist = test_casenet(case_net,dataset).T
        anstable = np.concatenate([[testsplit],predlist],0).T
        df = pandas.DataFrame(anstable)
        df.columns={'id','cancer'}
        df.to_csv('allstage1.csv',index=False)
        return
    
    if args.test2 ==1:
        testsplit = np.load('test.npy')
        dataset = DataBowl3Classifier(testsplit, config2, phase = 'test')
        predlist = test_casenet(case_net,dataset).T
        anstable = np.concatenate([[testsplit],predlist],0).T
        df = pandas.DataFrame(anstable)
        df.columns={'id','cancer'}
        df.to_csv('quick',index=False)
        return
    
    if args.test3 == 1:
        testsplit3 = np.load('stage2.npy')
        dataset = DataBowl3Classifier(testsplit3,config2,phase = 'test')
        predlist = test_casenet(case_net,dataset).T
        anstable = np.concatenate([[testsplit3],predlist],0).T
        df = pandas.DataFrame(anstable)
        df.columns={'id','cancer'}
        df.to_csv('stage2_ans.csv',index=False)
        return
    
    
def load_nodnet_data(trainsplit, valsplit, config1): 
    
    ############ 4 load data for the 'nodule-detection-net' ##################       
    dataset = DataBowl3Detector(trainsplit,config1,phase = 'train')
    train_loader_nod = DataLoader(dataset,batch_size = args.batch_size,
        shuffle = True,num_workers = args.workers,pin_memory=True)

    dataset = DataBowl3Detector(valsplit,config1,phase = 'val')
    val_loader_nod = DataLoader(dataset,batch_size = args.batch_size,
        shuffle = False,num_workers = args.workers,pin_memory=True)

    return train_loader_nod, val_loader_nod
    

def load_case_data(trainsplit, valsplit, config2):
    ########### 5 load data for the 'cancer-prediction-model' ################
    dataset = DataBowl3Classifier(trainsplit,config2,phase = 'train') 
    train_loader_case = DataLoader(dataset,batch_size = args.batch_size2,shuffle = False,num_workers = args.workers,pin_memory=True)  
   
    dataset = DataBowl3Classifier(valsplit,config2,phase = 'val')
    val_loader_case = DataLoader(dataset,batch_size = max([args.batch_size2,1]),
        shuffle = False,num_workers = args.workers,pin_memory=True)
    
    dataset = DataBowl3Classifier(trainsplit,config2,phase = 'val')
    all_loader_case = DataLoader(dataset,batch_size = max([args.batch_size2,1]),
        shuffle = False,num_workers = args.workers,pin_memory=True)   
    
    return train_loader_case, val_loader_case, all_loader_case
          
        
def save_model(model, net_name, epoch, save_dir):                        
            state_dict = model.module.state_dict()
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].cpu()

            torch.save({
                'epoch': epoch,
                'save_dir': save_dir,
                'state_dict': state_dict,
                'args': args},
                os.path.join(save_dir, '%03d.ckpt' % epoch))
    
'''
                       
if __name__ == '__main__':
    main()

