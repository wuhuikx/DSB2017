import torch
import numpy as np
import argparse
from importlib import import_module
parser = argparse.ArgumentParser(description='network surgery')
parser.add_argument('--model1', '-m1', metavar='MODEL', default='base',
                    help='model')
parser.add_argument('--model2', '-m2', metavar='MODEL', default='base',
                    help='model')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# args = parser.parse_args(['--model1','net_detector_3','--model2','net_classifier_3','--resume','../detector/results/res18-20170419-153425/020.ckpt'])
args = parser.parse_args()
# args:Namespace(model1='net_detector_3', model2='net_classifier_3', resume='../detector/results/res18/100.ckpt')

# ---------------set the detect model---------------#
nodmodel = import_module(args.model1)
config1, nod_net, loss, get_pbb = nodmodel.get_model()
checkpoint = torch.load(args.resume)
state_dict = checkpoint['state_dict']
nod_net.load_state_dict(state_dict)

#----------------set the classify model-------------#
casemodel = import_module(args.model2) #<module 'net_classifier_3' from '/home/wuhui/3dcnn/lung-detection-grt/training/classifier/net_classifier_3.pyc'>
config2 = casemodel.config
args.lr_stage2 = config2['lr_stage']
args.lr_preset2 = config2['lr']
topk = config2['topk']
case_net = casemodel.CaseNet(topk = topk,nodulenet=nod_net)
new_state_dict = case_net.state_dict()
torch.save({'state_dict': new_state_dict,'epoch':0},'result-wh/casenet/start.ckpt')   #'results/start.ckpt' 

# config2:
#{'conf_th': -1, 'bboxpath': '../detector/results/res18/bbox/', 'miss_thresh': 0.03, 'jitter_range': 0.15, 'detect_th': 0.05, 'nms_th': 0.05, 'lr_stage': array([ 50, 100, 140, 160]), 'crop_size': [96, 96, 96], 'filling_value': 160, 'startepoch': 20, 'radiusLim': [6, 100], 'datadir': '/home/data/data/result/prep/', 'lr': [0.01, 0.001, 0.0001, 1e-05], 'labelfile': './full_label.csv','random_sample': True, 'resample': None, 'preload_train': True, 'topk': 5, 'T': 1, 'scaleLim': [0.85, 1.15], 'padmask': False, 'miss_ratio': 1, 'preload_val': True, 'stride': 4, 'isScale': True, 'augtype': {'scale': False, 'rotate': False, 'flip': True, 'swap': False}}