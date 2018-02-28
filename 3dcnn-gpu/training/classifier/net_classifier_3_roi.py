import torch
from torch import nn
from layers import *
from torch.nn import DataParallel
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch import optim
from torch.autograd import Variable
from torch.utils.data import Dataset
from scipy.ndimage.interpolation import rotate
import numpy as np
import os
import sys
sys.path.append('../')
from config_training import config as config_training

config = {}
config['topk'] = 5
config['resample'] = None
config['datadir'] = config_training['preprocess_result_path']
config['preload_train'] = True
config['bboxpath'] = config_training['bbox_path']
config['labelfile'] = './full_label.csv'
config['preload_val'] = True

config['padmask'] = False

config['crop_size'] = [96,96,96]
config['scaleLim'] = [0.85,1.15]
config['radiusLim'] = [2,100]
config['jitter_range'] = 0.15
config['isScale'] = True

config['random_sample'] = True
config['T'] = 1
#config['topk'] = 5
config['stride'] = 4
config['augtype'] = {'flip':True,'swap':False,'rotate':False,'scale':False}

config['detect_th'] = 0.05   #0.05
config['conf_th'] = -1
config['nms_th'] = 0.05
config['filling_value'] = 160

config['startepoch'] = 20
config['lr_stage'] = np.array([50,100,140,160])
config['lr'] = [0.001,0.0001,0.00001,0.000001]
config['miss_ratio'] = 1
config['miss_thresh'] = 0.03


def nms_idx2(output, nms_th):
    print('output =', output.shape)
    output1 = np.copy(output)
    if output.shape[0] == 0:
        return output

    idx = np.argsort(-output[:, 0])
    output = output[idx]
    bboxes = [output[0]]
    idxes = [idx[0]]
    #print('output=',output[5])
    #print('output1=', output1[idx[5]])
    
    for i in np.arange(1, len(output)):
        bbox = output[i]
        flag = 1
        for j in range(len(bboxes)):
            if iou(bbox[1:5], bboxes[j][1:5]) >= nms_th:
                flag = -1
                break
        if flag == 1:
            bboxes.append(bbox)
            idxes.append(idx[i])
    
    print('len(bboxes)=',len(bboxes))
    print('bboxes =', bboxes[2])
    print('output =', output1[idxes[2]])
    bboxes = np.asarray(bboxes, np.float32)
    return bboxes

def nms_idx(output, nms_th1, nms_th2):
    #print('output =', output.shape)
    output1 = np.copy(output)
    if output.shape[0] == 0:
        return output

    idx = np.argsort(-output[:, 0])
    output = output[idx]
    bboxes = [output[0]]
    idxes = [idx[0]]
    #print('output=',output[5])
    #print('output1=', output1[idx[5]])
    
    for i in np.arange(1, len(output)):
        bbox = output[i]
        flag = 1
        for j in range(len(bboxes)):
            if iou(bbox[1:5], bboxes[j][1:5]) >= nms_th:
                flag = -1
                break
        if flag == 1:
            bboxes.append(bbox)
            idxes.append(idx[i])
    
    #print('len(bboxes)=',len(bboxes))
    #print('bboxes =', bboxes[2])
    #print('output =', output1[idxes[2]])
    bboxes = np.asarray(bboxes, np.float32)
    idxes = np.asarray(idxes, np.int)
    return bboxes, idxes


class GetPBB(object):
    def __init__(self):
        self.stride = 4
        self.anchors = np.asarray([10.0, 30.0, 60.0])

    def __call__(self, output, thresh = -3, ismask=False):
        output1 = np.copy(output)
            
        stride = self.stride
        anchors = self.anchors
        offset = (float(stride) - 1) / 2
        output_size = output1.shape
        oz = np.arange(offset, offset + stride * (output_size[0] - 1) + 1, stride)
        oh = np.arange(offset, offset + stride * (output_size[1] - 1) + 1, stride)
        ow = np.arange(offset, offset + stride * (output_size[2] - 1) + 1, stride)
        
        
        output1[:, :, :, :, 1] = oz.reshape((-1, 1, 1, 1)) + output1[:, :, :, :, 1] * anchors.reshape((1, 1, 1, -1))
        output1[:, :, :, :, 2] = oh.reshape((1, -1, 1, 1)) + output1[:, :, :, :, 2] * anchors.reshape((1, 1, 1, -1))
        output1[:, :, :, :, 3] = ow.reshape((1, 1, -1, 1)) + output1[:, :, :, :, 3] * anchors.reshape((1, 1, 1, -1))
        output1[:, :, :, :, 4] = np.exp(output1[:, :, :, :, 4]) * anchors.reshape((1, 1, 1, -1))
        mask = output1[..., 0] > thresh
        xx,yy,zz,aa = np.where(mask)
        output1 = output1[xx,yy,zz,aa]
        output = output[xx,yy,zz,aa]

        if ismask:
            return output1,output, [xx,yy,zz,aa]
        else:
            return output1

        #output = output[output[:, 0] >= self.conf_th] 
        #bboxes = nms(output, self.nms_th)
        
class CaseNet(nn.Module):
    def __init__(self,topk,nodulenet):
        super(CaseNet,self).__init__()
        #print('------------Initializing CaseNet-------------')
    
        self.NoduleNet  = nodulenet
        self.fc1 = nn.Linear(128,64)
        self.fc2 = nn.Linear(64,1)
        self.pool = nn.MaxPool3d(kernel_size=2)
        self.dropout = nn.Dropout(0.5)
        self.baseline = nn.Parameter(torch.Tensor([-30.0]).float())
        self.Relu = nn.ReLU()
    
    def __conf(self, pbbs=None, th=-1):
        #print('conf in: {}'.format(pbbs.shape))

        pbbs = pbbs[pbbs[:, 0]> th]
        
        return pbbs
    
    def __nms(self, pbbs=None, th=0.05):
        #print('nms in: {}'.format(pbbs.shape))
        pbbs = nms(pbbs, th)
        
        return pbbs
    
    def __get_topk(self, pbbs=None, k=300):
        #print('topk in: {}'.format(pbbs.shape))
        pbbs = pbbs[np.lexsort(-pbbs[:, : :-1].T)]
        pbbs = pbbs[:k]
        
        return pbbs
    
    def __get_proposals(self, pbbs, th = -1, target=None, topk=300):
        #pbbs = target.view(-1,5)
        #print('pbb_size1=', pbbs.size())
        #pbbs = pbbs.data.cpu().numpy()
        #print('pbbs=', pbbs[0:20,:])
        
        pbbs = self.__conf(pbbs,th)
        print('pbb_size2=', pbbs.shape)
        pbbs = self.__nms(pbbs)
        print('pbb_size3=', pbbs.shape)
        pbbs = self.__get_topk(pbbs, topk)
        print('pbb_size4=', pbbs.shape)

        return pbbs
    def __get_rois(self, pbbs):
        #print('rois in: {}'.format(pbbs.shape))
        pbbs = pbbs[:, 1:]
        return pbbs
    
    def __rm_neg_coord(self, pbbs):
        pbbs = [pbb for pbb in pbbs if (pbb>0).all()]
        return pbbs
        #print('case_net:', id(self.NoduleNet))
    def forward(self,xlist,coordlist,target):
#         xlist: n x k x 1x 96 x 96 x 96
#         coordlist: n x k x 3 x 24 x 24 x 24

        #print('----------step1----------')
        xsize = xlist.size()
        corrdsize = coordlist.size()
        xlist = xlist.view(-1,xsize[1],xsize[2],xsize[3],xsize[4])
        coordlist = coordlist.view(-1,corrdsize[1],corrdsize[2],corrdsize[3],corrdsize[4])
        #print('----------step2----------')
        
        print('xlist = {0} \t coordlist = {1}'.format(xlist.size(), coordlist.size()))
        noduleFeat,nodulePred = self.NoduleNet(xlist,coordlist)
        nodulePred1 = np.concatenate(nodulePred.data.cpu().numpy().tolist(),0) 
        
        print('noduleFeat = {0} \t nodulePred = {1}'.format(noduleFeat.size(), nodulePred1.shape))
        get_pbb = GetPBB()
        pbbs,nodulePred1,mask = get_pbb(nodulePred1,-3,True)
        
        
        print('pbbs =', pbbs.shape)
        #target = target[mask]
        print('---------nms---------')
        pbb,idxes = nms_idx(pbbs, 0.05)
        nodulePred1=nodulePred1[idxes]
        
        print('---------end---------')
        
        
        
        
        
        
        #print(nodulePred[0,0,0,0,:,:])
        '''nodulePred1 = nodulePred.view(-1,5)
        pbbs = self.__conf(nodulePred1.data.cpu().numpy(),0)
        print('pbb_size2=', pbbs.shape)
        print(pbbs)
        pbbs = self.__nms(pbbs)
        print('pbb_size3=', pbbs.shape)
        pbbs = self.__get_topk(pbbs, 5)
        print('pbb_size4=', pbbs.shape)'''
        
        
        print('xsize=',xsize)
        pbbs = self.__get_proposals(nodulePred.data.cpu().numpy(), -1, None, 5)  #nodulePred
        print('pbb_size1=', pbbs.shape)
        
        rois = self.__get_rois(pbbs)
        rois = self.__rm_neg_coord(rois)
        print('roi in pbbs: {}'.format(len(rois)))
        
        '''nodulePred1 = nodulePred.data.cpu().numpy() 
        nodulePred1 = nodulePred1[nodulePred1[:,:,:,:,:,0]>0]
        target1=target[nodulePred1[:,:,:,:,:,0]>0]
        print('nodulePred1=',nodulePred1)'''
        
        nodulePred1 = rois
        
        out1 = []
        
        for roi in rois:
            print(roi)
            x, y,z, d = (roi / 4).astype(int)
            if d % 2 != 0:
                d += 1
            
            r = d / 2
            print((x -r,x + r), (y - r , y + r), (z - r , z + r))
            this_feat = noduleFeat[:,:,x -r: x + r,
                                       y - r : y + r,
                                       z - r : z + r]
            
            this_pool = nn.MaxPool3d(kernel_size=d)
            centerFeat = this_pool(this_feat)
            
            centerFeat = centerFeat[:,:,0,0,0]
            out = self.dropout(centerFeat)
            out = self.Relu(self.fc1(out))
            out = self.fc2(out)
            out1.append(out)
        
        '''for i in nodulePred1:
            i = (i/4).astype(int)
            roi_pool = nn.MaxPool3d(kernel_size=i[4])
            centerFeat = roi_pool(noduleFeat[:,:,
                                               i[1] / 2 - i[4] /2 -1 : i[1] / 2 + i[4] /2 + 1,
                                               i[2] / 2 - i[4] /2 -1 : i[2] / 2 + i[4] /2 + 1,
                                               i[3] / 2 - i[4] /2 -1 : i[3] / 2 + i[4] /2 + 1])
        
            centerFeat = centerFeat[:,:,0,0,0]  #1x128
            #print('----------step5----------')
        
            out = self.dropout(centerFeat)
            out = self.Relu(self.fc1(out))
            out = self.fc2(out)
            out1.append(out)
            #out = out.view(xsize[0],xsize[1])'''
        out = out1[0]
        for i in out1:
            out = torch.cat((out,i),1)
            
   
        #print('----------step6----------')
        return out, target1

def nms(output, nms_th):
    if len(output) == 0:
        return output

    output = output[np.argsort(-output[:, 0])]
    bboxes = [output[0]]
    
    for i in np.arange(1, len(output)):
        bbox = output[i]
        flag = 1
        for j in range(len(bboxes)):
            if iou(bbox[1:5], bboxes[j][1:5]) >= nms_th:
                flag = -1
                break
        if flag == 1:
            bboxes.append(bbox)
    
    bboxes = np.asarray(bboxes, np.float32)
    return bboxes

def iou(box0, box1):
    
    r0 = box0[3] / 2
    s0 = box0[:3] - r0
    e0 = box0[:3] + r0

    r1 = box1[3] / 2
    s1 = box1[:3] - r1
    e1 = box1[:3] + r1

    overlap = []
    for i in range(len(s0)):
        overlap.append(max(0, min(e0[i], e1[i]) - max(s0[i], s1[i])))

    intersection = overlap[0] * overlap[1] * overlap[2]
    union = box0[3] * box0[3] * box0[3] + box1[3] * box1[3] * box1[3] - intersection
    return intersection / union