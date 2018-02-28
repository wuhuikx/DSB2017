import numpy as np
import torch
from torch.utils.data import Dataset
import os
import time
import collections
import random
from layers import iou
from scipy.ndimage import zoom
import warnings
from scipy.ndimage.interpolation import rotate
from layers import nms,iou
import pandas

def samples_boxlabel(img, pbb_neg, pbb_pos):
      
    pbb_box = []
    pbb_label = []  
    if pbb_neg.shape[-1] > 0:
        if len(pbb_neg.shape) == 1:
            pbb_neg = np.array([pbb_neg])
        for p in pbb_neg:
            isnod = False                   
            pbb_box.append(p)
            pbb_label.append(isnod)
            #---------------#
            #plot_bbox(img, p)


    if pbb_pos.shape[-1] > 0: 
        if len(pbb_pos.shape) == 1:
            pbb_pos = np.array([pbb_pos])
        if (pbb_pos.shape[-1]) == 4:
            pbb_pos1 = np.ones((pbb_pos.shape[0], pbb_pos.shape[1] + 1)) * 50.
            pbb_pos1[:,1:5] = pbb_pos.astype(np.float32)
            pbb_pos = pbb_pos1
        for p in pbb_pos:
            if p[-1] > 0:
                pbb_box.append(p)
                pbb_label.append(True)  
                #---------------#
                #plot_bbox(img, p)
 
    return pbb_box, pbb_label

class DataBowl3Classifier(Dataset):
    def __init__(self, split, config, phase = 'train'):
        assert(phase == 'train' or phase == 'val' or phase == 'test')
        
        self.random_sample = config['random_sample']
        self.T = config['T']
        self.topk = config['topk']
        self.crop_size = config['crop_size']
        self.stride = config['stride']
        self.augtype  = config['augtype']
        #self.labels = np.array(pandas.read_csv(config['labelfile']))
        
        #datadir = config['datadir']
        datadir = '../result-wh/prep_luna_full/'
        #bboxpath  = config['bboxpath']
        bboxpath = '../result-wh/bbox/'
        self.phase = phase
        self.candidate_box = []
        self.pbb_label = []
        self.config=config
        self.filenames1=[]
      
   
        idcs = split          
        self.filenames = [os.path.join(datadir, '%s_clean.npy' % idx) for idx in idcs]       
        idcs = [f.split('_')[0] for f in idcs]        
        for idx in idcs: 
           
            clean = np.load(os.path.join(datadir, '%s_clean.npy' % idx))          
            if phase == 'train':   
                
                pbb_neg = np.zeros(0) 
                if os.path.exists('../result-wh/training_samples/luna/iou_negtive/{0}.npy'.format(idx)):
                    pbb_neg = np.load('../result-wh/training_samples/luna/iou_negtive/{0}.npy'.format(idx))
                
                lbb = np.zeros(0)
                if os.path.exists(os.path.join(datadir,idx+'_label.npy')):
                    lbb = np.load(os.path.join(datadir,idx+'_label.npy'))
                                
                pbb_box, pbb_label = samples_boxlabel(clean, pbb_neg, lbb)
                
                
            if phase == 'val':
                pbb_neg = np.zeros(0)
                if os.path.exists('../result-wh/training_samples/luna/iou_negtive/{0}.npy'.format(idx)):
                    pbb_neg = np.load('../result-wh/training_samples/luna/iou_negtive/{0}.npy'.format(idx))
                    
                pbb_pos = np.zeros(0)
                if os.path.exists('../result-wh/training_samples/luna/iou_positive/{0}.npy'.format(idx)):
                    pbb_pos = np.load('../result-wh/training_samples/luna/iou_positive/{0}.npy'.format(idx))
        
                pbb_box, pbb_label = samples_boxlabel(clean, pbb_neg, pbb_pos)
                    
            if len(pbb_box) > 0:    
                self.candidate_box.append(np.array(pbb_box))
                self.pbb_label.append(np.array(pbb_label))
                self.filenames1.append(os.path.join(datadir, '%s_clean.npy' % idx))

        self.crop = simpleCrop(config,phase)
        

    def __getitem__(self, idx,split=None):
        t = time.time()
        np.random.seed(int(str(t%1)[2:7]))#seed according to time

        pbb = self.candidate_box[idx]
        pbb_label = self.pbb_label[idx]
        conf_list = pbb[:,0]
        T = self.T
        #print('-------pbb_label ={0}\tconf_list = {1}'.format(pbb_label, conf_list))
        
        self.topk=min(pbb.shape[0], 16)
        topk = self.topk
        #img = np.load(self.filenames[idx])
        img = np.load(self.filenames1[idx])
        patient_id = self.filenames1[idx].split('/')[-1].split('_')[0]
        
        if self.random_sample and self.phase=='train':
            chosenid = sample(conf_list,topk,T=T)
            #chosenid = conf_list.argsort()[::-1][:topk]
        else:
            chosenid = conf_list.argsort()[::-1][:topk]
            
        croplist = np.zeros([topk,1,self.crop_size[0],self.crop_size[1],self.crop_size[2]]).astype('float32')
        coordlist = np.zeros([topk,3,self.crop_size[0]/self.stride,self.crop_size[1]/self.stride,self.crop_size[2]/self.stride]).astype('float32')
        padmask = np.concatenate([np.ones(len(chosenid)),np.zeros(self.topk-len(chosenid))])
        isnodlist = np.zeros([topk])

        
        for i,id in enumerate(chosenid):
            target = pbb[id,1:]
            isnod = pbb_label[id]
            crop,coord = self.crop(img,target)
            if self.phase=='train':
                crop,coord = augment(crop,coord,
                                 ifflip=self.augtype['flip'],ifrotate=self.augtype['rotate'],
                                ifswap = self.augtype['swap'])
            crop = crop.astype(np.float32)
            croplist[i] = crop
            coordlist[i] = coord
            isnodlist[i] = isnod
            
        if self.phase!='test':
            #y = np.array([self.yset[idx]])
            return torch.from_numpy(croplist).float(), torch.from_numpy(coordlist).float(), torch.from_numpy(isnodlist).int(), pbb, patient_id
        else:
            return torch.from_numpy(croplist).float(), torch.from_numpy(coordlist).float(), torch.from_numpy(isnodlist).int()
    def __len__(self):
        if self.phase != 'test':
            return len(self.candidate_box)
        else:
            return len(self.candidate_box)
        

        
class simpleCrop():
    def __init__(self,config,phase):
        self.crop_size = config['crop_size']
        self.scaleLim = config['scaleLim']
        self.radiusLim = config['radiusLim']
        self.jitter_range = config['jitter_range']
        self.isScale = config['augtype']['scale'] and phase=='train'
        self.stride = config['stride']
        self.filling_value = config['filling_value']
        self.phase = phase
        
    def __call__(self,imgs,target):
        if self.isScale:
            radiusLim = self.radiusLim
            scaleLim = self.scaleLim
            scaleRange = [np.min([np.max([(radiusLim[0]/target[3]),scaleLim[0]]),1])
                         ,np.max([np.min([(radiusLim[1]/target[3]),scaleLim[1]]),1])]
            scale = np.random.rand()*(scaleRange[1]-scaleRange[0])+scaleRange[0]
            crop_size = (np.array(self.crop_size).astype('float')/scale).astype('int')
        else:
            crop_size = np.array(self.crop_size).astype('int')
        if self.phase=='train':
            jitter_range = target[3]*self.jitter_range
            jitter = (np.random.rand(3)-0.5)*jitter_range
        else:
            jitter = 0
        start = (target[:3]- crop_size/2 + jitter).astype('int')
        pad = [[0,0]]
        for i in range(3):
            if start[i]<0:
                leftpad = -start[i]
                start[i] = 0
            else:
                leftpad = 0
            if start[i]+crop_size[i]>imgs.shape[i+1]:
                rightpad = start[i]+crop_size[i]-imgs.shape[i+1]
            else:
                rightpad = 0
            pad.append([leftpad,rightpad])
        imgs = np.pad(imgs,pad,'constant',constant_values =self.filling_value)
        crop = imgs[:,start[0]:start[0]+crop_size[0],start[1]:start[1]+crop_size[1],start[2]:start[2]+crop_size[2]]
        
        normstart = np.array(start).astype('float32')/np.array(imgs.shape[1:])-0.5
        normsize = np.array(crop_size).astype('float32')/np.array(imgs.shape[1:])
        xx,yy,zz = np.meshgrid(np.linspace(normstart[0],normstart[0]+normsize[0],self.crop_size[0]/self.stride),
                           np.linspace(normstart[1],normstart[1]+normsize[1],self.crop_size[1]/self.stride),
                           np.linspace(normstart[2],normstart[2]+normsize[2],self.crop_size[2]/self.stride),indexing ='ij')
        coord = np.concatenate([xx[np.newaxis,...], yy[np.newaxis,...],zz[np.newaxis,:]],0).astype('float32')

        if self.isScale:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                crop = zoom(crop,[1,scale,scale,scale],order=1)
            newpad = self.crop_size[0]-crop.shape[1:][0]
            if newpad<0:
                crop = crop[:,:-newpad,:-newpad,:-newpad]
            elif newpad>0:
                pad2 = [[0,0],[0,newpad],[0,newpad],[0,newpad]]
                crop = np.pad(crop,pad2,'constant',constant_values =self.filling_value)

        return crop,coord

def sample(conf,N,T=1):
    if len(conf)>N:
        target = range(len(conf))
        chosen_list = []
        for i in range(N):
            chosenidx = sampleone(target,conf,T)
            chosen_list.append(target[chosenidx])
            target.pop(chosenidx)
            conf = np.delete(conf, chosenidx)

            
        return chosen_list
    else:
        return np.arange(len(conf))

def sampleone(target,conf,T):
    assert len(conf)>1
    p = softmax(conf/T)
    p = np.max([np.ones_like(p)*0.00001,p],axis=0)
    p = p/np.sum(p)
    return np.random.choice(np.arange(len(target)),size=1,replace = False, p=p)[0]

def softmax(x):
    maxx = np.max(x)
    return np.exp(x-maxx)/np.sum(np.exp(x-maxx))


def augment(sample, coord, ifflip = True, ifrotate=True, ifswap = True):
    #                     angle1 = np.random.rand()*180
    if ifrotate:
        validrot = False
        counter = 0
        angle1 = np.random.rand()*180
        size = np.array(sample.shape[2:4]).astype('float')
        rotmat = np.array([[np.cos(angle1/180*np.pi),-np.sin(angle1/180*np.pi)],[np.sin(angle1/180*np.pi),np.cos(angle1/180*np.pi)]])
        sample = rotate(sample,angle1,axes=(2,3),reshape=False)
        
    if ifswap:
        if sample.shape[1]==sample.shape[2] and sample.shape[1]==sample.shape[3]:
            axisorder = np.random.permutation(3)
            sample = np.transpose(sample,np.concatenate([[0],axisorder+1]))
            coord = np.transpose(coord,np.concatenate([[0],axisorder+1]))
            
    if ifflip:
        flipid = np.array([np.random.randint(2),np.random.randint(2),np.random.randint(2)])*2-1
        sample = np.ascontiguousarray(sample[:,::flipid[0],::flipid[1],::flipid[2]])
        coord = np.ascontiguousarray(coord[:,::flipid[0],::flipid[1],::flipid[2]])
    return sample, coord 
