import numpy as np
import torch
from torch.utils.data import Dataset
import os
import time
import collections
import random
from layers import iou
import scipy.ndimage 
from scipy.ndimage import zoom
import warnings
from scipy.ndimage.interpolation import rotate
from layers import nms,iou
import pandas
import math
from pltbbox import save_img
#from pltbbox import plot_bbox
import shutil
import cv2
import matplotlib.pyplot as plt

def plot_bbox(img, pbb):
    
    #---------------origin-----------#   
    label=pbb
    if pbb.shape[0] == 4:
        label1 = np.zeros(5)
        label1[1:5] = label
        label = label1
        #label = np.numpy([0, label]) #label[np.newaxis, ...]
   
   
    img1=img[0,int(np.floor(label[1])),:,:]
    img1=img1.astype(np.int32)
    
    # plt.figure(title = 'img1')
    print('1 Original Image')
    plt.imshow(img1, cmap = 'gray')
    plt.show()
    
    #---------------box in origin-----------#
    x1=int(label[2] - label[4] * 0.5) 
    x2=int(label[2] + label[4] * 0.5)
    y1=int(label[3] - label[4] * 0.5)
    y2=int(label[3] + label[4] * 0.5)
    cv2.rectangle(img1, (y1, x1), (y2, x2),(255,255,255),1)
    
    # plt.figure(title = 'bbox')
    print('2 Bounding Box')
    plt.imshow(img1, cmap= 'gray')
    plt.show()

def plot_crop(crop):    
    crop_pic = crop[0, int(crop.shape[-1]/2), :, :]    
    plt.imshow(crop_pic, cmap = 'gray')
    plt.show()
    
    
def check_DataBowl3Classifier_init(candidate_box, pbb_label, filenames):
    for i in range(len(candidate_box)):
        clean = np.load(filenames[i])
        
        lbb_name = filenames[i]
        lbb_name = lbb_name.replace('clean','label')
        
        lbb = np.load(lbb_name)
        for j in range(lbb.shape[0]):
            print('Location of the {0}th ground truth nodule = {1}'.format(j, lbb[j]))            
            plot_bbox(clean, lbb[j])
            
        
        for j in range(len(candidate_box[i])):
            bbox = candidate_box[i][j]
            label = pbb_label[i][j]
            
            print('The {0}th detected nodule = {1}'.format(j, label))
            print('>>>> label = ', label)
            plot_bbox(clean, bbox)

def check_getitem(img, pbb, lbb, crop):
    print('lbb = ', lbb)
    plot_bbox(img, pbb)
    
    #print('crop = ', crop.shape)
    #print('int(crop.shape[-1]/2)=',int(crop.shape[-1]/2))
    crop_pic = crop[0, int(crop.shape[-1]/2), :, :]
    #print('crop_pic = ', crop_pic.shape)
    plt.imshow(crop_pic, cmap = 'gray')
    #plt.title('crop_pic')
    plt.show()
    
    

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
        
        
        #check_DataBowl3Classifier_init(self.candidate_box, self.pbb_label, self.filenames)

        
        
  ############################## get the topK candidates - by WuHui ######################
    def __getitem__(self, idx, split=None):
        
        t = time.time()
        np.random.seed(int(str(t%1)[2:7]))#seed according to time  
        pbb = self.candidate_box[idx]   # bbox after nms and score>-1
        pbb_label = self.pbb_label[idx] # label:1 IoU >0.05, 0 IoU < 0.05
        conf_list = pbb[:, 0]
        T = self.T
    
        self.topk=pbb.shape[0]
        topk=self.topk
        img = np.load(self.filenames1[idx])
        patient_id = self.filenames1[idx].split('/')[-1].split('_')[0]
        
        is_augment = False
        if self.random_sample and self.phase=='train':            
            chosenid = sample(conf_list,topk,T=T)
            #chosenid = conf_list.argsort()[::-1][:topk]
            is_augment = True
        else:
            chosenid = conf_list.argsort()[::-1][:topk] #argsort the array index from min to max
           

       
        croplist2 = []
        coordlist2 = []
        isnodlist2=[]        
        scores2 = []
        targetlist = []
        for i,id in enumerate(chosenid):
            target = pbb[id,1:]
            isnod = pbb_label[id]
            score = conf_list[id]
      
            D = get_D(target)
            self.config['crop_size'] = [D, D, D]
            crop_size= (self.config['crop_size'])
            crop_handle = simpleCrop(self.config, self.phase)
            crop, coord = crop_handle(img, target)
                              
            #check_getitem(img, pbb[id], pbb_label[id], crop)
          
            #print('is_augment =', is_augment)
            croplist, coordlist, isnodlist, scores = get_torch_data(crop, coord, isnod, score, self.stride, is_augment)
            croplist2 += croplist
            coordlist2 += coordlist
            isnodlist2 += isnodlist       
            scores2 += scores
            targetlist.append(target)
            
        
        if self.phase!='test':
            return croplist2, coordlist2, isnodlist2, scores2, patient_id, targetlist
        else:
            return torch.from_numpy(croplist).float(), torch.from_numpy(coordlist).float(), torch.from_numpy(isnodlist).int(), conf_list
        
    def __len__(self):
        if self.phase != 'test':
            return len(self.candidate_box)
        else:
            return len(self.candidate_box)
        
        
def get_torch_data(crop, coord, isnod, score, stride, is_augment):
    
    croplist2 = []
    coordlist2 = []
    isnodlist2 = []
    scores = []
    
    crop_size = crop.shape
    crop_size = crop_size[1:4]   
    croplist = np.zeros([1,
                         1,
                         crop_size[0],
                         crop_size[1],
                         crop_size[2]]).astype('float32')
    coordlist = np.zeros([1,
                          3,
                          crop_size[0]/stride,
                          crop_size[1]/stride,
                          crop_size[2]/stride]).astype('float32')
    isnodlist = np.zeros([1])
        
    if is_augment and isnod:
        
        #isnodlist[0]=isnod
        #coordlist[0] = coord
        #crops = augment(crop,coord,False,False,False,False)
        
        crops = [crop]
        crop,coord = augment(crop,coord,ifflip=True,ifrotate=True,ifswap = True)        
        crops.append(crop)
 
        for i in crops:
            i = i.astype(np.float32)
            i = i[np.newaxis, ...]
            croplist2.append(torch.from_numpy(i))
            isnodlist2.append(torch.from_numpy(isnodlist))
            coordlist2.append(torch.from_numpy(coordlist))
        scores += [score] * len(crops)
        

    else:
        
        crop = crop.astype(np.float32)
        croplist[0] = crop
        isnodlist[0]=isnod
        coordlist[0] = coord                        
  
        croplist2.append(torch.from_numpy(croplist).float())
        coordlist2.append(torch.from_numpy(coordlist).float())
        isnodlist2.append(torch.from_numpy(isnodlist).int())
        scores.append(score)

    return croplist2, coordlist2, isnodlist2, scores

        
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
            pbb_pos1 = np.ones((pbb_pos.shape[0], pbb_pos.shape[1] + 1))
            pbb_pos1[:,1:5] = pbb_pos.astype(np.float32)
            pbb_pos = pbb_pos1
        for p in pbb_pos:
            if p[-1] > 0:
                pbb_box.append(p)
                pbb_label.append(True)  
                #---------------#
                #plot_bbox(img, p)
 
    return pbb_box, pbb_label


def get_D(target):
    D = int(math.ceil(target[-1]))
    t1 = D / 16
    if D % 16 != 0:
        t1 +=1
        D = t1 * 16
    if D<16:
        D=16                
    
    D = 96
    return D
 
              
        
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
    
    #print('---------1---------')
    #plot_crop(sample)
    
    if ifswap:
        if sample.shape[1]==sample.shape[2] and sample.shape[1]==sample.shape[3]:
            axisorder = np.random.permutation(3)
            sample = np.transpose(sample,np.concatenate([[0],axisorder+1]))
            coord = np.transpose(coord,np.concatenate([[0],axisorder+1]))
    
    #print('---------2---------')
    #plot_crop(sample)
    
    if ifflip:
        flipid = np.array([np.random.randint(2),np.random.randint(2),np.random.randint(2)])*2-1
        sample = np.ascontiguousarray(sample[:,::flipid[0],::flipid[1],::flipid[2]])
        coord = np.ascontiguousarray(coord[:,::flipid[0],::flipid[1],::flipid[2]])
    
    #print('---------3---------')
    #plot_crop(sample)
    return sample, coord 

'''
def augment(sample, target, ifflip, ifswap, ifresize, ifshift):
    if ifrotate:
        validrot = False
        counter = 0
        while not validrot:
            newtarget = np.copy(target)
            angle1 = np.random.rand()*180
            size = np.array(sample.shape[2:4]).astype('float')
            rotmat = np.array([[np.cos(angle1/180*np.pi),-np.sin(angle1/180*np.pi)],[np.sin(angle1/180*np.pi),np.cos(angle1/180*np.pi)]])
            newtarget[1:3] = np.dot(rotmat,target[1:3]-size/2)+size/2
            if np.all(newtarget[:3]>target[3]) and np.all(newtarget[:3]< np.array(sample.shape[1:4])-newtarget[3]):
                validrot = True
                sample = rotate(sample,angle1,axes=(2,3),reshape=False)
                for box in bboxes:
                    box[1:3] = np.dot(rotmat,box[1:3]-size/2)+size/2
            else:
                counter += 1
                if counter ==3:
                    break
  
    #print('-----------sample=',sample.shape)
    
    samples=[sample,]
    #print('----------samples1=',len(samples))
    #print(samples[0].shape)
    if ifswap:
        #if sample.shape[1]==sample.shape[2] and sample.shape[1]==sample.shape[3]:
        #axisorder = np.random.permutation(3)
        #sample = np.transpose(sample,np.concatenate([[0],axisorder+1]))
        samples.append(sample.transpose((0,2,1,3)))
        samples.append(sample.transpose((0,2,3,1)))
        samples.append(sample.transpose((0,1,3,2)))
        samples.append(sample.transpose((0,3,1,2)))
        samples.append(sample.transpose((0,3,2,1)))                
    #print('----------samples2=',len(samples))
    #print(samples[1].shape)
    
    if ifflip:
       #         flipid = np.array([np.random.randint(2),np.random.randint(2),np.random.randint(2)])*2-1
        flipid = np.array([1,np.random.randint(2),np.random.randint(2)])*2-1
        #print((np.ascontiguousarray(sample[::flipid[0],::flipid[1],::flipid[2]])).shape)
        samples.append(np.ascontiguousarray(sample[::flipid[0],::flipid[1],::flipid[2]]))        
    #print('----------samples3=',len(samples))
    #print(samples[len(samples)-1].shape)
    
    if ifresize:
        size = [16, 32, 64,  96]
        for i in size:
            _,sample_shape,_,_ = sample.shape
            #print(scipy.ndimage.interpolation.zoom(sample,i * 1.0/ sample_shape,mode='nearest').shape)
            resize_sample=scipy.ndimage.interpolation.zoom(sample[0],
                                                            i * 1.0/ sample_shape,
                                                            mode='nearest')  
            #print('sub_sample=',resize_sample.shape)
            samples.append(resize_sample[np.newaxis])  
    #print('----------samples4=',len(samples))
    #print(samples[-2].shape)
    #print(samples[-1].shape)
    #for i in samples:
    #    print(i.shape)
        
    return samples
'''