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
from pltbbox import plot_bbox
import shutil


class DataBowl3Classifier(Dataset):
    def __init__(self, split, config, phase = 'train'):
        
        assert(phase == 'train' or phase == 'val' or phase == 'test')
        #print('----------------DataBowl3Classifier-init----------------')
        self.random_sample = config['random_sample']
        self.T = config['T']
        self.topk = config['topk']
        self.crop_size = config['crop_size']
        self.stride = config['stride']
        self.augtype  = config['augtype']
        #self.labels = np.array(pandas.read_csv(config['labelfile']))
        
        datadir = config['datadir']
        bboxpath  = config['bboxpath']
        self.phase = phase
        self.candidate_box = []
        self.pbb_label = []
        self.config=config  # by WuHui
        self.names=[]
        self.filenames1=[]
        ################## load the preprocessing result ##################
        idcs = split  
        # filenames /home/data/data/result/prep/
        self.filenames = [os.path.join(datadir, '%s_clean.npy' % idx) for idx in idcs]
        #self.kagglenames = [f for f in self.filenames if len(f.split('/')[-1].split('_')[0])>20]
        #self.lunanames = [f for f in self.filenames if len(f.split('/')[-1].split('_')[0])<20]
        
        # ./full_label.csv
        #labels = np.array(pandas.read_csv(config['labelfile']))        
        #if phase !='test':
        #    self.yset = np.array([labels[labels[:,0]==f.split('-')[0].split('_')[0],1] for f in split]).astype('int')
        idcs = [f.split('-')[0] for f in idcs]
        
        
        count_total_sample = 0
        count_confth_sample = 0
        count_nms_sample = 0
        count_pos_sample = 0
        count_neg_sample = 0
        #print('bbox_path=',bboxpath)
        
        count_lbb_pos = 0       
        self.save_file = False
        self.detect_sample = False
        ############  load the bbox result and set the label based on IoU ###########
        for idx in idcs: 
            #if not idx == '75aef267ad112a21c870b7e2893aaf8a':
            #    continue
           
            clean = np.load(os.path.join(datadir, '%s_clean.npy' % idx))
            if self.save_file:
                file_dir = str(idx)
                if os.path.exists(file_dir):
                    shutil.rmtree(file_dir)    
                    os.mkdir(file_dir)
            
            ################ save total ###################
            pbb = np.load(os.path.join(bboxpath,idx+'_pbb.npy'))
            count_total_sample += len(pbb)
            
            #print('total_samples=',len(pbb))
            if self.save_file:
                file_dir = os.path.join(str(idx), 'total')             
                os.mkdir(file_dir)
                save_img(pbb, clean, file_dir)
            
            
            ################## save conf_th ##################
            pbb = pbb[pbb[:,0]>config['conf_th']] # nodule score > conf_th, conf_th = -1
            count_confth_sample += len(pbb)
            #print('conf_th={0}\tconfig_path samples={1}'.format(config['conf_th'], len(pbb)))

            if self.save_file:               
                file_dir = os.path.join(str(idx), 'total1')
                os.mkdir(file_dir)
                save_img(pbb, clean, file_dir)
            
            ################### save nms_th #####################
            pbb = nms(pbb, config['nms_th'])  #config['nms_th'] = 0.05
            count_nms_sample += len(pbb)
            #print('nms={0}\tnms_path={1}'.format(config['nms_th'], len(pbb)))
            if self.save_file:                
                file_dir = os.path.join(str(idx), 'nms')
                os.mkdir(file_dir)
                save_img(pbb, clean, file_dir) 
            
            lbb = np.load(os.path.join(bboxpath,idx+'_lbb.npy'))
            for i in lbb:
                if i[-1] > 0:
                    count_lbb_pos += 1
            
        
            pbb_label = []            
            pbb_box = []          
            ################## pos and neg ###############
            if self.detect_sample or phase == 'val':
                #scores=np.zeros(len(pbb))
                #i=0
                for p in pbb:
                    isnod = False
                    for l in lbb:
                        score = iou(p[1:5], l)
                        #if score -scores[i-1] > 0:
                        #    scores[i-1] = score 
                        if score > config['detect_th']:  #config['detect_th'] = 0.05
                            isnod = True
                            break
                    pbb_label.append(isnod)
                    pbb_box = pbb
                
                if self.save_file:
                    np.save(os.path.join(str(idx), 'nms/iou_score.npy'),scores)
                    print('label:', pbb_label)
                    print('pos={0}\tneg={1}\tdetect_th={2}'.format(pos, len(pbb)- pos, config['detect_th']))                
            else:
          
                for p in pbb:
                    isnod = False
                    for l in lbb:
                        score = iou(p[1:5], l)
                        if score > config['detect_th']:
                            isnod = True
                            break
                    if not isnod:
                        #print('p=',p)
                        pbb_box.append(p)
                        pbb_label.append(isnod)

                for p in lbb:
                    if p[-1] > 0:
                        box = np.ones(5)
                        box[1:5] = p
                        pbb_box.append(box)
                        pbb_label.append(True) 
              
            if len(pbb_box) > 0:    
                self.candidate_box.append(np.array(pbb_box))
                self.pbb_label.append(np.array(pbb_label))
                self.names.append(idx)
                self.filenames1.append(os.path.join(datadir, '%s_clean.npy' % idx))

            
            '''    
            if not os.path.exists('./val_set'):
                os.mkdir('./val_set')
            if not os.path.exists('./val_set/{0}'.format(idx)):
                os.mkdir('./val_set/{0}'.format(idx))
            np.save(os.path.join('./val_set/{0}/1clean.npy'.format(idx)),clean)
            np.save(os.path.join('./val_set/{0}/1pbb_box.npy'.format(idx)),pbb_box)
            np.save(os.path.join('./val_set/{0}/1pbb_label.npy'.format(idx)),pbb_label)
            np.save(os.path.join('./val_set/{0}/1gt_label.npy'.format(idx)),lbb)
            #print('pbb_box1=',pbb_box)
            #print('pbb_label1=',pbb_label)
            '''
            
            for i in pbb_label:
                if i:
                    count_pos_sample += 1
                else:
                    count_neg_sample += 1
        '''if phase == 'train':
            print('total_samples=',count_total_sample)
            print('conf_th={0}\tconfig_path samples={1}'.format(config['conf_th'], count_confth_sample))
            print('nms={0}\tnms_path={1}'.format(config['nms_th'], count_nms_sample))
            print('pos samples={0}\tneg samples={1}'.format(count_pos_sample,count_neg_sample))'''

         
            
        self.crop = simpleCrop(config,phase)

        
        
  ############################## get the topK candidates - by WuHui ######################
    def __getitem__(self, idx,split=None):
        
        t = time.time()
        np.random.seed(int(str(t%1)[2:7]))#seed according to time  
        pbb = self.candidate_box[idx]   # bbox after nms and score>-1
        pbb_label = self.pbb_label[idx] # label:1 IoU >0.05, 0 IoU < 0.05
        conf_list = pbb[:, 0]
        T = self.T
        #topk = self.topk
        self.topk=pbb.shape[0]
        topk=self.topk
        img = np.load(self.filenames1[idx])
        if self.random_sample and self.phase=='train':            
            chosenid = sample(conf_list,topk,T=T)
            #chosenid = conf_list.argsort()[::-1][:topk]
        else:
            chosenid = conf_list.argsort()[::-1][:topk] #argsort the array index from min to max

        #print('pbb_label=',pbb_label)  
        #print('pbb=',pbb)
        patient_id=self.names[idx]
        '''if os.path.exists(os.path.join('./samples/{0}'.format(patient_id))):
            shutil.rmtree(os.path.join('./samples/{0}'.format(patient_id)))
        #os.mkdir('samples')
        os.mkdir(os.path.join('./samples/{0}'.format(patient_id)))
        np.save(os.path.join('./samples/{0}/img.npy'.format(patient_id)),img)
        np.save(os.path.join('./samples/{0}/pbb.npy'.format(patient_id)),pbb)
        np.save(os.path.join('./samples/{0}/lbb.npy'.format(patient_id)),pbb_label)'''
        
        '''
        np.save(os.path.join('./val_set/{0}/2img.npy'.format(patient_id)),img)
        np.save(os.path.join('./val_set/{0}/2pbb.npy'.format(patient_id)),pbb)
        np.save(os.path.join('./val_set/{0}/2lbb.npy'.format(patient_id)),pbb_label)
        '''
        
        croplist2 = []
        coordlist2 = []
        isnodlist2=[]        
        scores = []
        for i,id in enumerate(chosenid):
            target = pbb[id,1:]
            isnod = pbb_label[id]
            score = conf_list[id]
            
            
            D = int(math.ceil(target[-1]))
            t1 = D / 16
            if D % 16 != 0:
                t1 +=1
            D = t1 * 16
            if D<16:
                D=16
                
            D = 96
            
            # D = 28(n)
            # D = 16(y)
            # D = 24(n)
            # D = 56(n)
            

            self.config['crop_size'] = [D, D, D]
            crop_size= (self.config['crop_size'])
            crop_handle = simpleCrop(self.config, self.phase)
            crop, coord = crop_handle(img, target)

            
            croplist = np.zeros([1,
                             1,
                             crop_size[0],
                             crop_size[1],
                             crop_size[2]]).astype('float32')
            coordlist = np.zeros([1,
                              3,
                              self.config['crop_size'][0]/self.stride,
                              self.config['crop_size'][1]/self.stride,
                              self.config['crop_size'][2]/self.stride]).astype('float32')
            isnodlist = np.zeros([1])
            
            
            if self.phase=='train':
                crops = []
                '''crop,coord = augment(crop,coord,
                                     ifflip=self.augtype['flip'],
                                     ifrotate=self.augtype['rotate'],
                                     ifswap = self.augtype['swap'])'''
                if isnod:
                    crops = augment(crop,coord,False,False,False,False)
                     # scores.extend(score * len(crops))
                    #augment(sample, target, ifflip = True, ifswap = True, ifresize=True, ifshift=True):
                    #crops.append(crop)
                    #print('crops=',crops)
                    #clean = np.load(os.path.join(datadir, '%s_clean.npy' % idx))
                    #for i in crops:    
                        #plot_bbox(crops[i], clean)
                
                isnodlist[0]=isnod
                coordlist[0] = coord
                if isnod:
                    for i in crops:
                        i = i.astype(np.float32)
                        i = i[np.newaxis, ...]
                        croplist2.append(torch.from_numpy(i).float())
                        isnodlist2.append(torch.from_numpy(isnodlist).int())
                        coordlist2.append(torch.from_numpy(coordlist).float())
                    scores += [score] * len(crops)
                else:
                    crop = crop.astype(np.float32)
                    croplist[0] = crop
                    croplist2.append(torch.from_numpy(croplist).float())
                    coordlist2.append(torch.from_numpy(coordlist).float())
                    isnodlist2.append(torch.from_numpy(isnodlist).int())
                    scores.append(score)
            else:
                isnodlist[0]=isnod
                coordlist[0] = coord
                
                crop = crop.astype(np.float32)
                croplist[0] = crop
                croplist2.append(torch.from_numpy(croplist).float())
                coordlist2.append(torch.from_numpy(coordlist).float())
                isnodlist2.append(torch.from_numpy(isnodlist).int())
                scores.append(score)
                
                #print('isnodlist=',isnodlist)    
        if self.phase!='test':
            return croplist2, coordlist2, isnodlist2, scores, patient_id
        else:
            return torch.from_numpy(croplist).float(), torch.from_numpy(coordlist).float(), torch.from_numpy(isnodlist).int(), conf_list
        
    def __len__(self):
        if self.phase != 'test':
            return len(self.candidate_box)
        else:
            return len(self.candidate_box)
############################# end by WuHui ##############################        
        

    
        
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

'''
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
'''
def augment(sample, target, ifflip, ifswap, ifresize, ifshift):
    '''if ifrotate:
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
    '''
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
