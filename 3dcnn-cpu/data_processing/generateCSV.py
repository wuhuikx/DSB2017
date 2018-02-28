stprocessing for bbox and generate json
import numpy as np
import os
import warnings
import pandas as pd
import math
import random
import json

def main():
    
    conf_th = 0  
    nms_th =  0.05 
    iou_th = 0.05   

    subset = '0'
    #luna_train = np.load('/home/wuhui/3dcnn/result-wh/inputdata/luna_train_{0}.npy'.format(subset))
    luna_val = np.load('/home/wuhui/3dcnn/result-wh/inputdata/luna_val_{0}.npy'.format(subset)) # substitute this with your own path
    
    idcs = luna_val    
    idcs = [f.split('_')[0] for f in idcs] 

  
    bboxpath = './save/'
    save_name_csv = './save/'
    
    results = []    
    clean_num = 0
    pbb_all = 0
    pbb_confth = 0
    pbb_nmsth = 0
    pbb_pos = 0
    pbb_neg = 0
    for idx in idcs:
                
        pbb = np.load(os.path.join(bboxpath, idx + '_pbb.npy'))
        lbb = np.load(os.path.join(bboxpath, idx + '_lbb.npy'))      
        
        
        clean_num += 1
        pbb_all += len(pbb)    
        pbb = pbb[pbb[:, 0] > conf_th]
        pbb_confth += len(pbb)  
        
        
        pbb = nms(pbb, nms_th)   
        pbb_nmsth += len(pbb)
        

        pos_sample = []
        neg_sample = []    
        isnods = []
        for p in pbb:            
            isnod = False
            for l in lbb:
                score = iou(p[1:5], l)
                if score > iou_th:
                    isnod = True
                    pos_sample.append(p)
                    break
            if not isnod:
                neg_sample.append(p)
            isnods.append(isnods)  
            
        samples = pos_sample + neg_sample         
        pbb_pos += len(pos_sample)
        pbb_neg += len(neg_sample)
        
        
        for i in range(len(samples)):
            result = {"pred":0, "Lable":0, "pScore":0, "pID":0, "X":0, "Y":0, "Z":0, "D":0}
            result["pred"] = float(samples[i][0])
            result["pScore"] = float(samples[i][0])
            if isnods[i]:
                result["Lable"] = 1
                               
            result["Z"] = float(samples[i][1])
            result["X"] = float(samples[i][2])
            result["Y"] = float(samples[i][3])
            result["D"] = float(samples[i][4])
            result["pID"] = idx
 
            results.append(result)
            
    
    json_file = os.path.join('{0}/detect_val.json'.format(save_name_csv))
    if os.path.exists(json_file):
        os.remove(json_file)
    fp = open(json_file, 'w')
    json.dump(results, fp, ensure_ascii=False)
    fp.close()
        
    
        

    ## --------------convert .json to csv------------------##
    files = []
    files.append(json_file)
    dsts = []
    dsts.append('{0}/pred_val.csv'.format(save_name_csv))
   

    for i in range(len(files)):
        fp = open(files[i], 'r')
        samples = json.load(fp)
        fp.close()

        colums = ['seriesuid', 'coordX', 'coordY', 'coordZ', 'probability']
        seriesuid = []
        coordX = []
        coordY = []
        coordZ = []
        probability = []
        for sample in samples:
            seriesuid.append(sample['pID'])
            coordX.append(sample['X'])
            coordY.append(sample['Y'])
            coordZ.append(sample['Z'])
            probability.append(sample['pScore'])

        uid = pd.Series(seriesuid, name='seriesuid')  
        x = pd.Series(coordX, name='coordX')  
        y = pd.Series(coordY, name='coordY')  
        z = pd.Series(coordZ, name='coordZ')  
        p = pd.Series(probability, name='probability')

        predictions = pd.concat([uid,x,y,z,p], axis=1)  
        predictions.to_csv(dsts[i], index=False, sep=',')



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

    

if __name__ == '__main__':
    main()
    print('end')
