import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import mlab

path = '/home/wuhui/3dcnn/result-wh/nodnet/input128/2017_09_14/0/json/'
#casenet_train = path + 'casenet_train_loss_40.json'
#casenet_val = path + 'casenet_val_loss_val_40.json'
#casenet_all = path + 'casenet_val_loss_all_40.json'

casenet_train = [path + 'train_json_100.json']
#casenet_train.append(path + 'train_json_90.json')

casenet_val = [path + 'val_json_100.json']
#casenet_val.append(path + 'val_json_90.json')

#nodnet_train = 'nodnet_train.json'
#nodnet_val = 'nodnet_val.json'

color_sequence = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c',
                  '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5',
                  '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f',
                  '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5']

def load_data(json_file):
    data = {
        'tpn':[],
        'fpn':[],
        'fnn':[],
        'tnn':[],
        'loss':[],
        'classify_loss':[],
        'regress_loss1':[],
        'regress_loss2':[],
        'regress_loss3':[],
        'regress_loss4':[],
        }
    
    for i in json_file:
        fp = open(i, 'r')
        data1 = json.load(fp)
        data['tpn'] += data1['tpn']
        data['fpn'] += data1['fpn']
        data['fnn'] += data1['fnn']
        data['tnn'] += data1['tnn']
        data['loss'] += data1['loss']
        data['classify_loss'] += data1['classify_loss']
        data['regress_loss1'] += data1['regress_loss1']
        data['regress_loss2'] += data1['regress_loss2']
        data['regress_loss3'] += data1['regress_loss3']
        data['regress_loss4'] += data1['regress_loss4']
        fp.close()

    return data
    
def draw_roc(fpr, tpr):
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    fpr=np.array(fpr, dtype=np.float64)
    tpr=np.array(tpr, dtype=np.float64)
    
    #label_x=range(1,len(fpr)+1)
    #label_y=range(1,len(tpr)+1)
    
    plt.plot(fpr, tpr, 'r', label='false positive rate')
    #plt.plot(label_y, tpr, 'b', label='true positive rate')
    #plt.xticks(label_x, label_y, rotation=0)
    plt.grid()
    plt.show()
      

#def get_fpr_tpr(fp, tp)

def draw_casenet_loss(json_file, val=False):
    
    title = 'train dataset'
    if val:
        title = 'validation dataset'
        
    data = load_data(json_file)
    
    loss = data['loss']
    classify_loss = data['classify_loss']
    regress_loss1 = data['regress_loss1']
    regress_loss2 = data['regress_loss2']
    regress_loss3 = data['regress_loss3']
    regress_loss4 = data['regress_loss4']
    #missloss = data['missloss']
    
    epoch = len(loss)
    labels = range(1, epoch + 1)   
    fig,ax = plt.subplots(1,4,figsize=(24,5))   
    
    
    line1, = ax[0].plot(labels, loss, '-', linewidth=2, label='loss', color = color_sequence[0]) 
    #ax[0].text(labels[-1] + 2, loss[-1], 'loss', fontsize = 10, color = color_sequence[0])
    
    line2, = ax[0].plot(labels, classify_loss, '-', linewidth=2, label='classify_loss', color = color_sequence[2])
    #ax[0].text(labels[-1] + 2, classify_loss[-1], 'classify_loss', fontsize = 10, color = color_sequence[2])
    
    line3, = ax[0].plot(labels, regress_loss1, '--', linewidth=2, label='regress_loss1', color = color_sequence[4]) 
    #ax[0].text(labels[-1] + 2, regress_loss1[-1], 'regress_loss1', fontsize = 10, color = color_sequence[4])
    
    line4, = ax[0].plot(labels, regress_loss2, '-', linewidth=2, label='regress_loss2', color = color_sequence[6])
    #ax[0].text(labels[-1] + 2, regress_loss2[-1], 'regress_loss2', fontsize = 10, color = color_sequence[6])
    
    line5, = ax[0].plot(labels, regress_loss3, '--', linewidth=2, label='regress_loss3', color = color_sequence[8])  
    #ax[0].text(labels[-1] + 2, regress_loss3[-1], 'regress_loss3', fontsize = 10, color = color_sequence[8])
    
    line6, = ax[0].plot(labels, regress_loss4, '-', linewidth=2, label='regress_loss4', color = color_sequence[10]) 
    #ax[0].text(labels[-1] + 2, regress_loss4[-1], 'regress_loss4', fontsize = 10, color = color_sequence[10])
    
    ax[0].grid(True)
    ax[0].legend(loc='upper right')
    ax[0].set_title(title)
    ax[0].set_xlabel('epoch')
    ax[0].set_ylabel('loss')
    
    
#     recall = data['recall']
#     precision = data['precision']
#     line3, = ax[1].plot(labels,recall, '--', linewidth=2, label='recall')
#     line4, = ax[1].plot(labels, precision, '-', linewidth=2, label='precision')   
#     ax[1].grid(True)
#     ax[1].legend(loc='best')  #bbox_to_anchor=[1.5, 1]
#     ax[1].set_title(title)
#     ax[1].set_xlabel('epoch')
#     ax[1].set_ylabel('train loss')
    
    
    tp = data['tpn']
    tn = data['tnn']
    fp = data['fpn']
    fn = data['fnn'] 
    
    tpr =  np.array(tp)[:] / (np.array(tp)[:] + np.array(fn)[:])
    fpr =  np.array(fp)[:] / (np.array(tp)[:] + np.array(fp)[:])
    tnr =  np.array(tn)[:] / (np.array(tn)[:] + np.array(fn)[:])
    
    line7, = ax[1].plot(labels,tp, '--', linewidth=2, label='tp', color = color_sequence[0])
    #ax[1].text(labels[-1] + 1, tp[-1], 'tp', fontsize = 12, color = color_sequence[0])
    line8, = ax[1].plot(labels,fp, '--', linewidth=2, label='fp', color = color_sequence[2])
    #ax[1].text(labels[-1] + 1, fp[-1], 'fp', fontsize = 12, color = color_sequence[2])
    line9, = ax[1].plot(labels, fn, '-', linewidth=2, label='fn', color = color_sequence[4])  
    #ax[1].text(labels[-1] + 1, fn[-1], 'fn', fontsize = 12, color = color_sequence[4])
    #line10, = ax[1].plot(labels, tnr, '-', linewidth=2, label='tnr')
    ax[1].grid(True)
    ax[1].legend(loc='right')   
    ax[1].set_title(title)
    ax[1].set_xlabel('epoch')
    #ax[1].set_ylabel('number')   
    
    
    line9, = ax[2].plot(labels, tpr, '-', linewidth=2, label='tpr', color = color_sequence[0]) 
    #ax[2].text(labels[-1] + 1, tpr[-1], 'tpr', fontsize = 12, color = color_sequence[0])
    line10, = ax[2].plot(labels, tnr, '-', linewidth=2, label='tnr', color = color_sequence[2])
    #ax[2].text(labels[-1] + 1, tnr[-1], 'tnr', fontsize = 12, color = color_sequence[2])
    
    ax[2].grid(True)
    ax[2].legend(loc='best')   
    ax[2].set_title(title)
    ax[2].set_xlabel('epoch')
    #ax[2].set_ylabel('number')  
    

    base_x = range(len(tpr))
    base_x = np.array(base_x)
    base_x = base_x * 1.0 / (len(tpr) - 1)   
    ax[3].plot(fpr, tpr, 'o')
    line9, = ax[3].plot(base_x, base_x, '-', linewidth = 2)
    ax[3].grid(True)   
    ax[3].set_title(title)
    ax[3].set_xlabel('false positive rate')
    ax[3].set_ylabel('true positive rate')   
    
    
    plt.show()


    
def draw_nodnet_loss(json_file):
    fp = open(json_file, 'r')
    data = json.load(fp)
    fp.close()

    loss = data['loss']['loss']
    classify = data['loss']['classify']

    epoch = len(loss)
    lables = range(1, epoch + 1)

    plt.xlabel('epoch')  
    plt.ylabel('train loss')  

    classify = np.array(classify, dtype=np.float64)
    loss = np.array(loss, dtype=np.float64)

    plt.plot(lables, loss,'r', label='loss')  
    plt.plot(lables, classify,'b',label='classify')

    #plt.xticks(lables, lables, rotation=0)  

    #plt.legend(bbox_to_anchor=[1.3, 1])  
    plt.grid()
    plt.show()

print('-------------------draw casenet train loss---------------------')
draw_casenet_loss(casenet_train)

print('-------------------draw casenet val loss---------------------')
draw_casenet_loss(casenet_val, True)

#print('-------------------draw casenet all loss---------------------')
#draw_casenet_loss(casenet_all, True)

#print('-------------------draw nodnet train loss---------------------')
#draw_nodnet_loss(nodnet_train)

#print('-------------------draw nodnet val loss---------------------')
#draw_nodnet_loss(nodnet_val)
