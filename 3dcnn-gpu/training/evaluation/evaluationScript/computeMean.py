import numpy as np
#fp = open('./wh/nodnet/evaluation/0/froc_pred_val_bootstrapping.csv', 'r')
fp = open('/home/wuhui/3dcnn/result-wh/nodnet/cpu/input128/evaluation/froc_pred_val_bootstrapping.csv', 'r')
#fp = open('../evaluation/evaluationScript/wh/nodnet/evaluation/froc_pred_val_bootstrapping.csv', 'r')
csv = fp.readlines()
fp.close()

res = []
num = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
j= 0
for i in csv[1:-1]:
    if float(i.split(',')[0]) - num[j] > 0.0 or float(i.split(',')[0]) - num[j] == 0.0:    
        res.append(np.array(i.split(',')))
        j = j + 1    
res.append(np.array(csv[-1].split(',')))
res = np.array(res)

print('res =', res)
print('mean =', np.mean(np.array(res[:, 1]).astype(float)))