import caffe
from multiprocessing import Queue, Process
import numpy as np
import time
import data
import psutil
from split_combine import SplitComb
import config
import os
import os.path

config = config.config

class GetPBB(object):
  def __init__(self, config):
    self.stride = config['stride']
    self.anchors = np.asarray(config['anchors'])

  def __call__(self, output,thresh = -3, ismask=False):
    stride = self.stride
    anchors = self.anchors
    output = np.copy(output)
    offset = (float(stride) - 1) / 2
    output_size = output.shape
    oz = np.arange(offset, offset + stride * (output_size[0] - 1) + 1, stride)
    oh = np.arange(offset, offset + stride * (output_size[1] - 1) + 1, stride)
    ow = np.arange(offset, offset + stride * (output_size[2] - 1) + 1, stride)

    output[:, :, :, :, 1] = oz.reshape((-1, 1, 1, 1)) + \
                      output[:, :, :, :, 1] * anchors.reshape((1, 1, 1, -1))
    output[:, :, :, :, 2] = oh.reshape((1, -1, 1, 1)) + \
                      output[:, :, :, :, 2] * anchors.reshape((1, 1, 1, -1))
    output[:, :, :, :, 3] = ow.reshape((1, 1, -1, 1)) + \
                      output[:, :, :, :, 3] * anchors.reshape((1, 1, 1, -1))
    #print('output_0=',output)
    output[:, :, :, :, 4] = np.exp(output[:, :, :, :, 4]) * \
                      anchors.reshape((1, 1, 1, -1))

    #print('output_1=',output)
    mask = output[..., 0] > thresh
    xx,yy,zz,aa = np.where(mask)
    #print('xx=',xx.shape)

    output = output[xx,yy,zz,aa]
    if ismask:
      return output,[xx,yy,zz,aa]
    else:
      return output

def test():
  test_net = caffe.Net('./models/test.prototxt', caffe.TEST)
  test_net.copy_from('./snapshot/_iter_60000.caffemodel')
  losses = []

  margin = 32
  sidelen = 144
  split_comber = SplitComb(sidelen,config['max_stride'],\
                                config['stride'],margin,config['pad_value'])
  test_data = data.DataBowl3Detector(config['datadir'], config['val_data'],\
                                  phase='test', split_comber=split_comber)
  cur_num = 0
  start_time = time.time()
  filenames = test_data.filenames
  testing_times = []
  while cur_num + config['TEST_BATCH'] <= len(filenames):
    start_test = time.time()
    test_blobs = test_data.get_minibatch(np.array([cur_num]), 'TEST')
    name = filenames[cur_num].split('/')[-1].split('_')[0]
    print 'name = ', name
    cur_num += config['TEST_BATCH']

    input_data = test_blobs[0]
    target = test_blobs[1]
    coord = test_blobs[2]
    nzhw = test_blobs[3]

    lbb = target #target[0]

    n_per_run = 1
    splitlist = range(0,len(input_data)+1,n_per_run)
    print 'splitlist ', splitlist
    outputlist = []

    for i in range(len(splitlist)-1):
      inputdata = input_data[splitlist[i]:splitlist[i+1]]
      inputcoord = coord[splitlist[i]:splitlist[i+1]]

      test_net.blobs['data'].reshape(*inputdata.shape)
      test_net.blobs['coord'].reshape(*inputcoord.shape)
      test_net.blobs['data'].data[...] = inputdata[...]
      test_net.blobs['coord'].data[...] = inputcoord[...]

      net_output = test_net.forward()
      pred = np.zeros(test_net.blobs['output'].data.shape)
      pred[...] = test_net.blobs['output'].data[...]
      pred_shape = pred.shape
      pred = np.reshape(pred, (pred.shape[0], 3, 5, pred.shape[2], \
                                    pred.shape[3], pred.shape[4]))
      pred = np.transpose(pred, (0, 3, 4, 5, 1, 2))
      outputlist.append(pred)

    output = np.concatenate(outputlist,0)
    output = split_comber.combine(output,nzhw=nzhw)

    thresh = -3
    get_pbb = GetPBB(config)
    pbb,mask = get_pbb(output,thresh,ismask=True)

    e = time.time()
    time_per_test = e - start_test
    testing_times.append(time_per_test)
    np.save(os.path.join(config['save_dir'], name+'_pbb.npy'), pbb)
    np.save(os.path.join(config['save_dir'], name+'_lbb.npy'), lbb)

  end_time = time.time()
  average_test_time = sum(testing_times) / len(testing_times)
  print '{} CTs were tested, total testing time is {}, average is {}' \
           .format(len(testing_times), sum(testing_times), average_test_time)
  print 'total time is {}'.format(end_time - start_time)


if __name__ == '__main__':
  caffe.set_mode_cpu()
  test()
