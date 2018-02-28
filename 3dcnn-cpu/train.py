import caffe
from multiprocessing import Queue, Process
import numpy as np
import time
from data import *
import data
import psutil
from split_combine import SplitComb
import config
import random

config = config.config


class GetData(object):
  def __init__(self, dbd, phase = 'TRAIN'):
    self.max_iters = config['max_iters']
    self.use_prefetch = config['use_prefetch']
    self.dataset = dbd
    self.phase = phase

    self.origin_images = len(self.dataset.bboxes)
    self.total_images = int(self.origin_images / (1 - config['r_rand']))
    self._shuffle_inds()
 
    if config['use_prefetch']:
      self._blob_queue = Queue(10)
      self._prefetch_process = Process(target = self.prepareDataThread, \
                                args=(self._blob_queue, ))
      self._prefetch_process.start()
      p = psutil.Process(self._prefetch_process.pid)
      p.cpu_affinity([67])
      def cleanup():
        print 'Terminating Fetcher Process'
        self._prefetch_process.terminate()
        self._prefetch_process.join()

      import atexit
      atexit.register(cleanup)

  def prepareDataThread(self, queue):
    while True:
      im_inds = self._get_next_minibatch_inds()
      blobs = self.dataset.get_minibatch(im_inds, self.phase)
      queue.put(blobs)

  def _shuffle_inds(self):
    self._perm = np.random.permutation(np.arange(self.total_images))
    self._cur = 0 

  def _get_next_minibatch_inds(self):
    if self._cur + config['IMS_PER_BATCH'] > self.total_images:
      self._shuffle_inds()

    im_inds = self._perm[self._cur:self._cur + config['IMS_PER_BATCH']]
    self._cur += config['IMS_PER_BATCH']

    return im_inds

  def _get_next_minibatch(self):
    if self.use_prefetch:
      return self._blob_queue.get()
    else:
      im_inds = self._get_next_minibatch_inds()

    return self.dataset.get_minibatch(im_inds)

class SolverWrapper(object):
  def __init__(self, solver_path):
    self._solver = caffe.SGDSolver(solver_path)

    self.max_iters = config['max_iters']
    self.use_prefetch = config['use_prefetch']
    self.dataset = DataBowl3Detector(config['datadir'], \
                     config['train_data'], phase = 'train',split_comber=None)
    self.val_data = DataBowl3Detector(config['datadir'], \
                     config['val_data'], phase = 'val',split_comber=None)

    self.getdata_train = GetData(self.dataset, 'TRAIN')
    self.getdata_val = GetData(self.val_data, 'VAL')

  def train(self):
    losses = []
    epoch_num = 0
    cur_num = 0
    loss_file = open('loss_log.txt', 'w')
    loss_file.write('Epoch' + '\t' + 'train_loss' + '\t' + 'val_loss' + '\n')

    #self._solver.net.copy_from('./snapshot/_iter_60000_rpn.caffemodel')

    for i in range(self.max_iters):
      start_time = time.time()
      train_blobs = self.getdata_train._get_next_minibatch()
      train_blobs[1] = np.transpose(train_blobs[1],(0, 5, 4, 1, 2, 3))
      print('get blob time: ', 1000*(time.time() - start_time))
      print('blobs[0][...]=', (train_blobs[0][...]).shape)
      print('blobs[1][...]=', (train_blobs[1][...]).shape)
      print('blobs[2][...]=', (train_blobs[2][...]).shape)

      self._solver.net.blobs['data'].data[...] = train_blobs[0][...]
      self._solver.net.blobs['label'].data[...] = train_blobs[1][...]
      self._solver.net.blobs['coord'].data[...] = train_blobs[2][...]
      self._solver.step(1)

      loss = 0.5 * self._solver.net.blobs['rpn_pos_cls_loss'].data + \
             0.5 * self._solver.net.blobs['rpn_neg_cls_loss'].data + \
             self._solver.net.blobs['rpn_bbox_x_loss'].data + \
             self._solver.net.blobs['rpn_bbox_y_loss'].data + \
             self._solver.net.blobs['rpn_bbox_z_loss'].data + \
             self._solver.net.blobs['rpn_bbox_d_loss'].data
      losses.append(float(loss))

      cur_num += config['IMS_PER_BATCH'] 
      if cur_num + config['IMS_PER_BATCH'] > self.getdata_train.total_images:
        ave_loss = sum(losses) / len(losses)
        losses = []
        print 'Training Epoch {} Finished, average_loss = {}'\
                .format(epoch_num, ave_loss)

        ave_val_loss = self.validate()
        loss_file.write(str(epoch_num) + '\t' + str(ave_loss) + '\t' + \
                        str(ave_val_loss) + '\n')
        loss_file.flush()

        epoch_num += 1
        cur_num = 0

  def validate(self):
    self._solver.test_nets[0].share_with(self._solver.net)
    losses = []
    cur_num = 0
    print self.getdata_val.total_images
    while cur_num + config['IMS_PER_BATCH'] <= self.getdata_val.total_images:
      cur_num += config['IMS_PER_BATCH']
      start_time = time.time()
      val_blobs = self.getdata_val._get_next_minibatch()
      val_blobs[1] = np.transpose(val_blobs[1],(0, 5, 4, 1, 2, 3))
      self._solver.test_nets[0].blobs['data'].data[...] = val_blobs[0][...]
      self._solver.test_nets[0].blobs['label'].data[...] = val_blobs[1][...]
      self._solver.test_nets[0].blobs['coord'].data[...] = val_blobs[2][...]

      print '--------------- validation --------------', \
                cur_num / config['IMS_PER_BATCH']

      output = self._solver.test_nets[0].forward()
      print 'bbox_d_loss: ',  \
            self._solver.test_nets[0].blobs['rpn_bbox_d_loss'].data
      print 'bbox_x_loss: ',  \
            self._solver.test_nets[0].blobs['rpn_bbox_x_loss'].data
      print 'bbox_y_loss: ',  \
            self._solver.test_nets[0].blobs['rpn_bbox_y_loss'].data
      print 'bbox_z_loss: ',  \
            self._solver.test_nets[0].blobs['rpn_bbox_z_loss'].data
      print 'rpn_neg_cls_loss: ',  \
            self._solver.test_nets[0].blobs['rpn_neg_cls_loss'].data
      print 'rpn_pos_cls_loss: ',  \
            self._solver.test_nets[0].blobs['rpn_pos_cls_loss'].data

      loss = 0.5 * self._solver.test_nets[0].blobs['rpn_pos_cls_loss'].data + \
             0.5 * self._solver.test_nets[0].blobs['rpn_neg_cls_loss'].data + \
             self._solver.test_nets[0].blobs['rpn_bbox_x_loss'].data + \
             self._solver.test_nets[0].blobs['rpn_bbox_y_loss'].data + \
             self._solver.test_nets[0].blobs['rpn_bbox_z_loss'].data + \
             self._solver.test_nets[0].blobs['rpn_bbox_d_loss'].data
      losses.append(float(loss))
    ave_loss = sum(losses) / len(losses)
    print 'average loss on validation data set: ', ave_loss
    return ave_loss

if __name__ == '__main__':
  caffe.set_mode_cpu()
  solver_wrapper = SolverWrapper('./models/solver.prototxt')

  solver_wrapper.train()
