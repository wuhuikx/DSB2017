import caffe
import numpy as np

num_anchors = 3

class AnchorTargetLayer(caffe.Layer):
  """
  AnchorTargetLayer

  bottom[0] = net['label']   (n * 5 * num_anchors * d * h * w)
  bottom[1] = net['output']  (n * (5*num_anchors) * d * h * w)

  top[0] = rpn_pos_cls_pred
  top[1] = rpn_pos_cls_labels
  top[2] = rpn_neg_cls_pred
  top[3] = rpn_neg_cls_labels

  top[4] = rpn_bbox_x_pred
  top[5] = rpn_bbox_x_targets
  top[6] = rpn_bbox_y_pred
  top[7] = rpn_bbox_y_targets
  top[8] = rpn_bbox_z_pred
  top[9] = rpn_bbox_z_targets
  top[10] = rpn_bbox_d_pred
  top[11] = rpn_bbox_d_targets

  top[12] = rpn_bbox_inside_weights
  top[13] = rpn_bbox_outside_weights
  """
  def setup(self, bottom, top):
    shape_bottom = bottom[0].data.shape
    labels_shape = [shape_bottom[0], shape_bottom[2], shape_bottom[3], \
                    shape_bottom[4], shape_bottom[5]]
    for i in range(14):
      top[i].reshape(*labels_shape)

  def reshape(self, bottom, top):
    """Reshape was done in setup and forward"""
    pass

  def forward(self, bottom, top):
    shape_bottom0 = bottom[0].data.shape
    shape_bottom1 = bottom[1].data.shape

    # check data shape
    assert (shape_bottom0[0] == shape_bottom1[0]), \
                 'batch size should be equal'
    assert (shape_bottom0[1] == 5), \
                 'bottom[0].shape[1] should be 5'
    assert (shape_bottom0[2] == 3), \
                 'bottom[0].shape[2] should be 3'
    assert (shape_bottom1[1] == 15), \
                 'bottom[1].shape[1] should be 15'
    assert (shape_bottom0[3] == shape_bottom1[2]), \
                 'D should be equal'
    assert (shape_bottom0[4] == shape_bottom1[3]), \
                 'H should be equal'
    assert (shape_bottom0[5] == shape_bottom1[4]), \
                 'W should be equal'

    label_shape = [shape_bottom0[0], shape_bottom0[2], shape_bottom0[3], \
                    shape_bottom0[4], shape_bottom0[5]]
    for i in range(14):
      top[i].reshape(*label_shape)

    labels = np.zeros(label_shape, dtype = np.float32)
    labels_pred = np.zeros(labels.shape)
    labels[:, :, :, :, :] = bottom[0].data[:, 0, :, :, :, :]
    labels_pred[:, :, :, :, :] = bottom[1].data[:, 0::5, :, :, :]

    pos_inds = np.where(labels > 0.5)
    pos_labels = np.empty(labels.shape, dtype = np.float32)
    pos_labels.fill(-1)

    pos_labels[pos_inds] = 1

    neg_inds = np.where(labels < -0.5)
    neg_labels = np.empty(labels.shape, dtype = np.float32)
    neg_labels.fill(-1)
    neg_num = 2 * labels.shape[0]

    if self.phase == 0:
      neg_inds = self.hardmining(neg_inds, labels_pred, neg_num)
    neg_labels[neg_inds] = 0

    bbox_inside_weights = np.zeros(labels.shape, dtype = np.float32)
    bbox_inside_weights[pos_inds] = 1.0
    bbox_outside_weights = np.zeros(labels.shape, dtype = np.float32)
    if len(pos_inds[0]) > 0:
      bbox_outside_weights[pos_inds] = 1.0 * labels.shape[0] / len(pos_inds[0])

    # top[0] = rpn_pos_cls_pred
    top[0].data[...] = labels_pred[...]

    # top[1] = rpn_pos_cls_labels
    top[1].data[...] = pos_labels[...]

    # top[2] = rpn_neg_cls_pred
    top[2].data[...] = labels_pred[...]

    # top[3] = rpn_neg_cls_labels
    top[3].data[...] = neg_labels[...]

    # top[4] = rpn_bbox_x_pred
    top[4].data[:, :, :, :, :] = bottom[1].data[:, 1::5, :, :, :]
    # top[5] = rpn_bbox_x_targets
    top[5].data[:, :, :, :, :] = bottom[0].data[:, 1, :, :, :, :]

    # top[6] = rpn_bbox_y_pred
    top[6].data[:, :, :, :, :] = bottom[1].data[:, 2::5, :, :, :]
    # top[7] = rpn_bbox_y_targets
    top[7].data[:, :, :, :, :] = bottom[0].data[:, 2, :, :, :, :]

    # top[8] = rpn_bbox_z_pred
    top[8].data[:, :, :, :, :] = bottom[1].data[:, 3::5, :, :, :]
    # top[9] = rpn_bbox_9_targets
    top[9].data[:, :, :, :, :] = bottom[0].data[:, 3, :, :, :, :]

    # top[10] = rpn_bbox_d_pred
    top[10].data[:, :, :, :, :] = bottom[1].data[:, 4::5, :, :, :]
    # top[11] = rpn_bbox_d_targets
    top[11].data[:, :, :, :, :] = bottom[0].data[:, 4, :, :, :, :]

    # top[12] = rpn_bbox_inside_weights
    top[12].data[...] = bbox_inside_weights[...]
    # top[5] = rpn_bbox_outside_weights
    top[13].data[...] = bbox_outside_weights[...]

  def backward(self, top, propagate_down, bottom):
    """propagates diff to bottom[0]"""

    bottom[1].diff[:, 0::5, :, :, :] = top[0].diff[:, :, :, :, :] + \
                                         top[2].diff[:, :, :, :, :]
    bottom[1].diff[:, 1::5, :, :, :] = top[4].diff[:, :, :, :, :]
    bottom[1].diff[:, 2::5, :, :, :] = top[6].diff[:, :, :, :, :]
    bottom[1].diff[:, 3::5, :, :, :] = top[8].diff[:, :, :, :, :]
    bottom[1].diff[:, 4::5, :, :, :] = top[10].diff[:, :, :, :, :]

    

  def hardmining(self, neg_inds, cls_score, k):
    """Hard negative mining"""
    cls = cls_score[neg_inds]

    if len(cls) <= k:
      return neg_inds

    arg_selected = cls.argsort()[-k:]
    return (neg_inds[0][arg_selected], neg_inds[1][arg_selected], \
            neg_inds[2][arg_selected], neg_inds[3][arg_selected], \
            neg_inds[4][arg_selected])

