import math
import caffe
from caffe import layers as L
from caffe import params as P

en_kwargs = {'engine': 3}
num_anchors = 3
softmax_kwargs = {'axis': 1, 'ignore_label': -1}
smoothl1_kwargs = {'sigma': 1.0}
batch_size = 4
im_scale = 128
im_size = [batch_size, 1, im_scale, im_scale, im_scale]
coord_size = [batch_size, 3, im_scale/4, im_scale/4, im_scale/4]
label_size = [batch_size, 5, num_anchors, im_scale/4, im_scale/4, im_scale/4]

def add_conv_bn_relu(net, from_layer, index, num_output, num_input):
  stdv = 1. / math.sqrt(1.0 * 3 * 3 * 3 * num_input)
  net['conv%d'%index] = L.Convolution(net[from_layer], \
                    num_output = num_output, kernel_size= 3, pad = 1, \
                    weight_filler=dict(type = 'uniform', \
                    min = -stdv, max = stdv), \
                    bias_filler = dict(type = 'uniform', \
                    min = -stdv, max = stdv))
  net['bn%d'%index] = L.BatchNorm(net['conv%d'%index], \
                     use_global_stats = False, \
                     filler=dict(type = 'uniform', \
                     min = 0, max = 1) , include = dict(phase = 0), **en_kwargs)
  net['test_bn%d'%index] = L.BatchNorm(net['conv%d'%index], \
                     use_global_stats = True, \
                     include = dict(phase = 1), name = 'bn%d'%index, \
                     top = 'bn%d'%index, ntop = 0, **en_kwargs)
  net['scale_bn%d'%index] = L.Scale(net['bn%d'%index], bias_term = True)
  net['relu%d'%index] = L.ReLU(net['scale_bn%d'%index], \
                               in_place = True) #, **en_kwargs)

def ResBody(net, from_layer, index, input_channel, output_channel, stride = 1):
  stdv1 = 1. / math.sqrt(1.0 * 3 * 3 * 3 * input_channel)
  net[index + '_conv1'] = L.Convolution(net[from_layer], \
                          num_output = output_channel, kernel_size = 3, \
                          pad = 1, stride = stride, \
                          weight_filler = dict(type = 'uniform', \
                          min = -stdv1, max = stdv1), \
                          bias_filler = dict(type = 'uniform', \
                          min = -stdv1, max = stdv1))
  net[index + '_bn1'] = L.BatchNorm(net[index + '_conv1'], \
                          use_global_stats = False, \
                          filler=dict(type = 'uniform', \
                          min = 0, max = 1), include = dict(phase = 0), **en_kwargs)
  net['test_' + index + '_bn1'] = L.BatchNorm(net[index + '_conv1'], \
                          use_global_stats = True, \
                          include = dict(phase = 1), name = index + '_bn1', \
                          top = index + '_bn1', ntop = 0, **en_kwargs)
  net['scale_' + index + '_bn1'] = L.Scale(net[index + '_bn1'], bias_term = True)
  net[index + '_relu1'] = L.ReLU(net['scale_' + index + '_bn1'], \
                                 in_place = True) #, **en_kwargs)
  stdv2 = 1. / math.sqrt(1.0 * 3 * 3 * 3 * output_channel)
  net[index + '_conv2'] = L.Convolution(net[index + '_relu1'], \
                          num_output = output_channel, kernel_size = 3, \
                          pad = 1, \
                          weight_filler = dict(type = 'uniform', \
                          min = -stdv2, max = stdv2), \
                          bias_filler = dict(type = 'uniform', \
                          min = -stdv2, max = stdv2))
  net[index + '_bn2'] = L.BatchNorm(net[index + '_conv2'], \
                          use_global_stats = False, \
                          filler=dict(type = 'uniform', \
                          min = 0, max = 1), include = dict(phase = 0), **en_kwargs)
  net['test_' + index + '_bn2'] = L.BatchNorm(net[index + '_conv2'], \
                          use_global_stats = True, \
                          include = dict(phase = 1), name = index + '_bn2', \
                          top = index + '_bn2', ntop = 0, **en_kwargs)
  net['scale_' + index + '_bn2'] = L.Scale(net[index + '_bn2'], bias_term = True)
  

  if stride != 1 or input_channel != output_channel:
    stdv3 = 1. / math.sqrt(1.0 * 1 * 1 * 1 * input_channel)
    net[index + '_shortcut_conv'] = L.Convolution(net[from_layer], \
                          num_output = output_channel, kernel_size = 1, \
                          stride = stride, \
                          weight_filler = dict(type = 'uniform', \
                          min = -stdv3, max = stdv3), \
                          bias_filler = dict(type = 'uniform', \
                          min = -stdv3, max = stdv3))
    net[index + '_shortcut_bn'] = L.BatchNorm(
                          net[index + '_shortcut_conv'], \
                          use_global_stats = False, \
                          filler=dict(type = 'uniform', \
                          min = 0, max = 1), include = dict(phase = 0), **en_kwargs)
    net['test_' + index + '_shortcut_bn'] = \
                          L.BatchNorm(net[index + '_shortcut_conv'], \
                          use_global_stats = True, \
                          include = dict(phase = 1), name = index + '_shortcut_bn', \
                          top = index + '_shortcut_bn', ntop = 0, **en_kwargs)
    net['scale_' + index + '_shortcut_bn'] = \
                          L.Scale(net[index + '_shortcut_bn'], bias_term = True)
    net[index] = L.Eltwise(net['scale_' + index + '_bn2'], \
                           net['scale_' + index + '_shortcut_bn'], \
                           name = index + '_eltwise')
  else:
    net[index] = L.Eltwise(net['scale_' + index + '_bn2'], \
                           net[from_layer], name = index + '_eltwise')

  net[index + '_relu2'] =  L.ReLU(net[index], \
                             in_place = True) #, **en_kwargs)


net = caffe.NetSpec()

net.data, net.label, net.coord = L.Input(shape=[ \
                     dict(dim = im_size), \
                     dict(dim = label_size), \
                     dict(dim = coord_size)], ntop = 3)


add_conv_bn_relu(net, 'data', 1, 24, 1)
add_conv_bn_relu(net, 'relu1', 2, 24, 24)

net.maxpool1 = L.Pooling(net['relu2'], kernel_size = 2, stride = 2)
# res1
ResBody(net, 'maxpool1', 'res1a', 24, 32)
ResBody(net, 'res1a', 'res1b', 32, 32)

net.maxpool2 = L.Pooling(net['res1b'], kernel_size = 2, stride = 2)
# res2
ResBody(net, 'maxpool2', 'res2a', 32, 64)
ResBody(net, 'res2a', 'res2b', 64, 64)

net.maxpool3 =  L.Pooling(net['res2b'], kernel_size = 2, stride = 2)
# res3
ResBody(net, 'maxpool3', 'res3a', 64, 64)
ResBody(net, 'res3a', 'res3b', 64, 64)
ResBody(net, 'res3b', 'res3c', 64, 64)

net.maxpool4 = L.Pooling(net['res3c'], kernel_size = 2, stride = 2)
# res4
ResBody(net, 'maxpool4', 'res4a', 64, 64)
ResBody(net, 'res4a', 'res4b', 64, 64)
ResBody(net, 'res4b', 'res4c', 64, 64)

# deconv4
deconv_stdv = 1. / math.sqrt(1.0 * 64 * 2 *2 *2)
net['deconv4'] = L.Deconvolution(net.res4c, convolution_param = \
                 dict(num_output = 64, kernel_size=2, stride=2, \
                      weight_filler = dict(type = 'uniform', \
                      min = -deconv_stdv, max = deconv_stdv), \
                      bias_filler = dict(type = 'uniform', \
                      min = -deconv_stdv, max = deconv_stdv)))
net['deconv4_bn'] = L.BatchNorm(net['deconv4'], \
                     use_global_stats = False, \
                     filler=dict(type = 'uniform', \
                     min = 0, max = 1), include = dict(phase = 0), **en_kwargs)
net['test_' + 'deconv4_bn'] = L.BatchNorm(net['deconv4'], \
                     use_global_stats = True, \
                     include = dict(phase = 1), name = 'deconv4_bn', \
                     top = 'deconv4_bn', ntop = 0, **en_kwargs)
net['scale_' + 'deconv4_bn'] = L.Scale(net['deconv4_bn'], bias_term = True)
net['deconv4_relu'] = L.ReLU(net['scale_' + 'deconv4_bn'], \
                             in_place = True) #, **en_kwargs)

net['concat_4'] = L.Concat(net['deconv4_relu'], net['res3c'], \
                           concat_param = {'concat_dim': 1})

ResBody(net, 'concat_4', 'res_bw1a', 128, 64)
ResBody(net, 'res_bw1a', 'res_bw1b', 64, 64)
ResBody(net, 'res_bw1b', 'res_bw1c', 64, 64)

net['deconv3'] = L.Deconvolution(net.res_bw1c, convolution_param = \
                 dict(num_output = 64, kernel_size=2, stride=2, \
                      weight_filler = dict(type = 'uniform', \
                      min = -deconv_stdv, max = deconv_stdv), \
                      bias_filler = dict(type = 'uniform', \
                      min = -deconv_stdv, max = deconv_stdv)))
net['deconv3_bn'] = L.BatchNorm(net['deconv3'], \
                     use_global_stats = False, \
                     filler=dict(type = 'uniform', \
                     min = 0, max = 1), include = dict(phase = 0), **en_kwargs)
net['test_' + 'deconv3_bn'] = L.BatchNorm(net['deconv3'], \
                     use_global_stats = True, \
                     include = dict(phase = 1), name = 'deconv3_bn', \
                     top = 'deconv3_bn', ntop = 0, **en_kwargs)
net['scale_' + 'deconv3_bn'] = L.Scale(net['deconv3_bn'], bias_term = True)
net['deconv3_relu'] = L.ReLU(net['scale_' + 'deconv3_bn'], \
                             in_place = True) #, **en_kwargs)


net['concat_3'] = L.Concat(net['deconv3_relu'], net['res2b'], net['coord'], \
                           concat_param = {'concat_dim': 1})

ResBody(net, 'concat_3', 'res_bw2a', 131, 128)
ResBody(net, 'res_bw2a', 'res_bw2b', 128, 128)
ResBody(net, 'res_bw2b', 'res_bw2c', 128, 128)

net['dropout'] = L.Dropout(net['res_bw2c'],dropout_ratio=0.5)

output_stdv = 1.0 / math.sqrt(1.0 * 128 * 1 * 1 * 1)
net['rpn_output'] = L.Convolution(net['dropout'], \
                          num_output = 64, kernel_size = 1, \
                          weight_filler = dict(type = 'uniform', \
                          min = -output_stdv, max = output_stdv), \
                          bias_filler = dict(type = 'uniform', \
                          min = -output_stdv, max = output_stdv))

net['rpn_output_relu'] = L.ReLU(net['rpn_output'], \
                                in_place = False)#, **en_kwargs)

output_stdv = 1.0 / math.sqrt(1.0 * 64 * 1 * 1 * 1)
net['output'] = L.Convolution(net['rpn_output_relu'], \
                   num_output = 5 * num_anchors, kernel_size = 1, \
                   weight_filler = dict(type = 'uniform', \
                   min = -output_stdv, max = output_stdv), \
                   bias_filler = dict(type = 'uniform', \
                   min = -output_stdv, max = output_stdv))

net['rpn_pos_cls_pred'], net['rpn_pos_cls_labels'], \
   net['rpn_neg_cls_pred'], net['rpn_neg_cls_labels'], \
   net['rpn_bbox_x_pred'], net['rpn_bbox_x_targets'], \
   net['rpn_bbox_y_pred'], net['rpn_bbox_y_targets'], \
   net['rpn_bbox_z_pred'], net['rpn_bbox_z_targets'], \
   net['rpn_bbox_d_pred'], net['rpn_bbox_d_targets'], \
   net['rpn_bbox_inside_weights'], net['rpn_bbox_outside_weights'] \
      =  L.Python(net['label'], net['output'], \
         python_param = dict(module = 'models.anchor_target_layer', \
                             layer = 'AnchorTargetLayer'), ntop = 14, \
                             name = 'rpn_anchor')

net['rpn_pos_cls_loss'] = L.SigmoidCrossEntropyLoss(net['rpn_pos_cls_pred'], \
                           net['rpn_pos_cls_labels'], \
                           propagate_down = [True, False], \
                           loss_param = dict(ignore_label = -1, normalization = 1), \
                           loss_weight = 0.5)
net['rpn_neg_cls_loss'] = L.SigmoidCrossEntropyLoss(net['rpn_neg_cls_pred'], \
                           net['rpn_neg_cls_labels'], \
                           propagate_down = [True, False], \
                           loss_param = dict(ignore_label = -1, normalization = 1), \
                           loss_weight = 0.5)

net['rpn_bbox_x_loss'] = L.SmoothL1Loss(net['rpn_bbox_x_pred'], \
                          net['rpn_bbox_x_targets'], \
                          net['rpn_bbox_inside_weights'], \
                          net['rpn_bbox_outside_weights'], \
                          loss_weight = 1.0)

net['rpn_bbox_y_loss'] = L.SmoothL1Loss(net['rpn_bbox_y_pred'], \
                          net['rpn_bbox_y_targets'], \
                          net['rpn_bbox_inside_weights'], \
                          net['rpn_bbox_outside_weights'], \
                          loss_weight = 1.0)

net['rpn_bbox_z_loss'] = L.SmoothL1Loss(net['rpn_bbox_z_pred'], \
                          net['rpn_bbox_z_targets'], \
                          net['rpn_bbox_inside_weights'], \
                          net['rpn_bbox_outside_weights'], \
                          loss_weight = 1.0)

net['rpn_bbox_d_loss'] = L.SmoothL1Loss(net['rpn_bbox_d_pred'], \
                          net['rpn_bbox_d_targets'], \
                          net['rpn_bbox_inside_weights'], \
                          net['rpn_bbox_outside_weights'], \
                          loss_weight = 1.0)


with open('train.prototxt', 'w') as f:
  f.write(str(net.to_proto()))


