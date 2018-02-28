config = {}

subset = 0

config['solver'] = './models/train.prototxt'

config['anchors'] = [10.0, 30.0, 60.0]
config['crop_size'] = [128, 128, 128] #[128, 128, 128]
config['bound_size'] = 12 #12
config['stride'] = 4
config['max_stride'] = 16
config['pad_value'] = 170
config['r_rand'] = 0.3
config['aug_scale'] = True
config['augtype'] = {'flip':True,'swap':False,'scale':True,'rotate':False}
config['num_neg'] = 800
config['th_neg'] = 0.02
config['th_pos_train'] = 0.5
config['th_pos_val'] = 1

config['batch_size'] = 1
config['IMS_PER_BATCH'] = 4 #4
config['VAL_BATCH'] = 4 #4
config['TEST_BATCH'] = 1
config['r_rand'] = 0.3
config['use_prefetch'] = True
config['max_iters'] = 100000


config['datadir'] = '../../prep_luna_full' # preprocessed data dir
config['train_data'] = '../../inputdata/luna_train_{0}.npy'.format(subset) # train data name list
config['val_data'] = '../../inputdata/luna_val_{0}.npy'.format(subset) # validation data name list
config['save_dir'] = './save/'
