import numpy as np
import os
import matplotlib.pyplot as plt


class sample_data(object):
    def __init__(self):
        self.lbb = None
        self.pbb = None
        self.slices = None
        self.label = None

def show(image, dir_path, idx):
    fig = plt.figure(0)
    plt.imshow(image, cmap='gray')
    plt.savefig(os.path.join(dir_path,str(idx) + '.png'))
    #plt.show()


def calculate_nodule_edge(nodule, w, h):
    nodule_x = np.round(nodule['X'])
    nodule_y = np.round(nodule['Y'])
    nodule_d = np.round(nodule['D'])

    print('nodule: ({0}, {1}, {2})'.format(nodule_x, nodule_y, nodule_d))
    nodule_left_top_x = int(nodule_x - nodule_d / 2)
    if nodule_left_top_x < 0:
        nodule_left_top_x = 0
        
    nodule_left_top_y = int(nodule_y - nodule_d / 2)
    if nodule_left_top_y < 0:
        nodule_left_top_y = 0
        
    nodule_right_bottom_x = int(nodule_left_top_x + nodule_d)
    if nodule_right_bottom_x > h - 1:
        nodule_right_bottom_x = h-1
        
    nodule_right_bottom_y = int(nodule_left_top_y + nodule_d)
    if nodule_right_bottom_y >= w - 1:
        nodule_right_bottom_y = w - 1
    print('nodule: ({0}, {1}, {2}, {3})'.format(nodule_left_top_x,
                                                nodule_left_top_y,
                                                nodule_right_bottom_x,
                                                nodule_right_bottom_y))
    nodule_edge = []

    for y in range(nodule_left_top_y, nodule_right_bottom_y + 1, 1):
        nodule_edge.append((nodule_left_top_x, y))
        nodule_edge.append((nodule_right_bottom_x, y))

    for x in range(nodule_left_top_x, nodule_right_bottom_x +1, 1):
        nodule_edge.append((x, nodule_left_top_y))
        nodule_edge.append((x, nodule_right_bottom_y))
        
    return nodule_edge

def gen_mask(edge, size=(1,1)):
    mask = np.zeros(size, dtype=np.int16)
    for x, y in edge:
        mask[x][y] = 255
    return mask

def bbox_image(image, mask):
    image = image.astype(np.int16)
    mask = mask.astype(np.int16)
    image = image + mask
    image[image > 255] = 255
    return image

def load_data(sample):
    lbb_path = os.path.join('/home/data/data/result/bbox', sample + '_lbb.npy')
    pbb_path = os.path.join('/home/data/data/result/bbox', sample + '_pbb.npy')
    prep_data_path = os.path.join('/home/data/data/result/prep', sample + '_clean.npy')
    prep_label_path = os.path.join('/home/data/data/result/prep', sample + '_label.npy')

    data = sample_data()
    data.lbb = np.load(lbb_path)
    data.label = np.load(prep_label_path)
    data.slices = np.load(prep_data_path)
    
    pbb = np.load(pbb_path)
    pbb_d = []
    for score, z, x, y, d in pbb:
        pbb_d.append({'X':x, 'Y':y, 'Z':z,'D':d, 'sorce':score})

    pbb = pbb_d
    #pbb.sort(key = lambda x: x['D'], reverse=True)
    data.pbb = pbb

    return data

def load_data1(pbb, clean):
    #prep_data_path = os.path.join('/home/data/data/result/prep', idx + '_clean.npy')
    data = sample_data()
    data.slices = clean #np.load(prep_data_path)
    #pbb = np.load(pbb_path)
    pbb_d = []
    for score, z, x, y, d in pbb:
        pbb_d.append({'X':x, 'Y':y, 'Z':z,'D':d, 'sorce':score})

    pbb = pbb_d
    data.pbb = pbb

    return data

def save_img(pbb, clean, file_path):
    sample = load_data1(pbb, clean)
    print(sample.slices.shape)
    _,_, w, h = sample.slices.shape
    for i in range(len(sample.pbb)):
        nodule_edge = calculate_nodule_edge(sample.pbb[i], w, h)
        nodule_z = sample.pbb[i]['Z']
        print('Z:', nodule_z)
        image_z = get_sample_image(sample, nodule_z)
        mask = gen_mask(nodule_edge, image_z.shape)
        bbox = bbox_image(image_z, mask)
        show(bbox, file_path, i)
        print('{0} {1}'.format(i, sample.pbb[i]))
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n')

def plot_bbox(pbb, clean):
    sample = load_data1(pbb, clean)
    print(sample.slices.shape)
    _,_,w,h=sample.slices.shape
    for i in range(len(sample.pbb)):
        nodule_edge = calculate_nodule_edge(sample.pbb[i], w, h)
        nodule_z = sample.pbb[i]['Z']
        print('Z:', nodule_z)
        image_z = get_sample_image(sample, nodule_z)
        mask = gen_mask(nodule_edge, image_z.shape)
        bbox = bbox_image(image_z, mask)
        fig = plt.figure(0)
        plt.imshow(bbox, cmap='gray')   
        plt.show()

        
    
def get_sample_image(sample, z):
    slices = sample.slices[0]
    z = int(np.round(z))

    return slices[z]

if __name__ == '__main__':
#     sample = load_data('605')
#     for i in range(len(sample.pbb)):
#         print(i)
#         nodule_edge = calculate_nodule_edge(sample.pbb[i])
#         nodule_z = sample.pbb[i]['Z']
#         image_z = get_sample_image(sample, nodule_z)
#         mask = gen_mask(nodule_edge, image_z.shape)
#         bbox = bbox_image(image_z, mask)
#         show(bbox)
    
    print('down')
