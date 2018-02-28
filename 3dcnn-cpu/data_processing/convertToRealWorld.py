import os
import shutil

import numpy as np
import pandas as pd
import SimpleITK as sitk


def get_short_long(uid_path):
    
    fp = open(uid_path, 'r')
    uids = fp.readlines()
    fp.close()


    uids = [uid.split('\n')[0] for uid in uids]
    uids = [uid.split(',') for uid in uids]

    short_to_long = {s: l for s, l in uids}

    long_to_short = {l: s for s, l in uids}

    return short_to_long, long_to_short


def voxel_to_world(voxel, origin, spacing, isflip):
    stretchedVoxelCoord = voxel * spacing
    if isflip:
        direction = np.array([1, -1, -1])
    else:
        direction = np.array([1, 1, 1])

    worldCoord = stretchedVoxelCoord * direction + origin

    return worldCoord


def load_itk_image(filename):
    with open(filename) as f:
        contents = f.readlines()
        line = [k for k in contents if k.startswith('TransformMatrix')][0]
        transformM = np.array(line.split(' = ')[1].split(' ')).astype('float')
        transformM = np.round(transformM)
        if np.any(transformM != np.array([1, 0, 0, 0, 1, 0, 0, 0, 1])):
            isflip = True
        else:
            isflip = False

    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)

    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))

    return numpyImage, numpyOrigin, numpySpacing, isflip


def get_state(state={}, index='000'):
    origin = np.array(state['origin'])
    is_flip = state['isflip']
    spacing = np.array(state['spacing'])
    extend_box = np.array(state['extendbox'])[:, 0]
    mask = np.array(state['mask'])
    resolution = np.array([1, 1, 1])

    return mask, origin, spacing, resolution, is_flip, extend_box


def convert_coordinate(state=(), coord=None):
    mask, origin, spacing, resolution, is_flip, extend_box = state
    D = coord[-1]
    zxy = coord[:-1]
    
    D = D * resolution[1] / spacing[1]
    zxy = zxy + extend_box
    zxy = zxy * resolution / spacing

    if is_flip:
        zxy[1:] = mask[1:] - zxy[1:]
    D = D * spacing[1]
    
    zxy = voxel_to_world(zxy, origin, spacing, is_flip)
    xyz = zxy[::-1]

    new_coord = np.concatenate([xyz, [D]])

    return new_coord


def read_csv(src):
    fp = open(src, 'r')
    lines = fp.readlines()
    fp.close()

    return lines


def save_csv(dest, lines):
    fp = open(dest, 'w')
    fp.writelines(lines)
    fp.close()


def get_line_result(line):
    name, x, y, z, p = (line.split('\n')[0]).split(',')
    x = float(x)
    y = float(y)
    z = float(z)

    return name, [z, x, y, 0.0], p


def convert_csv(states={}, short_long={}, src='', dest=''):
    old_lines = read_csv(src)
    title = old_lines[0]

    old_lines = old_lines[1:]

    new_lines = [
        title,
    ]

    for line in old_lines:
        s_name, zxyd, p = get_line_result(line)
        state = states[s_name]

        state = get_state(state)
        l_name = short_long[s_name]
        #l_name = s_name

        xyzd = convert_coordinate(state, zxyd)

        xyz = xyzd[:-1]
        str_xyz = [str(i) for i in xyz]
        new_line = str_xyz

        new_line.insert(0, l_name)
        new_line.append(p)

        new_line = ','.join(new_line)
        new_line += '\n'
        new_lines.append(new_line)
    

    save_csv(dest, new_lines)



_short_csv_src = './labels/shorter.csv'
_prep_state_src = './labels/luna_segments_reso.json'

subset = '0'
_pred_csv_src = '../save/pred_val.csv'
_pred_csv_dest = '../save/test_val.csv'


_pred_csv = pd.read_csv(_pred_csv_src)
_pred_data = _pred_csv.values

# load prep state
fp = open(_prep_state_src, 'r')
states = json.load(fp)
fp.close()

short_long, long_short = get_short_long(_short_csv_src)


convert_csv(states, short_long, _pred_csv_src, _pred_csv_dest)
