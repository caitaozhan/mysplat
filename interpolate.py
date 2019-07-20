import numpy as np
import re
import math
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, median_absolute_error, mean_squared_error
from utility import distance



def get_data(file):
    '''get the pathloss data of one tx from one file
    Args:
        file -- str
    Return:
        np.2darray
    '''
    int4 = r'(\d{4,4})'
    pattern = r'.*\D{}'.format(int4)
    p = re.compile(pattern)
    m = p.match(txfile)
    tx = int(m.group(1))
    print(tx)
    return np.loadtxt(txfile, delimiter=',')


def get_all_data(directory):
    '''get all the pathloss of all the tx from all files inside a directory
    Args:
        directory -- str -- folder that has one transmitter to all sensor pathloss data.
    Return:
        np.2darray -- for fspl model,  shape = (h, h), where h is the number of hypothesis, eg. grid_len x grid_len
        np.2darray -- for itwom model, shape = (h, h), where h is the number of hypothesis, eg. grid_len x grid_len
    '''
    files = sorted(glob.glob(directory + '/*'))
    fspl  = []
    itwom = []
    for f in files:
        try:
            data = np.loadtxt(f, delimiter=',')
            fspl.append(data[0])
            itwom.append(data[1])
        except:
            print(f, 'cannot np.loadtxt')
    return np.array(fspl), np.array(itwom)


def write_data(fspl, itwom, directory):
    '''write the pathloss of all tx to different files of a folder
    Args:
        fspl  -- np.2darray
        itwom -- np.2darray
        directory  -- str
    '''
    num_hypo = len(fspl)
    for i in range(num_hypo):
        filename = directory + '/{:04}'.format(i)
        with open(filename, 'w') as f:
            f.write(','.join(map(lambda x: str(x), fspl[i])))
            f.write('\n')
            f.write(','.join(map(lambda x: str(x), itwom[i])))


def visualize_tx(txfile):
    '''visualize the transmitter
    Args:
        txfile -- str -- eg. input/0000
    '''
    data = get_data(txfile)
    fspl  = data[0]
    itwom = data[1]
    clean_itwom(itwom, fspl)

    model = fspl
    grid_len = int(math.sqrt(len(model)))
    grid = np.zeros((grid_len, grid_len))
    model = model.reshape((grid_len, grid_len))
    for x in range(grid_len):
        for y in range(grid_len):
            grid[grid_len -1 - y][x] = model[x][y]

    model = itwom
    grid_len = int(math.sqrt(len(model)))
    grid2 = np.zeros((grid_len, grid_len))
    model = model.reshape((grid_len, grid_len))
    for x in range(grid_len):
        for y in range(grid_len):
            grid2[grid_len -1 - y][x] = model[x][y]

    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(16, 8))

    sns.heatmap(grid, square=True, ax=ax1)
    ax1.set_title(txfile + ' - fspl', fontsize=16)
    sns.heatmap(grid2, square=True, ax=ax2)
    ax2.set_title(txfile + ' - itwom', fontsize=16)
    f.savefig(txfile + '.png')


def clean_itwom(itwom, fspl):
    '''itwom has strange pathloss. eg. pathloss = 0 when distance between tx and rx is small (0 ~ 200m)
       Use fspl to replace the strange fspl
    Args:
        itwom -- np.1darray
        fslp  -- np.1darray
    '''
    if len(itwom) != len(fspl):
        print('itwom and fslp length not equal')
        return

    for i in range(len(itwom)):
        if itwom[i] <= 0.:
            itwom[i] = fspl[i]


def clean_all_itwom(itwom_all, fspl_all):
    '''
    Args:
        itwom -- np.2darray
        fspl  -- np.2darray
    Return:
        np.2darray
        np.2darray
    '''
    for itwom, fspl in zip(itwom_all, fspl_all):
        clean_itwom(itwom, fspl)


def interpolated_file(txfile):
    '''eg. output2/0001 --> interpolate2/0001
    Args:
        txfile -- str
    Return
        str
    '''
    pattern = r'.*\D(\d+)/(\d+)'
    p = re.compile(pattern)
    m = p.match(txfile)
    if m:
        print(m.groups())
        filename = 'interpolate' + m.group(1) + '/' + m.group(2)
    else:
        print(txfile, '-- interpolate filename no match')
        filename = 'interpolate0/error'
    return filename


def _interpolate_iwd(pre_inter, factor):
    '''Fix one transmitters, interpolate the sensors
    Args:
        pre_inter -- np.1darray -- pre interpolated, shape = gre_gl*gre_gl
        factor    -- int
    Return:
        np.1darray -- interpolated, shape = gre_gl*gre_gl*factor*factor
    '''
    pre_gl = int(math.sqrt(len(pre_inter)))                      # previous grid length (coarse grid)
    pre_inter = pre_inter.reshape((pre_gl, pre_gl))
    new_gl = pre_gl*factor                                       # new grid length (find grid)
    inter = np.zeros((new_gl, new_gl))
    for new_x in range(new_gl):
        for new_y in range(new_gl):
            if new_x%factor == 0 and new_y%factor == 0:                  # don't need to interpolate
                inter[new_x][new_y] = pre_inter[new_x//factor][new_y//factor]
            else:
                v_x, v_y = float(new_x)/factor, float(new_y)/factor  # virtual point in the coarse grid
                # pick some close points from the coarse grid
                points = []
                for pre_x in range(math.floor(v_x - 1), math.ceil(v_x + 1) + 1):
                    for pre_y in range(math.floor(v_y - 1), math.ceil(v_y + 1) + 1):
                        if pre_x >= 0 and pre_x < pre_gl and pre_y >= 0 and pre_y < pre_gl:
                            points.append((pre_x, pre_y, distance((v_x, v_y), (pre_x, pre_y))))
                points = sorted(points, key=lambda tup: tup[2])           # sort by distance
                threshold = min(9, len(points))
                weights = np.zeros(threshold)
                for i in range(threshold):
                    point = points[i]                             # inverse weighted distance
                    dist = distance((v_x, v_y), point)
                    weights[i] = 1./dist                          # normalize them
                weights /= np.sum(weights)
                idw = 0
                for i in range(threshold):
                    w = weights[i]
                    pre_rss = pre_inter[points[i][0]][points[i][1]]
                    idw += w*pre_rss
                inter[new_x][new_y] = idw
    return inter.reshape(new_gl*new_gl)


def interpolate_idw(data, factor=4):
    '''Interpolate by inverse distance weight, do two pass interpolation.
    Args:
        data -- np.2darray -- either fspl or itwom, shape = (h, h), where h = grid_len * grid_len
        factor -- int
    Return:
        np.2darray -- shape = (f^2*h, f^2*h), where f is factor
    '''
    # pass 1
    pre_hypo = len(data)
    pass_one_data = []
    for i in range(pre_hypo):
        data_inter  = _interpolate_iwd(data[i], factor)
        pass_one_data.append(data_inter)
    pass_one_data = np.array(pass_one_data)     # shape = (h, f^2*h)

    # pass 2
    pass_two_data = []
    pass_one_data = np.transpose(pass_one_data) # symmetry assumption
    inter_hypo = len(pass_one_data)
    for i in range(inter_hypo):
        data_inter = _interpolate_iwd(pass_one_data[i], factor)
        pass_two_data.append(data_inter)

    return np.array(pass_two_data)


def compute_error(pred, true):
    '''
    Args:
        pred -- np.2darray -- the interpolated data
        true -- np.2darray -- the true data
    Return:
        (float, float, float) -- mean absolute error, median absolute error, root mean square error
    '''
    pred = pred.flatten()
    true = true.flatten()
    return mean_absolute_error(true, pred), median_absolute_error(true, pred), math.sqrt(mean_squared_error(true, pred))


if __name__ == '__main__':
    # txfile = 'output2' + '/0002'

    # txfile = 'output2' + '/0002'
    # visualize_tx(txfile)
    # interpolate_idw(txfile, factor=2)
    # txfile = interpolated_file(txfile)
    # visualize_tx(txfile)
    # txfile = 'output3' + '/0004'
    # visualize_tx(txfile)
    
    DIR = 'output2'        # 25 hypothesis
    DIR2 = 'interpolate2'  # 100 hypothesis interpolated
    DIR3 = 'output3'       # 100 hypothesis
    
    fspl, itwom = get_all_data(DIR)
    clean_all_itwom(itwom, fspl)
    fspl_inter  = interpolate_idw(fspl, factor=2)
    itwom_inter = interpolate_idw(itwom, factor=2)

    fspl_true, itwom_true = get_all_data(DIR3)
    clean_all_itwom(itwom_true, fspl_true)
    mean, median, root = compute_error(fspl_inter, fspl_true)
    print('FSPL:\nmean absolute error     = {}\nmedian absolute error   = {}\nroot mean squared error = {}\n\n'.format(mean, median, root))
    
    mean, median, root = compute_error(itwom_inter, itwom_true)
    print('ITWOM:\nmean absolute error     = {}\nmedian absolute error   = {}\nroot mean squared error = {}'.format(mean, median, root))
    write_data(fspl_inter, itwom_inter, DIR2)
