import numpy as np
import re
import math
import os
import matplotlib.pyplot as plt
import seaborn as sns
from utility import distance


def get_data(txfile):
    '''get the pathloss data from files
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
        itwom -- np.2darray
        fslp  -- np.2darray
    '''
    if len(itwom) != len(fspl):
        print('itwom and fslp length not equal')
        return

    for i in range(len(itwom)):
        if itwom[i] <= 0.:
            itwom[i] = fspl[i]


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
    '''
    Args:
        pre_inter -- np.1darray -- pre interpolated
        factor    -- int
    Return:
        np.1darray
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


def interpolate_idw(txfile, factor=4):
    '''Interpolate by inverse distance weight
    Args:
        txfile -- str -- eg. output2/0000
        factor -- int
    
    '''
    data = get_data(txfile)
    inter_file = interpolated_file(txfile)
    print(inter_file)
    folder, _ = inter_file.split('/')
    if not os.path.exists(folder):
        os.mkdir(folder)

    fspl  = data[0]
    itwom = data[1]
    clean_itwom(itwom, fspl)
    fspl_inter  = _interpolate_iwd(fspl, factor)
    itwom_inter = _interpolate_iwd(itwom, factor)

    f = open(inter_file, 'w')
    f.write(','.join(map(lambda x: str(x), fspl_inter)))
    f.write('\n')
    f.write(','.join(map(lambda x: str(x), itwom_inter)))
    f.close()


if __name__ == '__main__':
    # txfile = 'output2' + '/0002'


    txfile = 'output2' + '/0002'
    visualize_tx(txfile)
    interpolate_idw(txfile, factor=2)
    txfile = interpolated_file(txfile)
    visualize_tx(txfile)
    txfile = 'output3' + '/0004'
    visualize_tx(txfile)
