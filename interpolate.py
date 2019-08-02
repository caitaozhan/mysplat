import re
import math
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, median_absolute_error, mean_squared_error
from utility import distance, customized_error, is_in_coarse_grid, hypo_in_coarse_grid
from utility import read_all_data, read_data, write_all_data, clean_all_itwom, clean_itwom


NEIGHBOR_NUM = 4


def visualize_tx(txfile):
    '''visualize the transmitter
    Args:
        txfile -- str -- eg. input/0000
    '''
    data = read_data(txfile)
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
    path = txfile + '.png'
    path = 'visualize/' + path.replace('/', '-')
    f.savefig(path)


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


def _interpolate_idw(pre_inter, factor):
    '''Fix one transmitters, interpolate the sensors
    Args:
        pre_inter -- np.1darray -- pre interpolated, shape = pre_gl*pre_gl
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
                v_x, v_y = float(new_x)/factor, float(new_y)/factor  # virtual point in the coarse grid / real point in the fine grid
                # pick some close points from the coarse grid
                points = []
                for pre_x in range(math.floor(v_x - 1), math.ceil(v_x + 1) + 1):
                    for pre_y in range(math.floor(v_y - 1), math.ceil(v_y + 1) + 1):
                        if pre_x >= 0 and pre_x < pre_gl and pre_y >= 0 and pre_y < pre_gl:
                            points.append((pre_x, pre_y, distance((v_x, v_y), (pre_x, pre_y))))
                points = sorted(points, key=lambda tup: tup[2])           # sort by distance
                threshold = min(NEIGHBOR_NUM, len(points))
                weights = np.zeros(threshold)
                for i in range(threshold):
                    point = points[i]
                    dist = distance((v_x, v_y), point)
                    weights[i] = (1./dist)**2                     # inverse weighted distance or inverse weighted square
                weights /= np.sum(weights)                        # normalize them
                idw = 0
                for i in range(threshold):
                    w = weights[i]
                    pre_rss = pre_inter[points[i][0]][points[i][1]]
                    idw += w*pre_rss
                inter[new_x][new_y] = idw
    return inter.reshape(new_gl*new_gl)


def _interpolate_idw_2(pre_inter, factor, tx, tx_pl):
    '''Fix one transmitters, interpolate the sensors
    Args:
        pre_inter -- np.1darray -- pre interpolated, shape = pre_gl*pre_gl
        tx         -- int -- the Tx that needs to interpolate Rx
        factor    -- int
    Return:
        np.1darray -- interpolated, shape = gre_gl*gre_gl*factor*factor
    '''
    pre_gl = int(math.sqrt(len(pre_inter)))                      # previous grid length (coarse grid)
    pre_inter = pre_inter.reshape((pre_gl, pre_gl))
    new_gl = pre_gl*factor                                       # new grid length (find grid)
    tx_x, tx_y = tx//new_gl, tx%new_gl
    inter = np.zeros((new_gl, new_gl))
    for new_x in range(new_gl):
        for new_y in range(new_gl):
            if new_x%factor == 0 and new_y%factor == 0:                  # don't need to interpolate
                inter[new_x][new_y] = pre_inter[new_x//factor][new_y//factor]
            else:
                v_x, v_y = float(new_x)/factor, float(new_y)/factor  # virtual point in the coarse grid / real point in the fine grid
                # pick some close points from the coarse grid
                points = []
                for pre_x in range(math.floor(v_x - 1), math.ceil(v_x + 1) + 1):
                    for pre_y in range(math.floor(v_y - 1), math.ceil(v_y + 1) + 1):
                        if pre_x >= 0 and pre_x < pre_gl and pre_y >= 0 and pre_y < pre_gl:
                            points.append((pre_x, pre_y, distance((v_x, v_y), (pre_x, pre_y))))  # the distance between the virtual Rx in the coarse grid and coarse Rx
                dist_to_tx_fine = distance((new_x, new_y), (tx_x, tx_y))       # distance in the fine grid
                if dist_to_tx_fine < 4:
                    dist_to_tx_coarse = distance((v_x, v_y), (tx_x/factor, tx_y/factor))
                    points.append((tx_x/factor, tx_y/factor, dist_to_tx_coarse)) # the additional Rx at the same location as the Tx
                points = sorted(points, key=lambda tup: tup[2])           # sort by distance
                threshold = min(NEIGHBOR_NUM, len(points))
                weights = np.zeros(threshold)
                for i in range(threshold):
                    point = points[i]
                    dist = distance((v_x, v_y), point)
                    dist = 0.01 if dist == 0.0 else dist
                    weights[i] = (1./dist)**2                             # inverse weighted distance or inverse weighted square
                weights /= np.sum(weights)                                # normalize them
                idw = 0
                for i in range(threshold):
                    w = weights[i]
                    try:
                        pre_rss = pre_inter[points[i][0]][points[i][1]]
                    except:
                        pre_rss = tx_pl                                   # the additional Rx at the same location as the Tx
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
        data_inter  = _interpolate_idw(data[i], factor)
        pass_one_data.append(data_inter)
    pass_one_data = np.array(pass_one_data)     # shape = (h, f^2*h)

    pass_one_data_copy = np.copy(pass_one_data)
    pass_one_data = np.transpose(pass_one_data) # symmetry assumption
    grid_len = int(math.sqrt(len(pass_one_data)))
    # pass 2
    tx_pl = pass_one_data_copy[0][0]
    pass_two_data = []
    inter_hypo = len(pass_one_data)
    for i in range(inter_hypo):
        if is_in_coarse_grid(i, grid_len, factor):
            coarse_hypo = hypo_in_coarse_grid(i, grid_len, factor)
            pass_two_data.append(pass_one_data_copy[coarse_hypo])
        else:
            data_inter = _interpolate_idw_2(pass_one_data[i], factor, i, tx_pl)
            pass_two_data.append(data_inter)
        
        # data_inter = _interpolate_iwd_2(pass_one_data[i], factor, i, tx_pl)
        # pass_two_data.append(data_inter)

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


def compute_weighted_error(pred, true):
    '''
    Args:
        pred -- np.2darray -- the interpolated data
        true -- np.2darray -- the true data
    Return:
        (float, float, float) -- mean absolute error, median absolute error, root mean square error
    '''
    return customized_error(pred, true)


def write_readme(directory):
    with open(directory + '/README.txt', 'w') as f:
        f.write('neighbor number = {}\n'.format(NEIGHBOR_NUM))
        print('neighbor number = {}\n'.format(NEIGHBOR_NUM))


def main1():
    DIR  = 'output2'       # 25 hypothesis
    DIR2 = 'interpolate2'  # 100 hypothesis interpolated
    DIR3 = 'output3'       # 100 hypothesis
    fspl, itwom = read_all_data(DIR)
    clean_all_itwom(itwom, fspl)
    fspl_inter  = interpolate_idw(fspl, factor=2)
    itwom_inter = interpolate_idw(itwom, factor=2)

    fspl_true, itwom_true = read_all_data(DIR3)
    clean_all_itwom(itwom_true, fspl_true)
    mean, median, root = compute_error(fspl_inter, fspl_true)
    print('FSPL:\nmean absolute error     = {}\nmedian absolute error   = {}\nroot mean squared error = {}\n\n'.format(mean, median, root))
    
    mean, median, root = compute_error(itwom_inter, itwom_true)
    print('ITWOM:\nmean absolute error     = {}\nmedian absolute error   = {}\nroot mean squared error = {}'.format(mean, median, root))
    write_all_data(fspl_inter, itwom_inter, DIR2)


def main2():
    DIR  = 'output7'       # 100 hypothesis
    DIR2 = 'interpolate7'  # 1600 hypothesis interpolated
    DIR3 = 'output8'       # 1600 hypothesis
    fspl, itwom = read_all_data(DIR)
    clean_all_itwom(itwom, fspl)
    fspl_inter  = interpolate_idw(fspl, factor=4)
    itwom_inter = interpolate_idw(itwom, factor=4)

    fspl_true, itwom_true = read_all_data(DIR3)
    clean_all_itwom(itwom_true, fspl_true)
    mean, median, root = compute_error(fspl_inter, fspl_true)
    print('FSPL:\nmean absolute error     = {}\nmedian absolute error   = {}\nroot mean squared error = {}\n\n'.format(mean, median, root))
    
    mean, median, root = compute_error(itwom_inter, itwom_true)
    print('ITWOM:\nmean absolute error     = {}\nmedian absolute error   = {}\nroot mean squared error = {}'.format(mean, median, root))
    write_all_data(fspl_inter, itwom_inter, DIR2)
    write_readme(DIR2)


def main3():
    DIR  = 'output9'       # 25 hypothesis
    DIR2 = 'interpolate9'  # 1600 hypothesis interpolated
    DIR3 = 'output8'       # 1600 hypothesis
    fspl, itwom = read_all_data(DIR)
    clean_all_itwom(itwom, fspl)
    fspl_inter  = interpolate_idw(fspl, factor=8)
    itwom_inter = interpolate_idw(itwom, factor=8)

    fspl_true, itwom_true = read_all_data(DIR3)
    clean_all_itwom(itwom_true, fspl_true)
    mean, median, root = compute_error(fspl_inter, fspl_true)
    print('FSPL:\nmean absolute error     = {}\nmedian absolute error   = {}\nroot mean squared error = {}\n\n'.format(mean, median, root))
    
    mean, median, root = compute_error(itwom_inter, itwom_true)
    print('ITWOM:\nmean absolute error     = {}\nmedian absolute error   = {}\nroot mean squared error = {}'.format(mean, median, root))
    write_all_data(fspl_inter, itwom_inter, DIR2)


def main4():
    DIR  = 'output10'       # 400 hypothesis
    DIR2 = 'interpolate10'  # 1600 hypothesis interpolated
    DIR3 = 'output8'        # 1600 hypothesis
    fspl, itwom = read_all_data(DIR)
    clean_all_itwom(itwom, fspl)
    fspl_inter  = interpolate_idw(fspl, factor=2)
    itwom_inter = interpolate_idw(itwom, factor=2)

    fspl_true, itwom_true = read_all_data(DIR3)
    clean_all_itwom(itwom_true, fspl_true)
    mean, median, root = compute_error(fspl_inter, fspl_true)
    print('FSPL:\nmean absolute error     = {}\nmedian absolute error   = {}\nroot mean squared error = {}\n\n'.format(mean, median, root))
    
    mean, median, root = compute_error(itwom_inter, itwom_true)
    print('ITWOM:\nmean absolute error     = {}\nmedian absolute error   = {}\nroot mean squared error = {}'.format(mean, median, root))
    write_all_data(fspl_inter, itwom_inter, DIR2)


def main5():
    DIR  = 'output11'        # 100 hypothesis
    DIR2 = 'interpolate11'   # 1600 hypothesis interpolated
    DIR3 = 'output12'        # 1600 hypothesis
    fspl, itwom = read_all_data(DIR)
    clean_all_itwom(itwom, fspl)
    fspl_inter  = interpolate_idw(fspl, factor=4)
    itwom_inter = interpolate_idw(itwom, factor=4)

    fspl_true, itwom_true = read_all_data(DIR3)
    clean_all_itwom(itwom_true, fspl_true)
    mean, median, root = compute_error(fspl_inter, fspl_true)
    print('FSPL:\nmean absolute error     = {}\nmedian absolute error   = {}\nroot mean squared error = {}\n\n'.format(mean, median, root))
    
    mean, median, root = compute_error(itwom_inter, itwom_true)
    print('ITWOM:\nmean absolute error     = {}\nmedian absolute error   = {}\nroot mean squared error = {}'.format(mean, median, root))
    write_all_data(fspl_inter, itwom_inter, DIR2)


def main6():
    DIR  = 'output13'       # 100 hypothesis
    DIR2 = 'interpolate13'  # 1600 hypothesis interpolated
    DIR3 = 'output14'        # 1600 hypothesis
    fspl, itwom = read_all_data(DIR)
    clean_all_itwom(itwom, fspl)
    fspl_inter  = interpolate_idw(fspl, factor=4)
    itwom_inter = interpolate_idw(itwom, factor=4)

    fspl_true, itwom_true = read_all_data(DIR3)
    clean_all_itwom(itwom_true, fspl_true)
    mean, median, root = compute_error(fspl_inter, fspl_true)
    print('FSPL:\nmean absolute error     = {}\nmedian absolute error   = {}\nroot mean squared error = {}\n'.format(mean, median, root))
    
    mean, median, root = compute_error(itwom_inter, itwom_true)
    print('ITWOM:\nmean absolute error     = {}\nmedian absolute error   = {}\nroot mean squared error = {}\n\n'.format(mean, median, root))
    write_all_data(fspl_inter, itwom_inter, DIR2)


def main7():
    '''Weighted Error
    '''
    # DIR  = 'output7'       # 100 hypothesis
    DIR2 = 'interpolate10'  # 1600 hypothesis interpolated
    DIR3 = 'output8'       # 1600 hypothesis
    _, itwom_inter = read_all_data(DIR2)
    fspl_true, itwom_true   = read_all_data(DIR3)
    clean_all_itwom(itwom_true, fspl_true)
    mean, median, std, coarse_mean, coarse_median, coarse_std, fine_mean, fine_median, fine_std = customized_error(itwom_inter, itwom_true)
    print('\nmean      = {:.3f}, median      = {:.3f}, std      = {:.3f}\ncoar mean = {:.3f}, coar median = {:.3f}, coar std = {:.3f}\nfine mean = {:.3f}, fine median = {:.3f}, fine std = {:.3f}'.format(\
             mean, median, std, coarse_mean, coarse_median, coarse_std, fine_mean, fine_median, fine_std))


if __name__ == '__main__':
    
    # main1()
    # main2()
    # main3()
    main4()
    # main5()
    # main6()
    main7()

    # tx_coarse = '0099'
    # tx_fine   = '0788'
    # visualize_tx('output7/' + tx_coarse)
    # visualize_tx('interpolate7/' + tx_fine)
    # visualize_tx('output8/' + tx_fine)
