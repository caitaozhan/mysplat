'''Utilities
'''
import math
import numpy as np
import os
import glob
import re
from sklearn.metrics import mean_absolute_error, median_absolute_error, mean_squared_error


def distance(point1, point2):
    '''
    Args:
        point1 -- (float, float)
        point2 -- (float, float)
    Return:
        float
    '''
    return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)


def is_in_coarse_grid(hypo, grid_len, factor):
    '''Check whether a location in the fine grid is in the coarse grid
    Args:
        hypo -- str
        grid_len -- int -- the grid length of the fine grid
        factor -- int -- the ratio of fine grid grid length and coarse grid grid length
    Return:
        bool
    '''
    x = hypo//grid_len
    y = hypo%grid_len
    if x%factor == 0 and y%factor == 0:
        return True
    return False


def hypo_in_coarse_grid(hypo, grid_len, factor):
    '''the hypothesis of the full fine grid is ensured to have its counterpart in the coarse grid
       so find out the one in the coarse grid
    Args:
        hypo -- str
        grid_len -- int -- the grid length of the fine grid
        factor -- int -- the ratio of fine grid grid length and coarse grid grid length
    Return:
        int -- the hypothesis in the coarse grid
    '''
    x = hypo//grid_len
    y = hypo%grid_len
    x = int(x/factor)
    y = int(y/factor)
    coarse_grid_len = int(grid_len/factor)
    return x*coarse_grid_len + y


def customized_error(pred, true, dist_th=4, factor=4):
    '''
    Args:
        pred -- np.2darray -- the interpolated data
        true -- np.2darray -- the true data
        dist_th -- int -- distance threshold
    Return:
        (float, float, float) -- mean absolute error, median absolute error, root mean square error
    '''
    size = len(pred)
    grid_len = int(math.sqrt(len(pred)))
    errors = []
    errors_coarse = []   # errors of Tx -- Rx where Tx exists in the coarse grid
    errors_fine = [0]     # errors of Tx -- Rx where Tx are only in the fine grid
    for i in range(size):
        tx = (i//grid_len, i%grid_len)
        in_coarse = is_in_coarse_grid(i, grid_len=40, factor=factor)
        for j in range(size):
            error = pred[i][j] - true[i][j]
            rx = (j//grid_len, j%grid_len)
            dist  = distance(tx, rx)
            if dist < dist_th:
                errors.append(abs(error))
                if in_coarse:
                    errors_coarse.append(abs(error))
                else:
                    errors_fine.append(abs(error))
    errors = np.array(errors)
    errors_coarse = np.array(errors_coarse)
    errors_fine = np.array(errors_fine)
    return errors.mean(), sorted(errors)[int(len(errors)/2)], errors.std(), \
           errors_coarse.mean(), sorted(errors_coarse)[int(len(errors_coarse)/2)], errors_coarse.std(), \
           errors_fine.mean(), sorted(errors_fine)[int(len(errors_fine)/2)], errors_fine.std()

def get_tx_index(txfile):
    '''get the tx 1d index from the filename
    Args:
        txfile -- str -- eg. 'output7/0001'
    Return:
        int -- eg. 1
    '''
    int4 = r'(\d{4,4})'
    pattern = r'.*\D{}'.format(int4)
    p = re.compile(pattern)
    m = p.match(txfile)
    if m:
        return(int(m.group(1)))
    else:
        print(txfile, 'no match')


def read_data(txfile):
    '''get the pathloss data from files
    Args:
        file -- str
    Return:
        np.2darray
    '''
    return np.loadtxt(txfile, delimiter=',')


def read_all_data(directory):
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


def read_all_itwom(directory):
    '''get all the pathloss of all the tx from all files inside a directory
    Args:
        directory -- str -- folder that has one transmitter to all sensor pathloss data.
    Return:
        np.2darray -- for fspl model,  shape = (h, h), where h is the number of hypothesis, eg. grid_len x grid_len
        np.2darray -- for itwom model, shape = (h, h), where h is the number of hypothesis, eg. grid_len x grid_len
    '''
    files = sorted(glob.glob(directory + '/*'))
    itwom = []
    for f in files:
        try:
            data = np.loadtxt(f, delimiter=',')
            itwom.append(data[1])
        except:
            print(f, 'cannot np.loadtxt')
    return np.array(itwom)


def write_data(fspl, itwom, filename):
    with open(filename, 'w') as f:
        f.write(','.join(map(lambda x: str(x), fspl)))
        f.write('\n')
        f.write(','.join(map(lambda x: str(x), itwom)))


def write_itwom(itwom, filename):
    with open(filename, 'w') as f:
        f.write(','.join(map(lambda x: str(x), itwom)))


def write_all_data(fspl, itwom, directory):
    '''write the pathloss of all tx to different files of a folder
    Args:
        fspl  -- np.2darray
        itwom -- np.2darray
        directory  -- str
    '''
    if not os.path.exists(directory):
        os.mkdir(directory)

    num_hypo = len(fspl)
    if len(itwom) != num_hypo:
        print('fspl and itwom length not equal!')
        return
    for i in range(num_hypo):
        filename = directory + '/{:04}'.format(i)
        write_data(fspl[i], itwom[i], filename)


def write_all_itwom(itwom, directory):
    '''write the pathloss of all tx to different files of a folder
    Args:
        itwom -- np.2darray
        directory  -- str
    '''
    if not os.path.exists(directory):
        os.mkdir(directory)

    num_hypo = len(itwom)
    for i in range(num_hypo):
        filename = directory + '/{:04}'.format(i)
        write_itwom(itwom[i], filename)


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

def indexconvert(index, grid_len):
    '''Convert index from 2d (1d) into 1d (2d)
    Args:
        index -- (int, int) or int -- 2d index or 1d index
    Return:
        int or (int, int) -- 1d index or 2d index
    '''
    if isinstance(index, tuple) and len(index) == 2:
        return index[0]*grid_len + index[1]
    if isinstance(index, int):
        return (index//grid_len, index%grid_len)
    raise Exception('convert failed. index = {}, grid length = {}'.format(index, grid_len))


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


def read_clean_itwom(txfile):
    '''Read all pathloss
    '''
    fspl, itwom = read_data(txfile)
    clean_itwom(itwom, fspl)
    return itwom
