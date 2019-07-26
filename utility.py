'''Utilities
'''
import math
import numpy as np


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
    '''Check whether a location in the fine grid is in the coarse grid
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


def customized_error(pred, true, dist_th=4):
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
    errors_fine = []     # errors of Tx -- Rx where Tx are only in the fine grid
    for i in range(size):
        in_coarse = is_in_coarse_grid(i, grid_len=40, factor=4)
        for j in range(size):
            error = pred[i][j] - true[i][j]
            tx = (i//grid_len, i%grid_len)
            rx = (j//grid_len, i%grid_len)
            dist  = distance(tx, rx)
            if dist <= dist_th:
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
