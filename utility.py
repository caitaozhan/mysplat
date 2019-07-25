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


def weighted_error(pred, true):
    '''
    Args:
        pred -- np.2darray -- the interpolated data
        true -- np.2darray -- the true data
    Return:
        (float, float, float) -- mean absolute error, median absolute error, root mean square error
    '''
    size = len(pred)
    grid_len = int(math.sqrt(len(pred)))
    errors = []
    for i in range(size):
        for j in range(size):
            error = pred[i][j] - true[i][j]
            tx = (i//grid_len, i%grid_len)
            rx = (j//grid_len, i%grid_len)
            dist  = distance(tx, rx)
            if dist <= 5:
                errors.append(abs(error))
            # errors.append(abs(error))
    errors = np.array(errors)
    print('length of errors', len(errors))
    half = int(len(errors)/2)
    return errors.mean(), sorted(errors)[half]
