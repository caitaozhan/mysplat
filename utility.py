'''Utilities
'''
import math

def distance(point1, point2):
    '''
    Args:
        point1 -- (float, float)
        point2 -- (float, float)
    Return:
        float
    '''
    return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)
