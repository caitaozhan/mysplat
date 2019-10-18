'''Interpolation for the IPSN 2020
'''

import math
import glob
import numpy as np
from collections import defaultdict
from utility import distance, indexconvert, read_data, clean_itwom, is_in_coarse_grid, hypo_in_coarse_grid, read_clean_itwom
from utility import get_tx_index, read_all_data, compute_error, compute_weighted_error, clean_all_itwom
from utility import write_all_itwom, read_all_itwom, customized_error

from input_output import Input, Output

class IpsnInterpolate:
    DIR_FULL = 'output8'   # data for full training 1600 x 1600
    NEIGHBOR_NUM = 4
    IDW_EXPONENT = 1

    def __init__(self, dir_full='output8', full_grid_len=40):
        self.full_grid_len       = full_grid_len
        self.full_data           = np.array(0)      # np.2darray -- first dimension iterates the tx location (hypothesis), second dimension is the sensor value for that tx
        self.same_tx_rx_pathloss = 0
        self.range               = 0                # range for neighbor's of interpolation
        self.index               = []
        self.full_data = self.init_full_data(dir_full)


    def init_full_data(self, dir_full):
        '''Init the full training data
        '''
        print('the full training data is at {}'.format(dir_full))
        txfiles = sorted(glob.glob(dir_full + '/*'))
        self.full_data = []
        for txfile in txfiles:
            try:
                itwom = read_clean_itwom(txfile)
            except Exception as e:
                print(txfile, e)
            else:
                self.full_data.append(itwom)
        self.same_tx_rx_pathloss = self.full_data[0][0]  # 50 meters apart
        return np.array(self.full_data)


    def get_coarse_data(self, coarse_gran):
        '''Get the coarse grid (some data is removed)
        Args:
            coarse_gran -- int -- coarse granularity, typical options = [6, 8, 10, 12, 14, 16, 18, 20]
        Return:
            np.2darray  -- coarse data, dimension same as the full data, but many values are replaced by zero
        '''
        print('the coarse granularity is ', coarse_gran)
        self.range = int(self.full_grid_len / coarse_gran * 1.5)
        coarse_data = []
        step = self.full_grid_len/coarse_gran
        start = int(step/2)
        self.index = []
        for i in range(coarse_gran):
            self.index.append(start + int(step*i))
        for tx_data in self.full_data:
            tx_data = tx_data.reshape(self.full_grid_len, self.full_grid_len) # transform into 2d array
            tx_data_coarse = np.zeros((self.full_grid_len, self.full_grid_len))
            for x in self.index:
                for y in self.index:
                    tx_data_coarse[x, y] = tx_data[x, y]
            coarse_data.append(tx_data_coarse.reshape(self.full_grid_len * self.full_grid_len))
        return np.array(coarse_data)


    def interpolate(self, coarse_gran, interpolate_func):
        '''Interpolation
        Args:
            coarse_gran      -- int      -- coarse granularity
            interpolate_func -- function -- options: idw, ildw
        Return:
            np.2darray -- interpolated full data
        '''
        print('interpolating ...')
        coarse_data = self.get_coarse_data(coarse_gran)
        try:
            inter_data = interpolate_func(self, coarse_data)
        except Exception as e:
            print('Oops!', e)
        else:
            return inter_data


    def idw(self, coarse_data):
        '''Inverse distance weighting interpolation
        Args:
            coarse_data -- np.2darray -- first dimension iterates the tx location (hypothesis), second dimension is the sensor value for that tx
        Return:
            np.2darray -- the zeros are filled
        '''
        for tx_1dindex in range(len(coarse_data)):
            print(tx_1dindex, end='  ')
            tx_data = coarse_data[tx_1dindex]
            if tx_data[tx_1dindex] == 0:
                tx_data[tx_1dindex] = self.same_tx_rx_pathloss  # ensure there is a sensor at the place of the tx
            coarse_data[tx_1dindex] = self._idw(tx_data)


    def _idw(self, tx_data):
        '''Interpolate one Tx
        Args:
            tx_data -- np.1darray  -- contains zeros values
        Return:
            tx_data -- np.1darray  -- the zero values are filled up
        '''
        grid_pathloss = tx_data.reshape(self.full_grid_len, self.full_grid_len)
        # find the zero elements and impute them
        for x in range(grid_pathloss.shape[0]):
            for y in range(grid_pathloss.shape[1]):
                if grid_pathloss[x][y] != 0.0:
                    continue
                points = []
                d = self.range
                for i in range(x - d, x + d + 1):
                    for j in range(y - d, y + d + 1):
                        if i < 0 or i >= grid_pathloss.shape[0] or j < 0 or j >= grid_pathloss.shape[1]:
                            continue
                        if grid_pathloss[i][j] == 0.0:
                            continue
                        dist = distance((x, y), (i, j))
                        points.append( (i, j, dist) )
                points = sorted(points, key=lambda tup: tup[2])
                threshold = min(IpsnInterpolate.NEIGHBOR_NUM, len(points))
                weights = np.zeros(threshold)
                for i in range(threshold):
                    dist = points[i][2]
                    weights[i] = (1./dist)**IpsnInterpolate.IDW_EXPONENT
                weights /= np.sum(weights)
                idw_pathloss = 0
                for i in range(threshold):
                    w = weights[i]
                    rss = grid_pathloss[points[i][0]][points[i][1]]
                    idw_pathloss += w * rss
                grid_pathloss[x][y] = idw_pathloss
        return grid_pathloss.reshape(self.full_grid_len * self.full_grid_len)


    @staticmethod
    def ildw(sensor_data):
        '''Inverse distance weighting interpolation
        Args:
            sensor_data -- np.2darray -- the size is full x full, there are zero values in there that need to interpolate.
        Return:
            np.2darray -- the zeros are filled
        '''
        pass


    @staticmethod
    def _ildw(tx_data):
        '''Interpolate one Tx
        Args:
            tx_data -- np.1darray  -- contains zeros values
        Return:
            tx_data -- np.1darray  -- the zero values are filled up
        '''
        return np.array(0)


    def compute_errors(self, inter_data, dist_close, dist_far):
        '''Compute the interpolation errors
        Args:
            inter_data  -- np.2darray
            dist_close  -- int
            dist_far    -- int -- distance threshold between tx and rx
            coarse_gran -- int
        '''
        grid_is_inter = np.zeros((self.full_grid_len, self.full_grid_len))  # the places for sensors in coarse grid, no interpolation happening
        for x in self.index:
            for y in self.index:
                grid_is_inter[x, y] = 1.
        size = len(inter_data)
        errors_all   = []    # errors for all tx-rx
        errors_close = []    # errors for tx-rx with small distance
        errors_far   = []    # errors for tx-rx with large distance
        for i in range(size):
            tx = (i//self.full_grid_len, i%self.full_grid_len)
            for j in range(size):
                rx = (j//self.full_grid_len, j%self.full_grid_len)
                if grid_is_inter[rx[0]][rx[1]] == 1.:
                    continue
                error = inter_data[i][j] - self.full_data[i][j]
                dist = distance(tx, rx)
                if dist < dist_close:
                    errors_close.append(error)
                elif dist > dist_far:
                    errors_far.append(error)
                errors_all.append(error)
        mean_error = np.mean(errors_all)
        mean_absolute_error = np.mean(np.absolute(errors_all))
        mean_error_close = np.mean(errors_close)
        mean_absolute_error_close = np.mean(errors_close)
        mean_error_far = np.mean(errors_far)
        mean_absolute_error_far = np.mean(np.absolute(errors_far))
        std = np.std(np.absolute(errors_all))
        std_close = np.std(np.absolute(errors_close))
        std_far = np.std(np.absolute(errors_far))

        return Output(None, mean_error, mean_absolute_error, mean_error_close, mean_absolute_error_close, \
                      mean_error_far, mean_absolute_error_far, std, std_close, std_far)


def main0():
    #step 0: arguments
    dir_full = 'output10'
    full_grid_len = 20
    coarse_gran = 6
    inter_methods = ['idw']
    interpolate_func=IpsnInterpolate.idw
    dist_close = 8
    dist_far = 32
    myinput = Input(dir_full, full_grid_len, coarse_gran, inter_methods, dist_close, dist_far)
    # step 1: interpolate
    ipsnInter = IpsnInterpolate(dir_full, full_grid_len)
    inter_data = ipsnInter.interpolate(coarse_gran, interpolate_func)
    # step 2: compute error
    myoutput = ipsnInter.compute_errors(inter_data, dist_close, dist_far)
    myoutput.methods = 'idw'
    # print('ITWOM:\nmean absolute error     = {}\nmedian absolute error   = {}\nroot mean squared error = {}'.format(mean, median, root))
    # step 3: save it to file


def main1():
    #step 0: arguments
    dir_full = 'output8'
    full_grid_len = 40
    coarse_gran = 12
    interpolate_func=IpsnInterpolate.idw
    # step 1: interpolate
    ipsnInter = IpsnInterpolate(dir_full, full_grid_len)
    inter_data = ipsnInter.interpolate(coarse_gran, interpolate_func)
    # step 2: compute error
    dist_th = 5
    mean, median, root = customized_error(inter_data, dist_th, coarse_gran)
    # print('ITWOM:\nmean absolute error     = {}\nmedian absolute error   = {}\nroot mean squared error = {}'.format(mean, median, root))
    # step 3: save it to file

if __name__ == '__main__':
    main0()
