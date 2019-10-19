'''Interpolation for the IPSN 2020
'''

import os
import math
import glob
import numpy as np
from collections import defaultdict
from utility import distance, indexconvert, read_data, clean_itwom, is_in_coarse_grid, hypo_in_coarse_grid, read_clean_itwom
from utility import get_tx_index, read_all_data, compute_error, compute_weighted_error, clean_all_itwom
from utility import write_all_itwom, read_all_itwom, customized_error
import argparse
from input_output import Input, Output

class IpsnInterpolate:
    DIR_FULL = 'output8'   # data for full training 1600 x 1600
    NEIGHBOR_NUM = 3
    IDW_EXPONENT = 1
    ILDW_DIST    = 0.5

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
            print()
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
        full_data = []
        for tx_1dindex in range(len(coarse_data)):
            print(tx_1dindex, end=' ')
            tx_data = coarse_data[tx_1dindex]
            if tx_data[tx_1dindex] == 0:
                tx_data[tx_1dindex] = self.same_tx_rx_pathloss  # ensure there is a sensor at the place of the tx
            full_data.append(self._idw(tx_data))
        return full_data


    def _idw(self, tx_data):
        '''Interpolate one Tx
        Args:
            tx_data -- np.1darray  -- contains zeros values
        Return:
            tx_data -- np.1darray  -- the zero values are filled up
        '''
        grid_pathloss = tx_data.reshape(self.full_grid_len, self.full_grid_len)
        grid_interpolate = np.copy(grid_pathloss)
        # find the zero elements and impute them
        for x in range(grid_pathloss.shape[0]):
            for y in range(grid_pathloss.shape[1]):
                if grid_pathloss[x][y] != 0.0:  # skip the ones that don't need to interpolate
                    continue
                points = []
                d = self.range
                for i in range(x - d, x + d + 1):
                    for j in range(y - d, y + d + 1):
                        if i < 0 or i >= grid_pathloss.shape[0] or j < 0 or j >= grid_pathloss.shape[1]:
                            continue
                        if grid_pathloss[i][j] == 0.0:  # only use the known point to interpolate
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
                grid_interpolate[x][y] = idw_pathloss
        return grid_interpolate.reshape(self.full_grid_len * self.full_grid_len)


    def ildw(self, coarse_data):
        '''Inverse distance weighting interpolation
        Args:
            sensor_data -- np.2darray -- the size is full x full, there are zero values in there that need to interpolate.
        Return:
            np.2darray -- the zeros are filled
        '''
        full_data = []
        for tx_1dindex in range(len(coarse_data)):
            print(tx_1dindex, end='  ', flush=True)
            tx_data = coarse_data[tx_1dindex]
            if tx_data[tx_1dindex] == 0:
                tx_data[tx_1dindex] = self.same_tx_rx_pathloss  # ensure there is a sensor at the place of the tx
            full_data.append(self._ildw(tx_data, tx_1dindex))
        return full_data


    def _ildw(self, tx_data, tx_1dindex):
        '''Interpolate one Tx
        Args:
            tx_data -- np.1darray  -- contains zeros values
        Return:
            tx_data -- np.1darray  -- the zero values are filled up
        '''
        tx = (tx_1dindex//self.full_grid_len, tx_1dindex%self.full_grid_len)   # ildw requires the location of the transmitter
        grid_pathloss = tx_data.reshape(self.full_grid_len, self.full_grid_len)
        grid_interpolate = np.copy(grid_pathloss)
        # find the zero elements and impute them
        for x in range(grid_pathloss.shape[0]):
            for y in range(grid_pathloss.shape[1]):
                if grid_pathloss[x][y] != 0.0:  # skip the ones that don't need to interpolate
                    continue
                points = []
                d = self.range
                for i in range(x - d, x + d + 1):
                    for j in range(y - d, y + d + 1):
                        if i < 0 or i >= grid_pathloss.shape[0] or j < 0 or j >= grid_pathloss.shape[1]:
                            continue
                        if grid_pathloss[i][j] == 0.0:  # only use the known point to interpolate
                            continue
                        dist = distance((x, y), (i, j))
                        points.append( (i, j, dist) )
                points = sorted(points, key=lambda tup: tup[2])
                threshold = min(IpsnInterpolate.NEIGHBOR_NUM, len(points))
                weights = np.zeros(threshold)
                dist_to_tx = np.zeros(threshold)
                for i in range(threshold):
                    nei = (points[i][0], points[i][1])
                    dist_to_tx[i] = distance(tx, nei)
                rx0 = (x, y)
                weights = IpsnInterpolate.get_log10_weight(dist_to_tx, tx, rx0)
                weights /= np.sum(weights)
                idw_pathloss = 0
                for i in range(threshold):
                    w = weights[i]
                    rss = grid_pathloss[points[i][0]][points[i][1]]
                    idw_pathloss += w * rss
                grid_interpolate[x][y] = idw_pathloss
        return grid_interpolate.reshape(self.full_grid_len * self.full_grid_len)


    @staticmethod
    def get_log10_weight(to_tx_dists, tx, rx0):
        '''get the weights from project_dists
        Args:
            project_dists  -- np.1darray
            tx -- (float, float)
            rx0 -- (float, float)  -- location to be interpolated
        Return:
            np.1darray
        '''
        for i, dist in enumerate(to_tx_dists):
            if dist == 0:
                to_tx_dists[i] = IpsnInterpolate.ILDW_DIST  # this value can tweak
        log10_dist_to_tx = np.log10(to_tx_dists)
        reference_dist = np.log10(distance(tx, rx0))
        log10_dist_to_rx0 = log10_dist_to_tx - reference_dist
        log10_dist_to_rx0 = np.absolute(log10_dist_to_rx0)
        weight = np.zeros(len(log10_dist_to_rx0))
        for i, dist in enumerate(log10_dist_to_rx0):
            if dist > 0:
                weight[i] = (1./dist)
        maxx = max(weight)
        for i, w in enumerate(weight):
            if w == 0:
                weight[i] = 2*maxx if maxx > 0 else 1
        return weight


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
                if grid_is_inter[rx[0]][rx[1]] == 1. or i == j:
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
        mean_absolute_error_close = np.mean(np.absolute(errors_close))
        mean_error_far = np.mean(errors_far)
        mean_absolute_error_far = np.mean(np.absolute(errors_far))
        std = np.std(np.absolute(errors_all))
        std_close = np.std(np.absolute(errors_close))
        std_far = np.std(np.absolute(errors_far))

        return Output(None, mean_error, mean_absolute_error, mean_error_close, mean_absolute_error_close, \
                      mean_error_far, mean_absolute_error_far, std, std_close, std_far)


    def save_for_localization(self, inter_data, output_dir):
        '''Save all the path loss to output_dir
        '''
        if os.path.exists(output_dir) is False:
            os.makedirs(output_dir)

        sen_num = self.full_grid_len*self.full_grid_len
        all_sensors = list(range(sen_num))   # sensors at all location
        # step 1: covariance matrix
        with open(output_dir + '/cov', 'w') as f:
            cov = np.zeros((sen_num, sen_num))
            for i in range(sen_num):
                for j in range(sen_num):
                    if i == j:
                        cov[i, j] = 1        # assume the std is 1
                    f.write('{} '.format(cov[i, j]))
                f.write('\n')

        # step 2: sensors
        with open(output_dir + '/sensors', 'w') as f:
            for sen_1dindex in all_sensors:
                sen_x = sen_1dindex//self.full_grid_len
                sen_y = sen_1dindex%self.full_grid_len
                f.write('{} {} {} {}\n'.format(sen_x, sen_y, 1, 1))  # uniform cost

        # step 3: hypothesis
        with open(output_dir + '/hypothesis', 'w') as f:
            for tx_1dindex in range(len(inter_data)):
                t_x = tx_1dindex // self.full_grid_len
                t_y = tx_1dindex % self.full_grid_len
                for sen_1dindex in all_sensors:
                    s_x = sen_1dindex//self.full_grid_len
                    s_y = sen_1dindex%self.full_grid_len
                    pathloss = inter_data[tx_1dindex][sen_1dindex]
                    f.write('{} {} {} {} {:.2f} {}\n'.format(t_x, t_y, s_x, s_y, pathloss, 1))



def main0(error_output_file, localization_output_dir, granularity, inter_methods):
    '''this is for 10 x 10 small grid, debugging usage
    '''
    f_error = open(error_output_file, 'a')
    
    #step 0: arguments
    dir_full = 'output7'
    full_grid_len = 10
    coarse_gran = 4
    inter_methods = inter_methods
    dist_close = 3
    dist_far = 7
    myinput = Input(dir_full, full_grid_len, coarse_gran, inter_methods, dist_close, dist_far)
    for method in inter_methods:
        # step 1: interpolate
        if method == 'idw':
            interpolate_func = IpsnInterpolate.idw
        elif method == 'ildw':
            interpolate_func = IpsnInterpolate.ildw
        else:
            raise Exception('method not valid')
        ipsnInter = IpsnInterpolate(dir_full, full_grid_len)
        inter_data = ipsnInter.interpolate(coarse_gran, interpolate_func)
        # step 2: compute error
        myoutput = ipsnInter.compute_errors(inter_data, dist_close, dist_far)
        myoutput.method = method
        f_error.write(myinput.log())
        f_error.write(myoutput.log())
        f_error.write('\n')
        # step 3: save localization input to file
        if method == 'ildw':
            ipsnInter.save_for_localization(inter_data, localization_output_dir)
    f_error.close()


def main1(error_output_file, localization_output_dir, granularity, inter_methods):
    '''this is for 40 x 40 large grid, evaluation usage
    '''
    f_error = open(error_output_file, 'a')

    #step 0: arguments
    dir_full = 'output8'
    full_grid_len = 40
    coarse_gran = granularity
    inter_methods = inter_methods
    for method in inter_methods:
        # step 1: interpolate
        if method == 'idw':
            interpolate_func = IpsnInterpolate.idw
            ildw_dist = -1
        elif method == 'ildw':
            interpolate_func = IpsnInterpolate.ildw
            ildw_dist = IpsnInterpolate.ILDW_DIST
        else:
            raise Exception('method not valid')
        ipsnInter = IpsnInterpolate(dir_full, full_grid_len)
        inter_data = ipsnInter.interpolate(coarse_gran, interpolate_func)
        # step 2: compute error
        dist_close = [8]
        dist_far   = [32]
        for d_c, d_f in zip(dist_close, dist_far):
            myinput  = Input(dir_full, full_grid_len, coarse_gran, inter_methods, ildw_dist, d_c, d_f)
            myoutput = ipsnInter.compute_errors(inter_data, d_c, d_f)
            myoutput.method = method
            f_error.write(myinput.log())
            f_error.write(myoutput.log())
            f_error.write('\n')
        # step 3: save localization input to file
        if method == 'ildw':
            ipsnInter.save_for_localization(inter_data, localization_output_dir)
    f_error.close()



if __name__ == '__main__':

    hint = 'python ipsn_interpolate.py -gran 12 -inter ildw'

    parser = argparse.ArgumentParser(description='Interpolation for IPSN | Hint: ' + hint)
    parser.add_argument('-eo', '--error_output_file', type=str, nargs=1, default=['ipsn/error'])
    parser.add_argument('-lo', '--localization_output_directory', type=str, nargs=1, default=['ipsn/12'])
    parser.add_argument('-gran', '--granularity', type=int, nargs=1, default=[12], help='Granularity for the sensors')
    parser.add_argument('-inter', '--inter_methods', type=str, nargs='+', default=['idw'], help='interpolation method')
    parser.add_argument('-ildw_dist', '--ildw_distance', type=float, nargs=1, default=[IpsnInterpolate.ILDW_DIST], help='a parameter for ildw')
    parser.add_argument('-gl', '--grid_len', type=int, nargs=1, default=[10], help='grid length')
    args = parser.parse_args()

    error_output_file       = args.error_output_file[0]
    localization_output_dir = args.localization_output_directory[0]
    granularity             = args.granularity[0]
    inter_methods           = args.inter_methods
    grid_len                = args.grid_len[0]
    IpsnInterpolate.ILDW_DIST = args.ildw_distance[0]

    if grid_len == 10:
        main0(error_output_file, localization_output_dir, granularity, inter_methods)
    elif grid_len == 40:
        main1(error_output_file, localization_output_dir, granularity,inter_methods)
    else:
        print('Do not support grid length = {}'.format(grid_len))
