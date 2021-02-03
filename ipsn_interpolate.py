'''Interpolation for the IPSN 2020
'''

import os
import math
import glob
import numpy as np
from collections import defaultdict
from utility_ipsn import distance, indexconvert, read_data, clean_itwom, is_in_coarse_grid, hypo_in_coarse_grid, read_clean_itwom
from utility_ipsn import get_tx_index, read_all_data, compute_error, compute_weighted_error, clean_all_itwom
from utility_ipsn import write_all_itwom, read_all_itwom, customized_error
import argparse
from input_output import Input, Output

class IpsnInterpolate:
    DIR_FULL = 'output8'   # data for full training 1600 x 1600
    NEIGHBOR_NUM = 3
    IDW_EXPONENT = 1
    ILDW_DIST    = 0.5

    def __init__(self, dir_full='output8', full_grid_len=40, ton=False, origin_gridlen=10, factor=2):
        '''
        Args:
            ton -- bool -- the case for ton is different
        '''
        self.full_grid_len       = full_grid_len
        self.full_data           = np.array(0)      # np.2darray -- first dimension iterates the tx location (hypothesis), second dimension is the sensor value for that tx
        self.same_tx_rx_pathloss = 0
        self.range               = 0                # range for neighbor's of interpolation
        self.index               = []
        if not ton:
            self.full_data = self.init_full_data(dir_full)
        else:
            self.full_data = self.init_ton_full_data(dir_full, origin_gridlen, factor) # this is a fake full data with zero elements


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


    def init_ton_full_data(self, origin_data_dir, origin_gridlen, factor):
        origin_data_10 = self.read_hypothesis(origin_data_dir, origin_gridlen)
        fake_data_20   = self.impute_zeros(origin_data_10, origin_gridlen=origin_gridlen, factor=factor)
        self.same_tx_rx_pathloss = -28
        return fake_data_20


    def read_hypothesis(self, origin_data_dir, origin_gridlen):
        '''read the hypothesis file
        Args:
            origin_data_dir -- str -- the directory of the original full data. For the TON case, it is 10x10 and already interpolated once
            origin_gridlen  -- int -- the grid length of the origin grid
        '''
        num_h = origin_gridlen * origin_gridlen   # number of hypothesis (location)
        origin_rssi = np.zeros((num_h, num_h))
        hypo_file = os.path.join(origin_data_dir, 'hypothesis')
        with open(hypo_file, 'r') as f:
            for line in f:
                line = line.split(' ')
                t_x, t_y, s_x, s_y, rssi = int(line[0]), int(line[1]), int(line[2]), int(line[3]), float(line[4])
                t_idx = t_x * origin_gridlen + t_y
                s_idx = s_x * origin_gridlen + s_y
                origin_rssi[t_idx, s_idx] = rssi
        return origin_rssi


    def impute_zeros(self, origin_data, origin_gridlen, factor):
        '''Inpute zeros so that a smaller granularity data grows to a larger granularity data with zeros
        Args:
            origin_data    -- np.ndarray, n=2 -- (_, h) smaller granularity data, first dimension for tx, second dimension for sensors
            origin_gridlen -- int -- the grid length of the smaller granularity data
            factor         -- int -- the interpolation factor
        Return:
            np.ndarray, n=2, shape is (_, h*f^2)
        '''
        coarse_data = []
        full_gridlen = origin_gridlen * factor
        for tx_data in origin_data:
            tx_data = tx_data.reshape((origin_gridlen, origin_gridlen))
            tx_data_coarse = np.zeros((full_gridlen, full_gridlen))
            for i in range(origin_gridlen):
                for j in range(origin_gridlen):
                    tx_data_coarse[i * factor, j * factor] = tx_data[i, j]
            coarse_data.append(tx_data_coarse.reshape(full_gridlen * full_gridlen))
        return np.array(coarse_data)


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


    def interpolate_ton(self, coarse_gran, factor):
        '''The interpolation for TON (transaction on networking is different from the origin one for IPSN
           The difference is that in the IPSN, we have the real full data from simulation, then we zero out some values
           In TON, we don't have the real full data, i.e. we have a small 10x10 data and interpolate to a larger 20x20
           This requires a two pass interpolation that assumes symmetry
        '''
        print('TON interpolating...')
        pass_one_data = self.full_data   # the full data here is the fake full data, actually the coarse data
        self.range = int(math.sqrt(len(self.full_data[0])) / coarse_gran * 2)
        pass_one_data = self.ildw_ton1(pass_one_data, origin_gridlen=coarse_gran, factor=factor)
        pass_one_data_copy = np.copy(pass_one_data)
        pass_one_data = np.transpose(np.array(pass_one_data))
        pass_two_data = self.ildw_ton2(pass_one_data, pass_one_data_copy, full_gridlen=coarse_gran*factor, factor=factor)
        return np.array(pass_two_data)


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
        '''Inverse log distance weighting interpolation
        Args:
            sensor_data -- np.2darray -- the size is full x full, there are zero values in there that need to interpolate.
        Return:
            np.2darray -- the zeros are filled
        '''
        full_data = []
        for tx_1dindex in range(len(coarse_data)):
            print(tx_1dindex, end='  ', flush=True)
            tx_data = coarse_data[tx_1dindex]                   # note that the index in low granularity and high granularity is different
            if tx_data[tx_1dindex] == 0:
                tx_data[tx_1dindex] = self.same_tx_rx_pathloss  # ensure there is a sensor at the place of the tx
            full_data.append(self._ildw(tx_data, tx_1dindex))
        return full_data


    def ildw_ton1(self, coarse_data, origin_gridlen, factor):
        '''Inverse log distance weighting interpolation. first pass of the interpolation
        Args:
            coarse_data -- np.array -- shape is (100, 400)
        Return:
            np.2darray -- the zeros are filled
        '''
        print('interpolation pass one')
        full_gridlen = origin_gridlen * factor
        full_data = []
        for tx_1dindex in range(len(coarse_data)):
            print(tx_1dindex, end='  ', flush=True)
            tx_data = coarse_data[tx_1dindex]                   # note that the index in low granularity and high granularity is different
            tx_2dindex = (tx_1dindex // origin_gridlen, tx_1dindex % origin_gridlen)
            tx_2dindex_full = (tx_2dindex[0] * factor, tx_2dindex[1] * factor)
            tx_1dindex_full = tx_2dindex_full[0] * full_gridlen + tx_2dindex_full[1]
            full_data.append(self._ildw(tx_data, tx_1dindex_full))
        return full_data


    def ildw_ton2(self, pass_one_data, pass_one_data_copy, full_gridlen, factor):
        '''Inverse log distance weighting interpolation. second pass of the interpolation
        Args:
            pass_one_data      -- np.array -- shape is (400, 100)
            pass_one_data_copy -- np.array -- shape is (100, 400)
        Return:
            np.2darray -- the zeros are filled
        '''
        print('interpolation pass two')
        pass_two_data = []
        pass_one_data = self.impute_zeros(pass_one_data, full_gridlen//factor, factor)
        for i in range(len(pass_one_data)):
            print(i, end='  ', flush=True)
            if is_in_coarse_grid(i, full_gridlen, factor):
                coarse_hypo = hypo_in_coarse_grid(i, full_gridlen, factor)
                pass_two_data.append(pass_one_data_copy[coarse_hypo])
            else:
                tx_data = pass_one_data[i]
                tx_data[i] = self.same_tx_rx_pathloss
                pass_two_data.append(self._ildw(tx_data, i))
        return pass_two_data


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


def main2(localization_output_dir):
    '''interpolation for the TON submission
       make a fake full 20x20 data, let the coarse granularity be 10x10, then the rest should remain unchanged
    '''
    origin_data_10 = '10.6.inter-ildw'
    origin_gridlen = 10
    full_grid_len = 20
    factor = 2
    ipsnInter = IpsnInterpolate(origin_data_10, full_grid_len, ton=True, origin_gridlen=origin_gridlen, factor=factor)
    inter_data = ipsnInter.interpolate_ton(coarse_gran=origin_gridlen, factor=factor)
    ipsnInter.save_for_localization(inter_data, localization_output_dir)


if __name__ == '__main__':

    hint = 'python ipsn_interpolate.py -gl 40 -gran 28 -inter ildw idw -ildw_dist 0.9 -eo ipsn/error-gran -lo ipsn/inter-28'

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
        main1(error_output_file, localization_output_dir, granularity, inter_methods)
    elif grid_len == 20:
        main2(localization_output_dir)
    else:
        print('Do not support grid length = {}'.format(grid_len))
