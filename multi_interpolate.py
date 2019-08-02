'''
multi-granularity interpolation
the different granularity is in Rx, where we have a higher density of sensors close to Tx
'''

import math
import glob
import numpy as np
from collections import defaultdict
from utility import distance, indexconvert, read_data, clean_itwom, is_in_coarse_grid, hypo_in_coarse_grid
from utility import get_tx_index, read_all_data, compute_error, compute_weighted_error, clean_all_itwom
from utility import write_all_itwom, read_all_itwom, customized_error


class Global:   # Global variables
    AREA_LEN  = 4000    # the area is 4000m x 4000m
    HIGH      = 0              # HIGH granularity is   0 ~ 400     (meters)
    MED       = 0            # MEDium granularity is 400 ~ 1200
    # LOW  = 1200           # LOW granularity is    > 1200
    GRAN_LEVEL  = [HIGH, MED]



class TxMultiGran:
    '''Encapsulate one Tx. Handles multiple granularity of RX of one Tx
    '''
    TX_GRID_LEN = 10      # the granularity of the Tx during training
    RX_CELL_LEN = {}      # grid_len --> cell length

    def __init__(self, x, y, debug=False):
        self.x = x              # the location of the Tx at Global.TX_GRID_LEn
        self.y = y
        self.debug = debug
        self.granularity_data = defaultdict(np.array)
        self.sensor_data = np.array(0)        


    def add_sensor_data(self, grid_len, pathloss):
        '''Add RSS data at a granularity
        Args:
            grid_len -- int -- this represents the granularity
            pathloss -- np.1darray -- the itwom data
        '''
        self.granularity_data[grid_len] = pathloss
        TxMultiGran.RX_CELL_LEN[grid_len] = int(Global.AREA_LEN/grid_len)


    def get_gran_level(self, gran_level, dist):
        '''The further to the Tx, the lower the granularity level
        Args:
            gran_level -- list<int> -- granularity levels in meters
            dist -- float
        Return:
            int -- a granularity level
        '''
        size = len(gran_level)
        for i in range(size-1):
            if dist >= gran_level[i] and dist < gran_level[i+1]:
                return gran_level[i]
        return gran_level[size-1]


    def combine_sensor_data(self):
        '''Combine sensor data from multiply granularity. each granularity has full data in 1 dimension (from the SPLAT!)
        '''
        grid_lens = sorted(TxMultiGran.RX_CELL_LEN.keys())  # [5, 10, 20]
        if len(grid_lens) != len(Global.GRAN_LEVEL):
            print((self.x, self.y), 'length of grid_lens and granularity level doesn\'t match !')
            return

        # map granularity level to grid length         # {1200 --> 5, 400 --> 10, 0 --> 20}
        gran_level2grid_len = {}
        for gran_level, grid_len in zip(reversed(Global.GRAN_LEVEL), grid_lens):
            gran_level2grid_len[gran_level] = grid_len

        # map grid length to ratio. the ratio of finest_rx_grid_len to different grid_len
        finest_rx_grid_len = grid_lens[-1]
        grid_len2ratio = {}
        for grid_len in grid_lens:
            grid_len2ratio[grid_len] = int(finest_rx_grid_len/grid_len)         # {5 --> 4, 10 --> 2, 20 --> 1}
        
        self.sensor_data = np.zeros((finest_rx_grid_len, finest_rx_grid_len))   # fill up this array
        
        t_x = self.x * grid_len2ratio[TxMultiGran.TX_GRID_LEN]
        t_y = self.y * grid_len2ratio[TxMultiGran.TX_GRID_LEN]
        for x in range(finest_rx_grid_len):
            for y in range(finest_rx_grid_len):
                dist = distance((x, y), (t_x, t_y)) * TxMultiGran.RX_CELL_LEN[finest_rx_grid_len]  # distance in meters
                gran_level = self.get_gran_level(Global.GRAN_LEVEL, dist)
                grid_len   = gran_level2grid_len[gran_level]
                ratio = grid_len2ratio[grid_len]
                if x%ratio == 0 and y%ratio == 0:
                    gran_data = self.granularity_data[grid_len]
                    gran_grid_len = int(math.sqrt(len(gran_data)))
                    index = x//ratio * gran_grid_len + y//ratio
                    self.sensor_data[x][y] = gran_data[index]
        if self.debug:
            num_wireless_link = np.count_nonzero(self.sensor_data)
            print('Tx = ({}, {})'.format(self.x, self.y), '; number of wireless links to sensors', num_wireless_link)


class MultiIntepolate:
    '''Encapsulate the interpolation method
    '''
    NEIGHBOUR_NUM = 4   # number of neighboor for averaging

    @staticmethod
    def idw_interpolate(txs, target_grid_len):
        '''Interpolation through inverse distance weight (IDW)
        Args:
            txs -- list<TxMultiGran>
            target_grid_len -- int -- interpolate both Tx and Rx into a grid of (target_grid_len, target_grid_len)
        '''
        # pass 1
        pass_one_data = []
        for tx in txs:
            tx_inter = MultiIntepolate._idw_interpolate(tx.sensor_data, target_grid_len)
            pass_one_data.append(tx_inter)
        pass_one_data = np.array(pass_one_data)      # tx_hypo * target_grid_len**2
        
        # pass 2
        tx_pl = pass_one_data[0][0]                  # assume tx_pl is the same everywhere
        coarse_grid_len = int(math.sqrt(len(pass_one_data)))
        pass_one_data_copy = np.copy(pass_one_data)
        pass_one_data = np.transpose(pass_one_data)  # target_grid_len**2 * tx_hypo
        pass_two_data = []
        inter_hypo = len(pass_one_data)
        for i in range(inter_hypo):
            if is_in_coarse_grid(i, target_grid_len, int(target_grid_len/coarse_grid_len)):
                coarse_hypo = hypo_in_coarse_grid(i, target_grid_len, int(target_grid_len/coarse_grid_len))
                pass_two_data.append(pass_one_data_copy[coarse_hypo])
            else:
                tx_inter = MultiIntepolate._idw_interpolate_2(pass_one_data[i], target_grid_len, i, tx_pl)
                pass_two_data.append(tx_inter)
        
        return np.array(pass_two_data)


    @staticmethod    
    def _idw_interpolate(sensor_data, target_grid_len):
        '''
        Args:
            sensor_data -- np.2darray -- Rx data from a single Tx
            target_grid_len -- int -- the target grid length for the Rx
        Return:
            np.1darray -- interpolated, shape = target_grid_len**2
        '''
        pre_gl = len(sensor_data)
        factor = int(target_grid_len/pre_gl)
        interpolate = np.zeros((target_grid_len, target_grid_len))
        for new_x in range(target_grid_len):
            for new_y in range(target_grid_len):
                if new_x%factor == 0 and new_y%factor == 0 and sensor_data[new_x//factor][new_y//factor] != 0:  # don't need to interpolate
                    interpolate[new_x][new_y] = sensor_data[new_x//factor][new_y//factor]
                else:
                    v_x, v_y = new_x/factor, new_y/factor    # virtual point in the coarse grid
                    points = []                              # pick some points from the coarse grid
                    for pre_x in range(math.floor(v_x - 2), math.ceil(v_x + 1) + 2):
                        for pre_y in range(math.floor(v_y - 2), math.ceil(v_y + 1) + 2):
                            if pre_x >= 0 and pre_x < pre_gl and pre_y >= 0 and pre_y < pre_gl and sensor_data[pre_x][pre_y] != 0:
                                points.append((pre_x, pre_y, distance((v_x, v_y), (pre_x, pre_y))))
                    points = sorted(points, key=lambda tup: tup[2])           # sort by distance
                    threshold = min(MultiIntepolate.NEIGHBOUR_NUM, len(points))
                    weights = np.zeros(threshold)
                    for i in range(threshold):
                        point = points[i]
                        dist = distance((v_x, v_y), point)
                        weights[i] = (1./dist)**2                     # inverse weighted distance or inverse weighted square
                    weights /= np.sum(weights)                        # normalize them
                    idw = 0
                    for i in range(threshold):
                        w = weights[i]
                        pre_rss = sensor_data[points[i][0]][points[i][1]]
                        idw += w*pre_rss
                    interpolate[new_x][new_y] = idw
        return interpolate.reshape(target_grid_len*target_grid_len)


    @staticmethod
    def _idw_interpolate_2(pre_inter, target_grid_len, tx, tx_pl):
        '''Fix one transmitters, interpolate the sensors
        Args:
            pre_inter       -- np.1darray -- pre interpolated, shape = pre_gl*pre_gl
            target_grid_len -- int -- the Tx that needs to interpolate Rx
            factor          -- int
        Return:
            np.1darray -- interpolated, shape = gre_gl*gre_gl*factor*factor
        '''
        pre_gl = int(math.sqrt(len(pre_inter)))                      # previous grid length (coarse grid)
        pre_inter = pre_inter.reshape((pre_gl, pre_gl))
        factor = int(target_grid_len/pre_gl)
        tx_x, tx_y = tx//target_grid_len, tx%target_grid_len
        inter = np.zeros((target_grid_len, target_grid_len))
        for new_x in range(target_grid_len):
            for new_y in range(target_grid_len):
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
                    threshold = min(MultiIntepolate.NEIGHBOUR_NUM, len(points))
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
        return inter.reshape(target_grid_len*target_grid_len)



def read_clean_itwom(txfile):
    '''Read all pathloss
    '''
    fspl, itwom = read_data(txfile)
    clean_itwom(itwom, fspl)
    return itwom



def main1():
    # granularity5  = 'output9'
    # granularity10 = 'output7'
    # granularity20 = 'output10'
    # txmg = TxMultiGran(4, 5, debug=True)
    # txs = [(4, 5), (8, 10)]  # the same tx in grid length 10 and 20
    # grid_lens = [10, 20]
    # granularity_directory = [granularity10, granularity20]
    # for tx, grid_len, directory in zip(txs, grid_lens, granularity_directory):
    #     index = indexconvert(tx, grid_len)
    #     txfile = directory + '/' + '{:04}'.format(index)
    #     itwom = read_clean_itwom(txfile)
    #     txmg.add_sensor_data(grid_len, itwom)
    # txmg.combine_sensor_data()

    DIR1 = 'output7'          # 100 hypotheses
    DIR2 = 'output10'         # 400 hypotheses
    DIR3 = 'interpolate7'     # 1600 hypotheses interpolated
    DIR4 = 'output8'          # 1600 hypotheses
    
    # DIR1 = 'output9'            # 25 hypotheses
    # DIR2 = 'output7'            # 100 hypotheses
    # DIR3 = 'interpolate9'       # 400 hypotheses interpolated
    # DIR4 = 'output10'           # 400 hypotheses
    
    txfiles = sorted(glob.glob(DIR1 + '/*'))
    txs = []
    factors = [2]   # factors in grid_len
    directories = [DIR2]
    for txfile in txfiles:
        tx_1dindex = get_tx_index(txfile)  # 1d index of TX
        try:
            itwom = read_clean_itwom(txfile)
        except:
            print(txfile)
            continue
        grid_len = int(math.sqrt(len(itwom)))
        x, y = indexconvert(tx_1dindex, grid_len)
        txmg = TxMultiGran(x, y, debug=True)
        txmg.add_sensor_data(grid_len, itwom)
        
        for factor, directory in zip(factors, directories):
            fine_grid_len = grid_len * factor                      # multi-granularity sensor
            fine_x = x*factor
            fine_y = y*factor
            txfile = directory + '/{:04}'.format(indexconvert((fine_x, fine_y), fine_grid_len))
            itwom = read_clean_itwom(txfile)
            txmg.add_sensor_data(fine_grid_len, itwom)
        txmg.combine_sensor_data()
        txs.append(txmg)
    
    itwom_inter = MultiIntepolate.idw_interpolate(txs, target_grid_len=40)

    fspl_true, itwom_true = read_all_data(DIR4)
    clean_all_itwom(itwom_true, fspl_true)

    mean, median, root = compute_error(itwom_inter, itwom_true)
    print('ITWOM:\nmean absolute error     = {}\nmedian absolute error   = {}\nroot mean squared error = {}'.format(mean, median, root))

    mean, median, std, coarse_mean, coarse_median, coarse_std, fine_mean, fine_median, fine_std = customized_error(itwom_inter, itwom_true)
    print('\nmean      = {:.3f}, median      = {:.3f}, std      = {:.3f}\ncoar mean = {:.3f}, coar median = {:.3f}, coar std = {:.3f}\nfine mean = {:.3f}, fine median = {:.3f}, fine std = {:.3f}'.format(\
             mean, median, std, coarse_mean, coarse_median, coarse_std, fine_mean, fine_median, fine_std))

    write_all_itwom(itwom_inter, DIR3)


if __name__ == '__main__':
    main1()
