'''
Call the splat
'''
import glob
import os
import shutil
import numpy as np
import urllib.request
import zipfile
import re
import random
import time
import subprocess
from collections import defaultdict
from splat_site import Site
from site_manager import SiteManager


class RunSplat:
    TERRAIN_DIR    = 'terrain_files'
    INPUT_DIR      = 'input'
    OUTPUT_DIR     = 'output'
    OUTPUT_DIR_CUR = ''
    RESULT_DIR     = 'result'
    TIMEOUT        = 0.2  #  timeout 0.2 second for subprocess.popen
    TIMEOUT_FILE   = 'command_timeout.txt'

    def __init__(self, siteman):
        self.siteman = siteman

    def generate_terrain_files(self):
        if os.path.exists(RunSplat.TERRAIN_DIR):
            shutil.rmtree(RunSplat.TERRAIN_DIR)
        terrain_temp_dir = RunSplat.TERRAIN_DIR + '/temp'
        os.makedirs(terrain_temp_dir)
        #--finds out the corresponding tiles of min/max lat/lon
        min_lat_int, max_lat_int = int( np.floor(self.siteman.min_lat) ), int( np.ceil(self.siteman.max_lat) )
        min_lon_int, max_lon_int = int( -np.ceil(self.siteman.max_lon) ), int( -np.floor(self.siteman.min_lon) )
        for cur_lat_int in range(min_lat_int - 1, max_lat_int+1): #download all NINE tiles:S north, south, east, west, north-east, north-west etc
            for cur_lon_int in range(min_lon_int-1, max_lon_int+1):
                lat_str, lon_str = str(cur_lat_int), str(cur_lon_int)
                if len(lon_str) < 3:
                    lon_str = '0' + lon_str
                terrain_file = "N" + str( lat_str ) + "W" + lon_str + ".hgt.zip"
                #------- link example: https://dds.cr.usgs.gov/srtm/version2_1/SRTM3/North_America/N10W110.hgt.zip   ----#
                print  ("Downloading Terrain file: ",terrain_file)
                terrain_file_url = "https://dds.cr.usgs.gov/srtm/version2_1/SRTM3/North_America/" + terrain_file
                try:
                    with urllib.request.urlopen(terrain_file_url) as response, open(terrain_temp_dir + '/' + str(terrain_file), 'wb') as f:
                        shutil.copyfileobj(response, f)
                except IOError as e:
                    print ("warning: terrain file "+terrain_file+" not found!", e)
                    continue
                #---uncompress the zip file-----------------#
                zip_ref = zipfile.ZipFile(terrain_temp_dir+"/"+ str(terrain_file), 'r')
                zip_ref.extractall(terrain_temp_dir)
                zip_ref.close()

        #-----now convert all the sdf files-------------#
        owd = os.getcwd()
        os.chdir(RunSplat.TERRAIN_DIR)
        for hgt_file in glob.glob('./temp/*.hgt'):
            subprocess.call(["srtm2sdf", hgt_file])
        os.chdir(owd)
        shutil.rmtree(terrain_temp_dir) #remove the temporary directory created at the beginning


    def call_splat(self):
        print('Generating pathloss for all pairs of sites')
        if os.path.exists(RunSplat.OUTPUT_DIR):
            shutil.rmtree(RunSplat.OUTPUT_DIR)
        os.mkdir(RunSplat.OUTPUT_DIR)

        owd = os.getcwd()
        os.chdir(RunSplat.INPUT_DIR)
        tx_files = sorted(glob.glob('tx*.qth'))
        rx_files = sorted(glob.glob('rx*.qth'))
        for tx in tx_files:
            for rx in rx_files:
                tx_index = tx[-8:-4]
                rx_index = rx[-8:-4]
                if tx_index != rx_index:
                    command = ['splat', '-d', '../' + RunSplat.TERRAIN_DIR, '-t', tx, '-r', rx]
                else:
                    virtual_file = 'v-' + rx
                    if not os.path.exists(virtual_file):
                        site = self.siteman.sites[(int(rx_index)+1)*2 - 1]
                        v_site = self.siteman.generate_virtual_site(site)
                        qthfile = 'v-{}-{:04}.qth'.format(v_site.kind, v_site.index)
                        lrpfile = 'v-{}-{:04}.lrp'.format(v_site.kind, v_site.index)
                        with open(qthfile, 'w') as fq, open(lrpfile, 'w') as fl:
                            fq.write(str(v_site))
                            fl.write(Site.LRP)
                    command = ['splat', '-d', '../' + RunSplat.TERRAIN_DIR, '-t', tx, '-r', virtual_file]
                subprocess.call(command)
        os.chdir(owd)


    def call_splat_parallel(self, num_cores=12):
        print('Generating pathloss for all pairs of sites in parallel')
        if os.path.exists(RunSplat.OUTPUT_DIR):
            shutil.rmtree(RunSplat.OUTPUT_DIR)
        os.mkdir(RunSplat.OUTPUT_DIR)

        f_timeout = open(RunSplat.TIMEOUT_FILE, 'w')
        owd = os.getcwd()
        os.chdir(RunSplat.INPUT_DIR)
        ps = []
        tx_files = sorted(glob.glob('tx*.qth'))
        rx_files = sorted(glob.glob('rx*.qth'))
        for tx in tx_files:
            print(tx)
            for rx in rx_files:
                tx_index = tx[-8:-4]
                rx_index = rx[-8:-4]
                if tx_index != rx_index:
                    command = ['splat', '-d', '../' + RunSplat.TERRAIN_DIR, '-t', tx, '-r', rx]
                else:
                    virtual_file = 'v-' + rx
                    if not os.path.exists(virtual_file):
                        site = self.siteman.sites[(int(rx_index)+1)*2 - 1]
                        v_site = self.siteman.generate_virtual_site(site)
                        qthfile = 'v-{}-{:04}.qth'.format(v_site.kind, v_site.index)
                        lrpfile = 'v-{}-{:04}.lrp'.format(v_site.kind, v_site.index)
                        with open(qthfile, 'w') as fq, open(lrpfile, 'w') as fl:
                            fq.write(str(v_site))
                            fl.write(Site.LRP)
                    command = ['splat', '-d', '../' + RunSplat.TERRAIN_DIR, '-t', tx, '-r', virtual_file]
                
                p = subprocess.Popen(command, stdout=subprocess.PIPE)
                ps.append((p, command, time.time()))    # some commands will run forever, add timeout manually. The API with timeout argument seems cannot support parallelism
                
                while len(ps) >= num_cores:
                    new_ps = []
                    for p, command, start in ps:
                        if p.poll() is not None:                              # (good) terminated
                            pass
                        else:                                                 # (patience) still running
                            elapse = time.time() - start
                            if elapse > RunSplat.TIMEOUT:                     # (bad) timeout
                                print('timemout: ', ' '.join(command))
                                print(' '.join(command), file=f_timeout)
                                p.kill()
                            else:                                             # (patience) still running
                                new_ps.append((p, command, start))
                    ps = new_ps

        while len(ps) > 0:
            new_ps = []
            for p, command, start in ps:
                if p.poll() is not None:
                    pass
                else:
                    elapse = time.time() - start
                    if elapse > RunSplat.TIMEOUT:                     # (bad) timeout
                        print('timemout: ', ' '.join(command))
                        print(' '.join(command), file=f_timeout)
                        p.kill()
                    else:                                             # (patience) still running
                        new_ps.append((p, command, start))
            ps = new_ps

        os.chdir(owd)
        f_timeout.close()


    def rerun_timeout(self, num_cores):
        have_timeout = 1
        while have_timeout == 1:
            have_timeout = self._rerun_timeout(num_cores)


    def _rerun_timeout(self, num_cores):
        '''SPLAT! has bugs, rerun by slightly changing the site (lat, lon) will not timeout again.
        '''
        print('Start rerunning the timeouted commands')
        int4 = r'(\d{4,4})'
        pattern = r'splat.*rx-{}.qth'.format(int4)
        p = re.compile(pattern)
        rxs = []
        commands = []
        with open(RunSplat.TIMEOUT_FILE, 'r') as f:
            for line in f:
                m = p.match(line)
                commands.append(line[:-1].split(' '))
                if m:
                    rxs.append(m.group(1))
                else:
                    print('no match')
        if len(commands) == 0:
            print('No timeout commands')
            return 0

        owd = os.getcwd()
        rxs = np.unique(rxs)
        f_timeout = open(RunSplat.TIMEOUT_FILE, 'w')
        os.chdir(RunSplat.INPUT_DIR)
        for rx in rxs:
            rx_site = self.siteman.sites[(int(rx) + 1)*2 - 1]
            rx_site = self.siteman.jitter_site(rx_site)
            qthfile = '{}-{:04}.qth'.format(rx_site.kind, rx_site.index)
            with open(qthfile, 'w') as f:
                f.write(str(rx_site))
        
        ps = []
        for command in commands:
            print(command)
            tx_index = command[4][-8:-4]
            rx_index = command[6][-8:-4]
            if tx_index == rx_index:
                virtual_file = 'v-' + command[6]
                if not os.path.exists(virtual_file):
                    site = self.siteman.sites[(int(rx_index) + 1)*2 - 1]
                    v_site = self.siteman.generate_virtual_site(site)
                    qthfile = 'v-{}-{:04}.qth'.format(v_site.kind, v_site.index)
                    lrpfile = 'v-{}-{:04}.lrp'.format(v_site.kind, v_site.index)
                    with open(qthfile, 'w') as fq, open(lrpfile, 'w') as fl:
                        fq.write(str(v_site))
                        fl.write(Site.LRP)
                command = ['splat', '-d', '../' + RunSplat.TERRAIN_DIR, '-t', command[4], '-r', virtual_file]
            
            p = subprocess.Popen(command, stdout=subprocess.PIPE)
            ps.append((p, command, time.time()))
            
            while len(ps) >= num_cores:
                new_ps = []
                for p, command, start in ps:
                    if p.poll() is not None:                    # (good) terminate
                        pass
                    else:                                       # (patience) still runing
                        elapse = time.time() - start
                        if elapse > RunSplat.TIMEOUT:           # (bad) timeout
                            print('timeout: ', ' '.join(command))
                            print(' '.join(command), file=f_timeout)
                            p.kill()
                        else:                                   # (patience) still running
                            new_ps.append((p, command, start))
                ps = new_ps

        while len(ps) > 0:
            new_ps = []
            for p, command, start in ps:
                if p.poll() is not None:
                    pass
                else:
                    elapse = time.time() - start
                    if elapse > RunSplat.TIMEOUT:                     # (bad) timeout
                        print('timemout: ', ' '.join(command))
                        print(' '.join(command), file=f_timeout)
                        p.kill()
                    else:                                             # (paience) still running
                        new_ps.append((p, command, start))
            ps = new_ps

        os.chdir(owd)
        f_timeout.close()
        return 1


    def preprocess_output(self):
        '''Preprocess the raw data
        '''
        RunSplat.OUTPUT_DIR_CUR = RunSplat.find_next_dir(RunSplat.OUTPUT_DIR)
        os.mkdir(RunSplat.OUTPUT_DIR_CUR)

        for output in glob.glob(RunSplat.INPUT_DIR + '/*report.txt'):
            os.remove(output)

        pathloss1s = defaultdict(list)
        pathloss2s = defaultdict(list)
        for output in sorted(glob.glob(RunSplat.INPUT_DIR + '/tx-*-to-rx-*.txt')):
            postive_float = r'(\d+\.\d+)'
            pattern1 = r'Free space.*\D{}.*'.format(postive_float)
            pattern2 = r'ITWOM Version 3.0.*\D{}.*'.format(postive_float)
            p1 = re.compile(pattern1)
            p2 = re.compile(pattern2)
            int4 = r'(\d{4,4})'
            pattern_file = r'.*tx-{}-to-rx.*'.format(int4)
            pfile = re.compile(pattern_file)
            mfile = pfile.match(output)
            if mfile:
                tx = mfile.group(1)
            else:
                continue
            with open(output, encoding="ISO-8859-1", mode='r') as f:
                content = f.read()
                
                m1 = p1.search(content)
                if m1:
                    pathloss1 = m1.group(1)
                    os.remove(output)                 # if exist match, then remove the file
                else:
                    pathloss1 = 0.
                    print(output, 'FSPL no match')    # no match, then need to rerun.
                
                m2 = p2.search(content)
                if m2:
                    pathloss2 = m2.group(1)
                    os.remove(output)
                else:
                    pathloss2 = 0.
                    print(output, 'ITWOM no match')

                pathloss1s[tx].append(pathloss1)
                pathloss2s[tx].append(pathloss2)

        # check whether each tx has pathloss to all rx
        self.complete = True
        for tx, pl in pathloss1s.items():
            if len(pl) != grid_len*grid_len:
                print(tx, '\'s data is incomplete!')
                self.complete = False
        for tx, pl in pathloss2s.items():
            if len(pl) != grid_len*grid_len:
                print(tx, '\'s data is incomplete!')
                self.complete = False

        # for output in glob.glob(RunSplat.INPUT_DIR + '/tx-*-to-rx-*.txt'):
        #     os.remove(output)

        for key, value in pathloss1s.items():
            with open(RunSplat.OUTPUT_DIR_CUR +'/'+key, 'a') as f:            # the output of SPLAT!
                f.write(','.join(map(lambda x: str(x), value)))
                f.write('\n')

        for key, value in pathloss2s.items():
            with open(RunSplat.OUTPUT_DIR_CUR +'/'+key, 'a') as f:            # the output of SPLAT!
                f.write(','.join(map(lambda x: str(x), value)))


    @staticmethod
    def find_next_dir(directory):
        '''If exists dir0, dir1, then return dir2
        Args:
            directory -- str
        Return:
            str
        '''
        int_ = r'(\d+)'
        pattern = r'.*\D{}'.format(int_)
        p = re.compile(pattern)
        nums = []
        for result in glob.glob(directory + '*'):
            m = p.match(result)
            if m:
                num = m.group(1)
                nums.append(int(num))
            else:
                print('no matching')
        new_dir = directory + str(max(nums)+1)
        return new_dir


    def generate_localization_input(self, sen_num=200, sensors=None):
        '''Generate the results that the localization algo needs
        Args:
            resultfile -- str
        '''
        if self.complete == False:
            print('Output data incomplete!')
            return

        if not os.path.exists('result0'):
            os.mkdir('result0')
        
        new_dir = RunSplat.find_next_dir(RunSplat.RESULT_DIR)
        os.mkdir(new_dir)

        # 1. sensors: randomly create them or from the paramaters
        random.seed(myseed)
        if sensors is None:
            sensors = random.sample(range(grid_len*grid_len), sen_num)
            sensors = [(index2d//grid_len, index2d%grid_len, index2d) for index2d in sensors]
        else:
            sen_num = len(sensors)
        random.seed(myseed)
        stds = [random.uniform(0.9, 1.1) for _ in sensors]
        with open(new_dir + '/sensors', 'w') as f:
            for sensor, std in zip(sensors, stds):
                f.write('{} {} {} {}\n'.format(sensor[0], sensor[1], std, 1))  # cost is constant 1

        # 2. covariance matrix
        with open(new_dir + '/cov', 'w') as f:
            cov = np.zeros((sen_num, sen_num))
            for i in range(sen_num):
                for j in range(sen_num):
                    if i == j:
                        cov[i, j] = stds[i] ** 2
                    f.write('{} '.format(cov[i, j]))
                f.write('\n')
        
        # 3. hypothesis file
        with open(new_dir + '/hypothesis', 'w') as f:
            for hypo in range(grid_len*grid_len):
                t_x = hypo//grid_len
                t_y = hypo%grid_len
                hypo4 = '{:04}'.format(hypo)
                output = np.loadtxt(RunSplat.OUTPUT_DIR_CUR + '/{}'.format(hypo4), delimiter=',')
                means_fspl  = output[0]   # free space path loss
                # means_itwom = output[1]   # Irregular Terrain with Obstructions Model
                means = means_fspl
                for i, sen in enumerate(sensors):
                    s_x = sen[0]
                    s_y = sen[1]
                    s_index2d = sen[2]
                    mean = means[s_index2d]
                    std  = stds[i]
                    f.write('{} {} {} {} {} {}\n'.format(t_x, t_y, s_x, s_y, mean, std))

        print('data generated in', new_dir)
        with open(new_dir + '/README', 'w') as f:
            f.write('grid len  = {}\n'.format(grid_len))
            f.write('ref point = {}\n'.format(ref_point))
            f.write('cell len  = {}\n'.format(cell_len))
            f.write('tx height = {}\n'.format(tx_height))
            f.write('rx height = {}\n'.format(rx_height))
            f.write('seed      = {}\n'.format(myseed))
        
        return sensors


def transform_sensors(sensors, factor, new_grid_len):
    '''When grid becomes fine, the grid location of a sensor also changes
    Args:
        sensors -- list<(int, int)>
        factor  -- int
    Return:
        list<(int, int)>
    '''
    new_sensors = []
    for x, y, _ in sensors:
        x *= factor
        y *= factor
        new_sensors.append((x, y, x*new_grid_len+y))
    return new_sensors


if __name__ == '__main__':
    grid_len  = 10
    ref_point = (40.746000, -73.026220)
    cell_len  = 400
    tx_height = 30
    rx_height = 15
    myseed = 0
    
    siteman = SiteManager(grid_len, tx_height, rx_height)
    siteman.generate_sites(ref_point, cell_len)
    siteman.create_input_files(RunSplat.INPUT_DIR)

    runsplat = RunSplat(siteman)
    # runsplat.generate_terrain_files()  # only need to run for the first time
    runsplat.call_splat_parallel(num_cores=11)
    runsplat.rerun_timeout(num_cores=11)
    runsplat.preprocess_output()
    sensors = runsplat.generate_localization_input(sen_num=100, sensors=None)

    # ************************* #

    previous_grid_len = grid_len
    grid_len  = 40
    ref_point = (40.746000, -73.026220)
    cell_len  = 100
    tx_height = 30
    rx_height = 15
    myseed = 0
    new_sensors = transform_sensors(sensors, int(grid_len/previous_grid_len), grid_len)

    siteman = SiteManager(grid_len, tx_height, rx_height)
    siteman.generate_sites(ref_point, cell_len)
    siteman.create_input_files(RunSplat.INPUT_DIR)

    runsplat = RunSplat(siteman)
    # runsplat.generate_terrain_files()  # only need to run for the first time
    runsplat.call_splat_parallel(num_cores=10)
    runsplat.rerun_timeout(num_cores=10)
    runsplat.preprocess_output()
    runsplat.generate_localization_input(sen_num=100, sensors=new_sensors)



    # ********************** #


    # grid_len  = 5
    # ref_point = (40.746000, -73.026220)
    # cell_len  = 200
    # tx_height = 30
    # rx_height = 15
    # myseed = 0
    
    # siteman = SiteManager(grid_len, tx_height, rx_height)
    # siteman.generate_sites(ref_point, cell_len)
    # siteman.create_input_files(RunSplat.INPUT_DIR)

    # runsplat = RunSplat(siteman)
    # # runsplat.generate_terrain_files()  # only need to run for the first time
    # runsplat.call_splat_parallel(num_cores=10)
    # runsplat.rerun_timeout(num_cores=10)
    # runsplat.preprocess_output()
    # sensors = runsplat.generate_localization_input(sen_num=10, sensors=None)

    # # ************************* #

    # previous_grid_len = grid_len
    # grid_len  = 10
    # ref_point = (40.746000, -73.026220)
    # cell_len  = 100
    # tx_height = 30
    # rx_height = 15
    # myseed = 0
    # new_sensors = transform_sensors(sensors, int(grid_len/previous_grid_len), grid_len)

    # siteman = SiteManager(grid_len, tx_height, rx_height)
    # siteman.generate_sites(ref_point, cell_len)
    # siteman.create_input_files(RunSplat.INPUT_DIR)

    # runsplat = RunSplat(siteman)
    # # runsplat.generate_terrain_files()  # only need to run for the first time
    # runsplat.call_splat_parallel(num_cores=10)
    # runsplat.rerun_timeout(num_cores=10)
    # runsplat.preprocess_output()
    # runsplat.generate_localization_input(sen_num=10, sensors=new_sensors)