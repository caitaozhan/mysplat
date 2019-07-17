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
from collections import defaultdict
import subprocess
from splat_site import Site
from site_manager import SiteManager


class RunSplat:
    TERRAIN_DIR = 'terrain_files'
    INPUT_DIR   = 'input'
    OUTPUT_DIR  = 'output'

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
                    command = ['splat', '-d', RunSplat.TERRAIN_DIR, '-t', tx, '-r', rx]
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
                    command = ['splat', '-d', RunSplat.TERRAIN_DIR, '-t', tx, '-r', virtual_file]
                subprocess.call(command)
            # break
        os.chdir(owd)


    def call_splat_parallel(self, num_cores=12):
        print('Generating pathloss for all pairs of sites')
        if os.path.exists(RunSplat.OUTPUT_DIR):
            shutil.rmtree(RunSplat.OUTPUT_DIR)
        os.mkdir(RunSplat.OUTPUT_DIR)

        owd = os.getcwd()
        os.chdir(RunSplat.INPUT_DIR)
        ps = []
        tx_files = sorted(glob.glob('tx*.qth'))
        rx_files = sorted(glob.glob('rx*.qth'))
        for tx in tx_files:
            for rx in rx_files:
                tx_index = tx[-8:-4]
                rx_index = rx[-8:-4]
                if tx_index != rx_index:
                    command = ['splat', '-d', RunSplat.TERRAIN_DIR, '-t', tx, '-r', rx]
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
                    command = ['splat', '-d', RunSplat.TERRAIN_DIR, '-t', tx, '-r', virtual_file]
                
                p = subprocess.Popen(command, stdout=subprocess.PIPE)
                ps.append((p, command))
                
                while len(ps) == num_cores:
                    for p, c in ps:
                        p.communicate()
                        print(c)
                    ps = []
            # break
        os.chdir(owd)


    def preprocess_output(self):
        '''Preprocess the raw data
        '''
        if os.path.exists(RunSplat.OUTPUT_DIR):
            shutil.rmtree(RunSplat.OUTPUT_DIR)
        os.mkdir(RunSplat.OUTPUT_DIR)

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
                else:
                    print(output, 'no match')
                m2 = p2.search(content)
                if m2:
                    pathloss2 = m2.group(1)
                else:
                    print(output, 'no match')

                pathloss1s[tx].append(pathloss1)
                pathloss2s[tx].append(pathloss2)

        for output in glob.glob(RunSplat.INPUT_DIR + '/tx-*-to-rx-*.txt'):
            os.remove(output)

        for key, value in pathloss1s.items():
            with open(RunSplat.OUTPUT_DIR+'/'+key, 'a') as f:
                f.write(','.join(map(lambda x: str(x), value)))
                f.write('\n')

        for key, value in pathloss2s.items():
            with open(RunSplat.OUTPUT_DIR+'/'+key, 'a') as f:
                f.write(','.join(map(lambda x: str(x), value)))


    def generate_localization_input(self, resultfile, sen_num=200):
        '''Generate the results that the localization algo needs
        Args:
            resultfile -- str
        '''
        if not os.path.exists(resultfile):
            os.mkdir(resultfile)
        results = sorted(glob.glob(resultfile+'*'))
        latest_r = results[-1]
        

        
if __name__ == '__main__':
    grid_len  = 20
    ref_point = (40.746000, -73.026220)
    cell_len  = 100
    tx_height = 30
    rx_height = 15
    siteman = SiteManager(grid_len, tx_height, rx_height)
    # siteman.generate_sites(ref_point, cell_len)
    # siteman.create_input_files(RunSplat.INPUT_DIR)

    runsplat = RunSplat(siteman)
    # runsplat.generate_terrain_files()  # only need to run for the first time
    # runsplat.call_splat()
    runsplat.call_splat_parallel(num_cores=11)
    runsplat.preprocess_output()
    # runsplat.generate_localization_input('result', sen_num=200)
