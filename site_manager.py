'''
Manage all the Sites: site generation etc.
'''
import os
import shutil
from splat_site import Site
from mpu import haversine_distance



class SiteManager:
    def __init__(self, grid_len, tx_height, rx_height):
        self.grid_len  = grid_len  # int
        self.cell_len  = 0         # int -- meters
        self.ref_point = (0, 0)   # (float, float) -- (lat, lon) -- the lower left corner is the reference point
        self.tx_height = tx_height
        self.rx_height = rx_height
        self.sites     = []


    def __str__(self):
        return '\n'.join(map(lambda x: str(x), self.sites))


    def generate_sites(self, reference_point, cell_len):
        '''Generate all the sites according to reference point, grid length, and cell length
        Args:
            reference_point (float, float) -- (lat, lon)
            cell_len        int -- meters
        '''
        print('reference point = {}, cell length = {} m'.format(reference_point, cell_len))
        lat_step, lon_step = SiteManager.find_step(reference_point, cell_len)
        self.lat_step = lat_step
        self.lon_step = lon_step
        print('lat step = {}, lon step = {}'.format(lat_step, lon_step))
        print('start creating sites')
        self.sites = []
        for x in range(self.grid_len):
            for y in range(self.grid_len):
                lon = reference_point[1] + x*lon_step
                lat = reference_point[0] + y*lat_step
                index = x*self.grid_len + y
                self.sites.append(Site('tx', index, lat, lon, self.tx_height, index//self.grid_len, index%self.grid_len))
                self.sites.append(Site('rx', index, lat, lon, self.tx_height, index//self.grid_len, index%self.grid_len))
        self.min_lat = reference_point[0]
        self.max_lat = reference_point[0] + (self.grid_len-1)*lat_step
        self.min_lon = reference_point[1]
        self.max_lon = reference_point[1] + (self.grid_len-1)*lon_step


    def generate_virtual_site(self, site):
        '''Generate a vitual site when tx and rx are at the same location
        Args:
            site: Site
        Return:
            Site
        '''
        return Site(site.kind, site.index, site.lat+self.lat_step/3, site.lon+self.lon_step/3, site.height, site.x, site.y)


    def create_input_files(self, inputfile):
        '''create the .qth and .lcp files
        '''
        if os.path.exists(inputfile):
            shutil.rmtree(inputfile)
        os.mkdir(inputfile)

        for site in self.sites:
            qthfile = 'input/{}-{:04}.qth'.format(site.kind, site.index)
            lrpfile = 'input/{}-{:04}.lrp'.format(site.kind, site.index)
            with open(qthfile, 'w') as fq, open(lrpfile, 'w') as fl:
                fq.write(str(site))
                fl.write(Site.LRP)


    @staticmethod
    def find_step(ref_point, cell_len):
        '''
        Args:
            cell_len (int) the length of a cell in meters
        Return:
            (float, float): the step in lattitude and longtitude
        '''
        lat, lon = ref_point[0], ref_point[1]
        eps = 1e-10
        cell_len /= 1000. # from m to km

        low, high = 0, 1
        while low <= high:
            mid = (high + low)/2
            lat2 = lat + mid
            dist = haversine_distance(ref_point, (lat2, lon))
            # print('{} {} {} {}'.format(low, mid, high, dist))
            if dist < cell_len + eps and dist > cell_len - eps:
                break
            if dist > cell_len:
                high = mid
            elif dist < cell_len:
                low  = mid

        low, high = 0, 1
        while low <= high:
            mid = (high + low)/2
            lon2 = lon + mid
            dist = haversine_distance(ref_point, (lat, lon2))
            # print('{} {} {} {}'.format(low, mid, high, dist))
            if dist < cell_len + eps and dist > cell_len - eps:
                break
            if dist > cell_len:
                high = mid
            elif dist < cell_len:
                low  = mid
        return lat2 - lat, lon2 - lon




def main1():
    grid_len  = 20
    ref_point = (40.746000, -73.026220)
    cell_len  = 100
    tx_height    = 30
    rx_height    = 15
    inputfile  = 'input'
    siteman = SiteManager(grid_len, tx_height, rx_height)
    siteman.generate_sites(ref_point, cell_len)
    siteman.create_input_files(inputfile)
    # print(siteman)


if __name__ == '__main__':
    main1()
