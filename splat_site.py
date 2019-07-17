'''
A site -- the qth file in SPLAT
'''

class Site:

    LRP = "15.000  ;   Earth Dielectric Constant (Relative permittivity)\n" + \
          "0.005   ;   Earth Conductivity (Siemens per meter)           \n" + \
          "301.000 ;   Atmospheric Bending Constant (N-units)           \n" + \
          "800.000 ;   Frequency in MHz (20 MHz to 20 GHz)              \n" + \
          "5       ;   Radio Climate (5 = Continental Temperate)        \n" + \
          "0       ;   Polarization (0 = Horizontal, 1 = Vertical)      \n" + \
          "0.50    ;   Fraction of situations (50% of locations)        \n" + \
          "0.90    ;   Fraction of time (90% of the time)\n"


    def __init__(self, kind, index, lat, lon, height, x=0, y=0):
        self.kind  = kind # either 'tx' or 'rx'
        self.index = index
        self.lat = lat    # lattitude  is the Y axis
        self.lon = lon    # longtitude is the X axis
        self.height = height
        self.x = x
        self.y = y
    
    def __str__(self):
        return '{}-{:04}\n{}\n{}\n{}m\n'.format(self.kind, self.index, self.lat, abs(self.lon), self.height)


if __name__ == "__main__":
    s = Site('tx', 0, 40.747034, -72.867045, 30.0)
    print(s)
    print(Site.LRP)

    s = Site('rx', 0, 40.747034, -72.867045, 30.0)
    print(s)
    print(Site.LRP)