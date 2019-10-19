import json


class Default:
    dir_full      = 'output8'
    full_grid_len = 40        # the fine granularity
    coarse_gran   = 12
    inter_methods = ['idw', 'ildw']
    ildw_dist     = -1
    dist_close    = 8
    dist_far      = 32

class Input:
    '''Encapsulate the inputs
    '''
    def __init__(self, dir_full      = Default.dir_full,
                       full_grid_len = Default.full_grid_len,
                       coarse_gran   = Default.coarse_gran,
                       inter_methods = Default.inter_methods,
                       ildw_dist     = Default.ildw_dist,       # the distance when tx and rx are at the same grid cell
                       dist_close    = Default.dist_close,
                       dist_far      = Default.dist_far):
        self.dir_full      = dir_full
        self.full_grid_len = full_grid_len
        self.coarse_gran   = coarse_gran
        self.inter_methods = inter_methods
        self.ildw_dist     = ildw_dist
        self.dist_close    = dist_close
        self.dist_far      = dist_far

    def to_json_str(self):
        '''return json formated string
        '''
        inputdict = {
            'dir_full': self.dir_full,
            'full_grid_len': self.full_grid_len,
            'coarse_gran': self.coarse_gran,
            'inter_methods': self.inter_methods,
            'ildw_dist': self.ildw_dist,
            'dist_close': self.dist_close,
            'dist_far': self.dist_far
        }
        return json.dumps(inputdict)

    @classmethod
    def from_json_str(cls, json_str):
        '''Init an Input object from json string
        Args:
            json_str -- str
        Return:
            Input
        '''
        inputdict = json.loads(json_str)
        return cls.from_json_dict(inputdict)

    @classmethod
    def from_json_dict(cls, json_dict):
        dir_full      = json_dict['dir_full']
        full_grid_len = json_dict['full_grid_len']
        coarse_gran   = json_dict['coarse_gran']
        inter_methods = json_dict['inter_methods']
        ildw_dist     = json_dict['ildw_dist']
        dist_close    = json_dict['dist_close']
        dist_far      = json_dict['dist_far']
        return cls(dir_full, full_grid_len, coarse_gran, inter_methods, ildw_dist, dist_close, dist_far)

    def log(self):
        return self.to_json_str() + '\n'


class Output:
    '''Encapsulate the output of the algorithm, including the metrics
    '''
    def __init__(self, method = None,
                       mean_error = None,
                       mean_absolute_error = None,
                       mean_error_close = None,
                       mean_absolute_error_close = None,
                       mean_error_far = None,
                       mean_absolute_error_far = None,
                       std = None,
                       std_close = None,
                       std_far = None
                       ):
        self.method = method
        self.mean_error = round(mean_error, 3)
        self.mean_absolute_error = round(mean_absolute_error, 3)
        self.mean_error_close = round(mean_error_close, 3)
        self.mean_absolute_error_close = round(mean_absolute_error_close, 3)
        self.mean_error_far = round(mean_error_far, 3)
        self.mean_absolute_error_far = round(mean_absolute_error_far, 3)
        self.std = round(std, 3)
        self.std_close = round(std_close, 3)
        self.std_far = round(std_far, 3)


    def get_metric(self, metric):
        '''Return the metric
        '''
        if metric == 'mean_error':
            return self.mean_error
        elif metric == 'mean_absolute_error':
            return self.mean_absolute_error
        elif metric == 'mean_error_close':
            return self.mean_error_close
        elif metric == 'mean_absolute_error_close':
            return self.mean_absolute_error_close


    def to_json_str(self):
        '''return json formated string
        Return:
            str
        '''
        outputdict = {
            "method": self.method,
            "mean_error": self.mean_error,
            "mean_absolute_error": self.mean_absolute_error,
            "mean_error_close": self.mean_error_close,
            "mean_absolute_error_close": self.mean_absolute_error_close,
            "mean_error_far": self.mean_error_far,
            "mean_absolute_error_far": self.mean_absolute_error_far,
            "std":self.std,
            "std_close":self.std_close,
            "std_far":self.std_far
        }
        return json.dumps(outputdict)

    def log(self):
        return self.to_json_str() + '\n'

    @classmethod
    def from_json_str(cls, json_str):
        '''Init an Output object from json
        Args:
            json_str -- str
        Return:
            Output
        '''
        outputdict = json.loads(json_str)
        return cls.from_json_dict(outputdict)

    @classmethod
    def from_json_dict(cls, json_dict):
        method = json_dict["method"]
        mean_error = json_dict["mean_error"]
        mean_absolute_error = json_dict["mean_absolute_error"]
        mean_error_close = json_dict["mean_error_close"]
        mean_absolute_error_close = json_dict["mean_absolute_error_close"]
        mean_error_far = json_dict["mean_error_far"]
        mean_absolute_error_far = json_dict["mean_absolute_error_far"]
        std = json_dict["std"]
        std_close = json_dict["std_close"]
        std_far = json_dict["std_far"]
        return cls(method, mean_error, mean_absolute_error, mean_error_close, mean_absolute_error_close, \
                   mean_error_far, mean_absolute_error_far, std, std_close, std_far)


class IOUtility:

    @staticmethod
    def read_logs(logs):
        '''Read logs
        Args:
            logs -- [str, ...] -- a list of filenames
        Return:
            data -- [ (Input, {str: Output}), ... ] -- data to plot
        '''
        data = []
        for log in logs:
            f = open(log, 'r')
            while True:
                inputline = f.readline()
                if inputline == '':
                    break
                myinput = Input.from_json_str(inputline)
                output_by_method = {}
                outputline = f.readline()
                while outputline != '' and outputline != '\n':
                    output = Output.from_json_str(outputline)
                    output_by_method[output.method] = output
                    outputline = f.readline()
                data.append((myinput, output_by_method))
        return data
