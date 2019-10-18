import json


class Default:
    dir_full      = 'output8'
    full_grid_len = 40        # the fine granularity
    coarse_gran   = 12
    inter_methods = ['idw', 'ildw']
    dist_close    = 8
    dist_far      = 32

class Input:
    '''Encapsulate the inputs
    '''
    def __init__(self, dir_full      = Default.dir_full,      
                       full_grid_len = Default.full_grid_len,
                       coarse_gran   = Default.coarse_gran,  
                       inter_methods = Default.inter_methods,
                       dist_close    = Default.dist_close,    
                       dist_far      = Default.dist_far):    
        self.dir_full      = dir_full
        self.full_grid_len = full_grid_len
        self.coarse_gran   = coarse_gran
        self.inter_methods = inter_methods
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
        dist_close    = json_dict['dist_close']
        dist_far      = json_dict['dist_far']
        return cls(dir_full, full_grid_len, coarse_gran, inter_methods, dist_close, dist_far)

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
        self.mean_error = mean_error
        self.mean_absolute_error = mean_absolute_error
        self.mean_error_close = mean_error_close
        self.mean_absolute_error_close = mean_absolute_error_close
        self.mean_error_far = mean_error_far
        self.mean_absolute_error_far = mean_absolute_error_far
        self.std = std
        self.std_close = std_close
        self.std_far = std_far


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
