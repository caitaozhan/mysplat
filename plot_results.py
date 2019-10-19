'''
Line plot
X -- various amount of training data
Y -- interpolation error
'''

import numpy as np
import tabulate
from collections import defaultdict
import matplotlib.pyplot as plt
from input_output import Input, Output, IOUtility


class PlotResult:
    plt.rcParams['font.size'] = 60
    plt.rcParams['lines.linewidth'] = 10
    plt.rcParams['lines.markersize'] = 20

    METHOD = ['ILDW', 'IDW']
    _COLOR = ['r',     'b']
    COLOR  = dict(zip(METHOD, _COLOR))

    @staticmethod
    def reduce_avg(vals):
        vals = [val for val in vals if val is not None]
        vals = [val for val in vals if np.isnan(val)==False]
        return np.mean(vals)

    @staticmethod
    def error_traning(data):
        '''
            y-axis: interpolation mean error
            x-axis: various amount of training data
        '''
        # step 1: prepare data
        d = {}
        reduce_f = PlotResult.reduce_avg

        # mean absolute error
        metrics = ['mean_absolute_error', 'mean_absolute_error_close']
        methods = ['idw', 'ildw']
        for metric in metrics:
            table = defaultdict(list)
            for myinput, output_by_method in data:
                if myinput.coarse_gran == 20:  # skip 20, makes plot look a little worse
                    continue
                table[myinput.coarse_gran].append({method: output.get_metric(metric) for method, output in output_by_method.items()})
            print_table = [[x] + [reduce_f([(y_by_method[method] if method in y_by_method else None) for y_by_method in list_of_y_by_method]) for method in methods] for x, list_of_y_by_method in sorted(table.items())]
            d[metric] = print_table
            print('Metric:', metric)
            print(tabulate.tabulate(print_table, headers = ['GRANULARITY'] + methods), '\n')

        # mean error
        metrics = ['mean_error', 'mean_error_close']
        methods = ['idw', 'ildw']
        for metric in metrics:
            table = defaultdict(list)
            for myinput, output_by_method in data:
                if myinput.coarse_gran == 20:  # skip 20, makes plot look a little worse
                    continue
                table[myinput.coarse_gran].append({method: output.get_metric(metric) for method, output in output_by_method.items()})
            print_table = [[x] + [reduce_f([(y_by_method[method] if method in y_by_method else None) for y_by_method in list_of_y_by_method]) for method in methods] for x, list_of_y_by_method in sorted(table.items())]
            d[metric] = print_table
            print('Metric:', metric)
            print(tabulate.tabulate(print_table, headers = ['GRANULARITY'] + methods), '\n')


        # step 2: the plot
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(40, 20))
        fig.subplots_adjust(left=0.065, right=0.98, top=0.98, bottom=0.18, wspace = 0.25)

        mae  = np.array(d['mean_absolute_error'])                # mae
        maec = np.array(d['mean_absolute_error_close'])          # maec
        gran      = list(mae[:, 0])
        xlabels   = [int(round(g**2/1600, 2)*100) for g in gran]
        idw_mae   = mae[:, 1]
        ildw_mae  = mae[:, 2]
        idw_maec  = maec[:, 1]
        ildw_maec = maec[:, 2]
        ind = np.arange((len(idw_mae)))
        ax0.plot(idw_mae, color=PlotResult.COLOR['IDW'], marker='o', label='IDW,   All Location')
        ax0.plot(ildw_mae, color=PlotResult.COLOR['ILDW'], marker='o', label='ILDW, All Location')
        ax0.plot(idw_maec, color=PlotResult.COLOR['IDW'], marker='o', linestyle='--', label='IDW,   Near Location')
        ax0.plot(ildw_maec, color=PlotResult.COLOR['ILDW'], marker='o', linestyle='--', label='ILDW, Near Location')
        ax0.set_ylabel('Mean Absolute Error (dB)', fontsize=70)
        ax0.set_xlabel('Percentage of Full Training Data', labelpad=30, fontsize=70)
        ax0.set_xticks(ind)
        ax0.set_xticklabels(xlabels)
        ax0.tick_params(axis='y', direction='in', length=10, width=3, pad=15)
        ax0.set_ylim([0, 5])
        ax0.set_yticks(np.arange(0, 5.1, step=0.5))
        ax0.legend()

        me  = np.array(d['mean_error'])              # me
        mec = np.array(d['mean_error_close'])        # mec
        idw_me   = me[:, 1]
        ildw_me  = me[:, 2]
        idw_mec  = mec[:, 1]
        ildw_mec = mec[:, 2]
        ax1.plot(idw_me, color=PlotResult.COLOR['IDW'], marker='o', label='IDW,   All Location')
        ax1.plot(ildw_me, color=PlotResult.COLOR['ILDW'], marker='o', label='ILDW, All Location')
        ax1.plot(idw_mec, color=PlotResult.COLOR['IDW'], marker='o', linestyle='--', label='IDW,   Near Location')
        ax1.plot(ildw_mec, color=PlotResult.COLOR['ILDW'], marker='o', linestyle='--', label='ILDW, Near Location')
        ax1.set_ylabel('Mean Error (dB)', fontsize=70)
        ax1.set_xlabel('Percentage of Full Training Data', labelpad=30, fontsize=70)
        ax1.set_xticks(ind)
        ax1.set_xticklabels(xlabels)
        ax1.tick_params(axis='y', direction='in', length=10, width=3, pad=15)
        ax1.legend()

        Y_0 = [0] * len(ind)
        ax1.plot(Y_0, linewidth=4, color='k')
        plt.figtext(0.25, 0.01, '(a)', weight='bold')
        plt.figtext(0.76, 0.01, '(b)', weight='bold')

        plt.savefig('ipsn/inter_error.png')



def plot_various_training():
    logs = ['ipsn/error-gran']
    data = IOUtility.read_logs(logs)
    PlotResult.error_traning(data)


if __name__ == '__main__':
    plot_various_training()
