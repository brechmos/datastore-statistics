from pathlib import Path
import argparse
import numpy as np
import scipy.stats as stats
from reader import DataFile

class DataStoreStatistics:
    """
    """

    def __init__(self, filenames):
        self._filenames = filenames

    def _iqr(values):
        return diff(percentile(values, [25, 75]))
    
    def _zscore(values):
        """
        Compute the Z-Score of the values.
        """
        return stats.zscore(values)

    def flag_metrics(self, metric='mean', statistic='zscore', distance=2, lh='both'):
        """
        metric: type of metric to check, needs to match one of metrics from reader get_stats
        distance: number of standard deviations away from the mean to flag
        lh: check only 'lower' than mean, 'higher' than mean, or 'both'
        """

        #
        #  Create the readers for the data
        #
        readers = []
        for filename in self._filenames:
            readers += [DataFile.get_reader(filename)]

        #
        #  Compute metric for each reader
        #
        values = [reader.get_metrics()[metric] for reader in readers]

        #
        #
        #
        if statistic == 'zscore':
            values = stats.zscore(values)
            scale = 1
        
        elif statistic == 'iqr':
            scale = np.diff(np.percentile(values, [25, 75]))[0]

        # 
        #  Loop over the readers and flag ones beyond the distnace * sstatistic
        #
        for reader, value in zip(readers, values):

            if lh == 'both':
                if np.abs(value) > scale * distance:
                    print(f'{np.abs(value):3.1f} {scale*distance:3.1f}: {reader}')

            elif lh == 'lower':
                if value < - (scale * distance):
                    print(f'{np.abs(value):3.1f} {scale*distance:3.1f}: {reader}')

            elif lh == 'higher':
                if value > (scale * distance):
                    print(f'{np.abs(value):3.1f} {scale*distance:3.1f}: {reader}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-folder", type=str, dest="data_folder", help="data folder mounting point", default="data/")
    parser.add_argument("--metric", type=str, dest="metric", help="Metric to check (mean [def], median, min, max).", default='mean')
    parser.add_argument("--statistic", type=str, dest="statistic", help="Stastic for flagging (zscore [def], iqr). ", default='zscore')
    parser.add_argument("--distance", type=float, dest="distance",
                        help="Distance for statistic for positive flag (1.5 def). ", default=1.5)
    parser.add_argument("--direction", type=str, dest="direction", help="Direction of standard deviations to flag.", default='both')

    args = parser.parse_args()

    print(' ')

    filenames = Path(args.data_folder).glob('*')
    dss = DataStoreStatistics(filenames)
    dss.flag_metrics(metric=args.metric, statistic=args.statistic, distance=args.distance, lh=args.direction)
