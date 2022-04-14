from pathlib import Path
import argparse
import numpy as np
from reader import DataFile

class DataStoreStatistics:
    """
    """

    def __init__(self, filenames):
        self._filenames = filenames

    def flag_metrics(self, metric='mean', nstds=2, lh='both'):
        """
        metric: type of metric to check, needs to match one of metrics from reader get_stats
        nstds: number of standard deviations away from the mean to flag
        lh: check only 'lower' than mean, 'higher' than mean, or 'both'
        """

        readers = []
        for filename in self._filenames:
            readers += [DataFile.get_reader(filename)]

        mean_values = [reader.get_metrics()[metric] for reader in readers]

        mean, std = np.mean(mean_values), np.std(mean_values)
        print(f'Overall mean of metric "{metric}" is {mean:0.1f} and std is {std:0.1f}')

        for reader, mean_value in zip(readers, mean_values):

            if lh == 'both':
                if np.abs(mean_value-mean) > nstds*std:
                    print(reader)

            elif lh == 'lower':
                if mean_value < mean-nstds*std:
                    print(reader)

            elif lh == 'higher':
                if mean_value > mean+nstds*std:
                    print(reader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-folder", type=str, dest="data_folder", help="data folder mounting point", default="data/")
    parser.add_argument("--metric", type=str, dest="metric", help="Metric to check", default='mean')
    parser.add_argument("--nstds", type=int, dest="nstds", help="Number of standard deviations away to flag.", default=2)
    parser.add_argument("--direction", type=str, dest="direction", help="Direction of standard deviations to flag.", default='both')

    args = parser.parse_args()

    print(' ')

    filenames = Path(args.data_folder).glob('*')
    dss = DataStoreStatistics(filenames)
    dss.flag_metrics(args.metric, nstds=args.nstds, lh=args.direction)
