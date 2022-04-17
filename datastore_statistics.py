from pathlib import Path
import argparse
import numpy as np
from reader import DataFile
import scipy.stats as stats

class DataStoreStatistics:
    """
    """

    def __init__(self, filenames):
        self._filenames = filenames

        if len(filenames) == 0:
            raise('No filenames were passed to DataStoreStatistics')

    def flag_metrics(self, metric='mean', nstds=2, lh='both', z_score_threshold=3):
        """
        metric: type of metric to check, needs to match one of metrics from reader get_stats
        nstds: number of standard deviations away from the mean to flag
        lh: check only 'lower' than mean, 'higher' than mean, or 'both'
        """

        readers = []
        for filename in self._filenames:
            readers += [DataFile.get_reader(filename)]

        mean_values = [reader.get_metrics()[metric] for reader in readers]

        # Compute Z-Score
        zscores = stats.zscore(mean_values)

        mean, std = np.mean(mean_values), np.std(mean_values)
        print(f'Overall mean of metric "{metric}" is {mean:0.1f} and std is {std:0.1f}')

        for reader, zscore in zip(readers, zscores):

            if lh == 'both':
                if np.abs(zscore) > z_score_threshold:
                    print(f'Z-Score {zscore:1.1f}: {reader}')

            elif lh == 'lower':
                if zscore < -z_score_threshold:
                    print(f'Z-Score {zscore:1.1f}: {reader}')

            elif lh == 'higher':
                if zscore > z_score_threshold:
                    print(f'Z-Score {zscore:1.1f}: {reader}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-folder", type=str, dest="data_folder", help="data folder mounting point", default="data/")
    parser.add_argument("--metric", type=str, dest="metric", help="Metric to check", default='mean')
    parser.add_argument("--nstds", type=int, dest="nstds", help="Number of standard deviations away to flag.", default=2)
    parser.add_argument("--direction", type=str, dest="direction", help="Direction of standard deviations to flag.", default='both')
    parser.add_argument("--z-score-threshold", type=float, dest="z_score_threshold", help="Threshold on the Z-Score at which to flag.", default=3)

    args = parser.parse_args()

    print(' ')

    filenames = list(Path(args.data_folder).glob('*'))
    dss = DataStoreStatistics(filenames)
    dss.flag_metrics(args.metric, nstds=args.nstds, lh=args.direction, z_score_threshold=args.z_score_threshold)

