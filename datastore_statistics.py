import argparse
from datetime import datetime
import getpass
import logging
import os
from pathlib import Path
import socket
import sys

import numpy as np
import scipy.stats as stats

from reader import DataFile

#
# This is to setup a logger that goes to scren and to a file.
#

class DataStoreStatistics:
    """
    """

    def __init__(self, filenames, logger):
        self._filenames = filenames
        self._logger = logger

    def _iqr(values):
        return diff(percentile(values, [25, 75]))
    
    def flag_metrics(self, metric='mean', statistic='zscore', distance=2, lh='both'):
        """
        metric: type of metric to check, needs to match one of metrics from reader get_stats
        distance: number of standard deviations away from the mean to flag
        lh: check only 'lower' than mean, 'higher' than mean, or 'both'
        """

        self._logger.info(f'Checking the {metric} of the images for a {statistic} more than {distance} in direction {lh}')
        self._logger.info(' ')

        #
        #  Create the readers for the data
        #
        self._logger.debug(f'Loading {len(self._filenames)} files...')
        readers = []
        for filename in self._filenames:
            readers += [DataFile.get_reader(filename)]

        #
        #  Compute metric for each reader
        #
        self._logger.debug('Computing metrics for each file...')
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
        self._logger.debug('Running through data, checking for outliers...')
        for reader, value in zip(readers, values):

            # TODO: Change the separator so it is parseable (e.g., CSV file)
            if lh == 'both':
                if np.abs(value) > scale * distance:
                    self._logger.info(f'{np.abs(value):3.1f} {scale*distance:3.1f} {reader}')

            elif lh == 'lower':
                if value < - (scale * distance):
                    self._logger.info(f'{np.abs(value):3.1f} {scale*distance:3.1f} {reader}')

            elif lh == 'higher':
                if value > (scale * distance):
                    self._logger.info(f'{np.abs(value):3.1f} {scale*distance:3.1f} {reader}')

def check_file_writable(fnm):
    """
    https://www.novixys.com/blog/python-check-file-can-read-write/
    """
    if os.path.exists(fnm):
        # path exists
        if os.path.isfile(fnm): # is it a file or a dir?
            # also works when file is a link and the target is writable
            return os.access(fnm, os.W_OK)
        else:
            return False # path is a dir, so cannot write as a file
    # target does not exist, check perms on parent dir
    pdir = os.path.dirname(fnm)
    if not pdir: pdir = '.'
    # target is creatable if parent dir is writable
    return os.access(pdir, os.W_OK)

if __name__ == '__main__':
    #
    #  Parse arguemnts
    #
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-folder", type=str, dest="data_folder", help="data folder mounting point", default="data/")
    parser.add_argument("--metric", type=str, dest="metric", help="Metric to check (mean [def], median, min, max).", default='mean')
    parser.add_argument("--statistic", type=str, dest="statistic", help="Stastic for flagging (zscore [def], iqr). ", default='zscore')
    parser.add_argument("--distance", type=float, dest="distance",
                        help="Distance for statistic for positive flag (1.5 def). ", default=1.5)
    parser.add_argument("--direction", type=str, dest="direction", help="Direction of standard deviations to flag.", default='both')
    parser.add_argument("--verbose", dest="verbose", help="More information.", action="store_true")
    parser.add_argument("--logfile", type=str, dest="logfile", help="Optional log file.", default='')

    args = parser.parse_args()

    #
    # Log information
    #
    log = logging.getLogger('logger')

    if args.verbose:
        loglevel = logging.DEBUG
    else:
        loglevel = logging.INFO

    log.setLevel(loglevel)

    formatter = logging.Formatter('%(message)s')

    # File handler
    if not args.logfile == '':
        
        # Check to make sure we can write to the logfile.
        if not check_file_writable(args.logfile):
           print(f'Log file {args.logfile} is not writeable.') 
           sys.exit(1)

        fh = logging.FileHandler(args.logfile, mode='w', encoding='utf-8')
        fh.setLevel(loglevel)
        fh.setFormatter(formatter)
        log.addHandler(fh)

        log.info(f'File written on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} by {getpass.getuser()} on {socket.gethostname()}')
        log.info(' ')

    # Console output
    ch = logging.StreamHandler()
    ch.setLevel(loglevel)
    ch.setFormatter(formatter)
    log.addHandler(ch)

    #
    #  Call the DSS
    #
    filenames = list(Path(args.data_folder).glob('*'))
    dss = DataStoreStatistics(filenames, logger=log)
    dss.flag_metrics(metric=args.metric, statistic=args.statistic, distance=args.distance, lh=args.direction)
