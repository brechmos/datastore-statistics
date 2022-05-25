import argparse
import logging
import re
import os
import numpy as np
from collections import OrderedDict
import scipy.stats as stats
import pathlib
from reader import DataFile
import shutil
import pandas as pd
import sys
import tqdm
from utils import check_file_writable, threshold

class CustomDictOne(OrderedDict):
    def __init__(self,*arg,**kw):
        super(CustomDictOne, self).__init__(*arg, **kw)

def add_stats(file):
    data = DataFile.get_reader(file['filename']).get_data()

    mask = data < 0.1

    file['data_shape'] = data.shape

    file['min'] = np.min(data)
    file['mean'] = np.mean(data)
    file['max'] = np.max(data)
    file['std'] = np.std(data)
    file['p25'], file['median'], file['p75'] = np.percentile(data, [25, 50, 75])

    file['mask_shape'] = mask.shape

#
#  Parse arguemnts
#

output_format_choices= ['screen', 'json', 'csv']
statistic_choices= ['zscore', 'iqr', 'stats']
direction_choices= ['absolute', 'lower', 'higher']

parser = argparse.ArgumentParser()
parser.add_argument("--data-directory", type=str, dest="data_directory", help="Data directory mounting point", default="data/")
parser.add_argument("--filename-regexp", type=str, dest="filename_regexp",
                    help="Regexp on the filename). ",
                    default='.*\.(jpeg|jpg|png|dcm|dicom)')
parser.add_argument("--metric", type=str, dest="metric", help="Metric to check (mean [def], median, min, max).", default='mean')
parser.add_argument("--statistic", type=str, dest="statistic",
                    help="Static for flagging (zscore [def], iqr, stats [show all]). ",
                    choices=statistic_choices, default=statistic_choices[0])
parser.add_argument("--output-format", type=str, dest="output_format",
                    help="output Format screen [def], json, csv). ",
                    choices=output_format_choices, default=output_format_choices[0])
parser.add_argument("--threshold", type=float, dest="threshold",
                    help="Threshold for statistic for positive flag (None). ", default=None)
parser.add_argument("--threshold-direction", type=str, dest="threshold_direction",
                    help="Direction of threshold (None). ", 
                    choices=direction_choices, default=direction_choices[0])
parser.add_argument("--verbose", dest="verbose", help="More information.", action="store_true")
parser.add_argument("--logfile", type=str, dest="logfile", help="Optional log file.", default='')

args = parser.parse_args()

keep_directory_structure = True

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
# Get filenames
#
path = pathlib.Path(args.data_directory)
filenames = list([x for x in sorted(path.rglob('*')) if re.match(args.filename_regexp, str(x.name))])
file_list = [CustomDictOne({'filename': filename}) for filename in filenames]

#
#  Add stats
#
with tqdm.tqdm(file_list, desc='Loading data...', file=sys.stdout) as pbar:
    for filename in file_list:
        add_stats(filename)
        pbar.update(10)

#
#  If thresholding
#
if args.statistic == 'zscore': 

    # Compute z-score
    zscores = stats.zscore([x[args.metric] for x in file_list])

    # Add z-score
    for zscore, filename in zip(zscores, file_list):
        filename['zscore-'+args.metric] = zscore

    # Apply threshold
    if args.threshold is not None:
        file_list = [filename for filename in file_list if threshold(filename['zscore-'+args.metric], args.threshold, args.threshold_direction)]

elif args.statistic == 'iqr':

    iqr = np.diff(np.percentile([x[args.metric] for x in file_list], [25, 75]))[0]

    for filename in file_list:
        filename['iqr_scale-'+args.metric] = filename[args.metric] / iqr

    # Apply threshold
    if args.threshold is not None:
        file_list = [filename for filename in file_list if threshold(filename['iqr_scale-'+args.metric], args.threshold, args.threshold_direction)]

#
# Print out
#
df = pd.DataFrame.from_dict(file_list)

if args.output_format == 'screen':
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.max_colwidth', 54, 'display.width', shutil.get_terminal_size().columns):
        log.info(df)

elif args.output_format == 'csv':
    log.info(df.to_csv())

elif args.output_format == 'json':
    log.info(df.to_json())
