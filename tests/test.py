from pathlib import Path
import argparse
import numpy as np
from reader import DataFile
from pathlib import Path

directory = Path('tests/data')
filenames = list(directory.glob('*.jpeg'))

print(' ')
dss = DataStoreStatistics(filenames)
dss.flag_metrics()

print(' ')
dss.flag_metrics('max', nstds=1)

print(' ')
dss.flag_metrics('max', nstds=1, lh='lower')

print(' ')
dss.flag_metrics('max', nstds=1, lh='higher')

print(' ')
dss.flag_metrics('median', nstds=2, lh='both')
