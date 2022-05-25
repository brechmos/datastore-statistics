import os
import numpy as np

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


def threshold(x, thr, thr_direction):
    if thr_direction == 'absolute':
        return np.abs(x) > thr
    elif thr_direction == 'lower':
        thr = -thr if thr < 0 else thr
        return x < -thr
    elif thr_direction == 'higher':
        return x > thr

