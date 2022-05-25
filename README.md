# Data Store Statistics

The goal of this is to find outliers in large image datasets.  For example, an outlier would be an image that has a mean signal that is more than 3 Z-Scores away from the rest of the data.  Or it could be an image that has a maximum signal that is more than 1.5 inter-quartile range.

## Example:

```
$ python3 ds.py --data-directory tests/data --metric mean --threshold=2 --threshold-direction absolute

Checking the mean of the images for a zscore more than 2.0 in direction both
                                                filename data_shape  min        mean   max         std    p25  median     p75 mask_shape  zscore-mean
0  tests/data/ADNI_016_S_6834_MR_Axial_rsfMRI__Eyes_O...   (64, 64)    0  286.690186  1645  372.269190  21.00    37.0  627.00   (64, 64)     3.166938
1  tests/data/ADNI_019_S_4293_MR_Axial_2D_PASL_straig...   (64, 64)    0  359.009766  1171  351.553084  13.00   149.5  691.00   (64, 64)     4.390354
2  tests/data/ADNI_019_S_4293_MR_Axial_2D_PASL_straig...   (64, 64)    0  382.993408  1171  363.073309  17.00   205.0  720.25   (64, 64)     4.796081
3  tests/data/ADNI_019_S_4293_MR_Axial_2D_PASL_straig...   (64, 64)    0  243.097656  1708  345.401036  16.75    74.0  295.25   (64, 64)     2.429491
```

# Test Data
MedicalMNIST
https://www.kaggle.com/datasets/amritpal333/adni4dicomnano10514

# To Do

* [ ] Create proper package for project 
* [ ] Create executable python script rather than __main__
* [ ] Output log file with arbitrary separators for parsing
* [ ] Recursive through directory structure
* [ ] Regex on filename
* [ ] Load images from Dataset/Blobstorage
* [ ] Add help description to --help
* [ ] Stats on one dataset, outliers in another dataset
