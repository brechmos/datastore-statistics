# Data Store Statistics

The goal of this is to find outliers in large image datasets.  For example, an outlier would be an image that has a mean signal that is more than 3 Z-Scores away from the rest of the data.  Or it could be an image that has a maximum signal that is more than 1.5 inter-quartile range.

## Example:

```
$ python3 datastore_statistics.py --data-folder tests/data --metric mean --distance=2 --direction both

Checking the mean of the images for a zscore more than 2.0 in direction both

2.3 2.0 tests/data2/009076_mask.jpeg: min 0.0, mu 0.0, med 0.0 max 0.0
2.0 2.0 tests/data2/005088.jpeg: min 0.0, mu 10.3, med 0.0 max 241.0
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
