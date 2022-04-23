# Data Store Statistics

The goal of this is to find outliers in large image datasets.  For example, an outlier would be an image that has a mean signal that is more than 3 Z-Scores away from the rest of the data.  Or it could be an image that has a maximum signal that is more than 1.5 inter-quartile range.

## Example:

```
$ python3 datastore_statistics.py --data-folder tests/data --metric mean --distance=4 --direction both

4.8 4.0: tests/data/ADNI_019_S_4293_MR_Axial_2D_PASL_straight_no_ASL__br_raw_20180618154855516_760_S696759_I1011350.dcm: min 0.0, mu 383.0, med 205.0 max 1171.0
4.4 4.0: tests/data/ADNI_019_S_4293_MR_Axial_2D_PASL_straight_no_ASL__br_raw_20180618154739217_802_S696759_I1011350.dcm: min 0.0, mu 359.0, med 149.5 max 1171.0
```

# Test Data
MedicalMNIST
https://www.kaggle.com/datasets/amritpal333/adni4dicomnano10514
