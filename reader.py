from pathlib import Path
import numpy as np
import pathlib

import imageio
import nibabel
import pydicom

class DataFile:
    """
    Base class

    Requires the definition of several methods and defines the static method that
    instantitates the reader.
    """
    def __init__(self):
        pass

    def get_data(self):
        return None

    def get_shape(self):
        return None

    def get_type(self):
        return None

    def __str__(self):
        metrics = self.get_metrics()
        return f'{self._filename}: min {metrics["min"]:0.1f}, mu {metrics["mean"]:0.1f}, med {metrics["median"]:0.1f} max {metrics["max"]:0.1f}'

    @staticmethod
    def get_reader(filename):
        """
        Static method to return the correct reader object given the filename.
        """

        # Convert to string, if needed
        if isinstance(filename, str):
            filename = Path(filename)

        # Instantiate the Dicom reader
        if filename.suffix == '.dcm':
            return DICOM(filename)

        # Instantiate the NIFTI reader
        elif '.nii' in str(filename).lower():
            return NII(filename)

        # Instantiate the jpeg/png/tiff reader
        elif filename.suffix in ['.jpeg', '.jpg', '.png', '.tiff', '.tif']:
            return ImageIO(filename)
        else:
            raise('No related reader type')

    def get_metrics(self):
        """
        Return basic stats on the data.
        """
        return {
                'mean': np.mean(self.get_data()),
                'std': np.std(self.get_data()),
                'median': np.median(self.get_data()),
                'min': np.min(self.get_data()),
                'max': np.max(self.get_data()),
                'p0.1': np.percentile(self.get_data(), 0.1),
                'p99.9': np.percentile(self.get_data(), 99.9)
        }


class ImageIO(DataFile):
    """
    DICOM Class.
    """
    def __init__(self, filename):
        super().__init__()

        self._filename = filename
        
        self._object = imageio.imread(self._filename)

    def get_type(self):
        return 'imageio'

    def get_data(self):
        return self._object

    def get_shape(self):
        return self._object.shape


class DICOM(DataFile):
    """
    DICOM Class.
    """
    def __init__(self, filename):
        super().__init__()

        self._filename = filename
        
        self._object = pydicom.dcmread(self._filename)

    def get_type(self):
        return 'dicom'

    def get_data(self):
        return self._object.pixel_array

    def get_shape(self):
        return self._object.pixel_array.shape


class NII(DataFile):
    """
    NIFTI Class
    """
    def __init__(self, filename):
        super().__init__()

        self._filename = filename
        
        self._object = nib.load(self._filename)

    def get_type(self):
        return 'nii'

    def get_data(self):
        return self.get_fdata()

    def get_shape(self):
        return self.dataobj.shape

if __name__ == '__main__':
    from pathlib import Path

    directory = Path('tests/data')
    filenames = list(directory.glob('*'))

    for filename in filenames:
        reader = DataFile.get_reader(filename)

        print(reader.get_type(), reader.get_shape(), reader.get_metrics())

