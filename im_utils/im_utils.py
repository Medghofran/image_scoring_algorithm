import math
import numpy as np
import skimage.measure as skm
import skimage.exposure as ske
from PIL import (
    Image, ImageStat
)
import cv2
from dom import DOM


class ImA():
    def __init__(self, imarray) -> None:
        self._im_rgb = np.asarray(imarray, dtype='uint8')
        self._im_grayscale = np.asarray(imarray.convert('L'), dtype='int32')
        self._resolution = self._im_grayscale.shape
        self._iqa = DOM()
        
    def res(self):
        return self._resolution

    @staticmethod
    def open(im_path):
        # Open raw image
        im = Image.open(im_path)
        # Assert that image is found
        if im == None:
            raise 'Image could not be read'
        # Create image analysis instance
        ima = ImA(im)
        return ima

    def blur(self) -> float:
        # Extract blur factor
        return skm.blur_effect(self._im_grayscale)

    def sharpness(self) -> float:
        # Extract sharpness from norms 
        sharpness = self._iqa.get_sharpness(self._im_rgb)
        return sharpness / np.sqrt(2)

    def centricity() -> float:
        pass

    def wb() -> float:
        pass

    def cb() -> float:
        pass

    def is_low_contrast(self, thres = 0.35):
        return ske.is_low_contrast(self._im_rgb, thres)

    def rms_contrast(self):
        # Normalize the GrayScale Image 
        normalized = np.array(self._im_grayscale, dtype='float32')
        normalized *= 255.0 / normalized.max()
        # Get the mean from the normalized image
        mean = np.mean(normalized)
        # Calculate RMS contrast 
        rms_constrast = np.sqrt(np.sum(np.square(normalized - mean)) / (self._resolution[0] * self._resolution[1]))
        return rms_constrast

    def contrast(self) -> float:
        # Convert image into YUV domain and extract the Y plane
        Y = cv2.cvtColor(self._im_rgb, cv2.COLOR_RGB2YUV)[:,:,0]
        # compute min and max of Y
        min = np.min(Y)
        max = np.max(Y)
        # Compute contrast
        contrast = (max - min)/ (max + min)
        return contrast

    def illumination() -> float:
        pass

    def brightness(self) -> float:
        stat = ImageStat.Stat(Image.fromarray(self._im_rgb))
        r,g,b = stat.rms
        return math.sqrt(0.241*(r**2) + 0.691*(g**2) + 0.068*(b**2)) / 255

    def evaluate() -> float:
        pass


# FUNCTIONAL TESTS
if __name__ == "__main__":
    print ("Bright image:")
    ima = ImA.open('tests/test_data/bright.jpg')
    print(ima.brightness())
    print(ima.rms_contrast())
    print(ima.is_low_contrast())
    
    print ("Dim image:")
    im_dim = ImA.open('tests/test_data/dim.jpg')
    print(im_dim.brightness())
    print(im_dim.rms_contrast())
    print(im_dim.is_low_contrast())

    print ("High contrast image:")
    ima = ImA.open('tests/test_data/high_contrast.jpg')
    print(ima.brightness())
    print(ima.rms_contrast())
    print(ima.is_low_contrast())