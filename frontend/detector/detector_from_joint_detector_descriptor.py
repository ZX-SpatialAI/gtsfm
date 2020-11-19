"""A wrapper over joint detector-descriptor to convert it to a detector.

Authors: Ayush Baid
"""
import numpy as np

from common.image import Image
from frontend.detector.detector_base import DetectorBase
from frontend.detector_descriptor.detector_descriptor_base import \
    DetectorDescriptorBase


class DetectorFromDetectorDescriptor(DetectorBase):
    """A wrapper class to expose the Detector component of a
    DetectorDescriptor. 

    Performs the joint detection and description but returns only the features.
    """

    def __init__(self, detector_descriptor: DetectorDescriptorBase):
        """Initialize a detector from a joint detector descriptor.

        Args:
            detector_descriptor: joint detector descriptor.
        """
        super().__init__()

        self.detector_descriptor = detector_descriptor

    def detect(self, image: Image) -> np.ndarray:
        """Detect the features in an image.

        Refer to documentation in DetectorBase for more details.

        Args:
            image: input image.

        Returns:
            detected features as a numpy array of shape (N, 2+).
        """
        features, _ = self.detector_descriptor.detect_and_describe(image)

        return features