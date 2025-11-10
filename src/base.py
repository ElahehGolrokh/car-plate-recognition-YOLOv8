import easyocr
import numpy as np
import os

from abc import ABC, abstractmethod
from matplotlib.figure import Figure
from pathlib import Path
from typing import Tuple, Union
from ultralytics import YOLO


class PrecictorBase(ABC):

    """
    Gets predictions and visualization of a yolo saved model on test images
    and videos

    Notes:
        - you have to override these methods: ``_get_yolo_predictions``,
                                              ``_visualize_predictions``,
                                              ``run()``
        - don't override these methods: ``_check_input``,
                                        ``_crop_plate``,
                                        ``_read_plate``

    ...
    Attributes
    ----------
        input: path to input image/video or dir of test images
        model_path: path to saved YOLOv8 model
        output_name: name of plt saved figure of final prediction
        reader: easyocr.Reader which would be passed if the user wants to
                read the plate number
        save_output: bool flag to save the output or not

    Private Methods
    ---------------
        _check_input()
        _get_yolo_predictions()
        _visualize_predictions()
        _crop_plate()
        _read_plate()

    Public Methods
    --------------
        run()

    Examples:
        >>> predictor = PrecictorBase(input,
                                      model_path,
                                      output_name,
                                      reader)
        >>> predictor.run()

    """
    def __init__(self,
                 input: str,
                 model_path: str,
                 output_name: str,
                 reader: easyocr.Reader = None,
                 save_output: bool = True) -> None:
        self.input = input
        self.model = YOLO(model_path)
        self.output_name = output_name
        self.reader = reader
        self.save_output = save_output
        self._labels = []
        self._check_input()

    def _check_input(self) -> None:
        """Checks if the input path exists"""
        if not os.path.exists(self.input):
            raise FileNotFoundError('No such file or directory')

    @abstractmethod
    def run(self) -> Union[Figure, Path, bytes]:
        """
        Run YOLOv8 prediction on the input.

        Returns
        -------
        Figure (for images)
            Matplotlib figure showing predictions.
        Path (for saved videos)
            Path to the output video file.
        bytes (for in-memory video)
            Raw video bytes suitable for Gradio streaming.
        """

    @abstractmethod
    def _get_yolo_predictions(self,) -> None:
        """
        Performs YOLO inference and updates the current frame or image.
        (No return value; modifies state in subclasses.)
        """

    @abstractmethod
    def _visualize_predictions(self,) -> Union[Figure, str]:
        """
        Visualizes YOLO predictions on an image or video frame.

        Returns
        -------
        Figure or str
            Figure for image-based prediction, or recognized plate string
            for video.
        """

    @staticmethod
    def _crop_plate(image: np.ndarray, *bbx: list) -> np.ndarray:
        """
        Crops car plate from original image
        """
        plate_crop = image[bbx[0][1]: bbx[0][3], bbx[0][0]: bbx[0][2]]
        return plate_crop

    def _read_plate(self, image: np.ndarray, *bbx: list) -> Tuple:
        """
        Get the results from easyOCR for each image
        """
        plate_crop = self._crop_plate(image, bbx)
        ocr_result = self.reader.readtext(plate_crop,
                                          allowlist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_.•· ")
        return ocr_result
