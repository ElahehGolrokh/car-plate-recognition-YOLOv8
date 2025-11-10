import cv2
import easyocr
from matplotlib.figure import Figure
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import os
import tempfile

from pathlib import Path
from typing import Union
from ultralytics import YOLO

from .base import PrecictorBase


class ImagePredictor(PrecictorBase):
    """
    Performs YOLOv8 inference on one or multiple test images and visualizes
    the detected license plates with OCR-based text extraction.

    This class extends `PrecictorBase` to handle still image inputs. It supports
    both single-image and directory-based batch predictions. The output can be
    displayed using matplotlib or saved to disk, depending on configuration.

    Attributes
    ----------
    input : str
        Path to a test image or directory containing test images.
    model_path : str
        Path to the trained YOLOv8 model file.
    output_name : str
        Filename for the saved output plot (if `save_output=True`).
    reader : easyocr.Reader
        EasyOCR reader instance used for reading plate text from detected regions.
    save_output : bool
        Whether to save the annotated image(s) to disk or return the matplotlib figure.

    Private Methods
    ---------------
    _check_input()
        Ensures the provided input path exists and is valid.
    _get_yolo_predictions()
        Runs YOLOv8 inference on the input image(s).
    _visualize_predictions()
        Draws bounding boxes, labels, and OCR results on detected objects.
    _crop_plate()
        Extracts the license plate region from an image.
    _read_plate()
        Applies OCR on cropped plate regions to extract alphanumeric text.

    Public Methods
    --------------
    run()
        Executes the full inference pipeline and returns the annotated image
        (as a matplotlib Figure) or saves it to disk.
    """
    def __init__(self,
                 input: str,
                 model_path: str,
                 output_name: str,
                 reader: easyocr.Reader = None,
                 save_output: bool = True) -> None:
        super().__init__(input, model_path, output_name, reader, save_output)

    def run(self) -> Figure:
        """
        Reads images and Write the prediction results on each one
        """
        if os.path.isdir(self.input):
            for file in os.listdir(self.input):
                file_path = os.path.join(self.input, file)
                fig = self._get_yolo_predictions(file_path, file)
        else:
            fig = self._get_yolo_predictions(self.input, self.output_name)

        return fig

    def _get_yolo_predictions(self, file_path: str, file: str) -> Figure:
        """
        Returns YOLOv8 prediction results for an image.

        :param file_path: path to test image or directory of images
        :param file: input file name. it will be used to save the output
        """
        results = self.model.predict(file_path)
        fig = self._visualize_predictions(results, file_path, file)
        return fig

    def _visualize_predictions(self,
                               results: list,
                               file_path: str,
                               file: str) -> Figure:
        """
        Visualizes YOLO predictions on an image.

        :param results: List of predictions
        :param file_path: path to test image or directory of images
        :param file: input file name. it will be used to save the output
        """
        plate_detected = False
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        fig, ax = plt.subplots(1)
        ax.imshow(image)

        # Add bounding boxes
        for result in results:
            for bbx in result.boxes:
                plate_detected = True
                x_min, y_min, x_max, y_max = map(int, bbx.xyxy[0])
                bounding_box = [x_min, y_min, x_max, y_max]
                label = self.model.names[int(bbx.cls)]
                confidence = bbx.conf.item()

                width = x_max - x_min
                height = y_max - y_min

                # Create a rectangle patch
                rect = patches.Rectangle((x_min, y_min),
                                         width,
                                         height,
                                         linewidth=2,
                                         edgecolor='g',
                                         facecolor='none')
                ax.add_patch(rect)

                # Add label
                plt.text(x_min,
                         y_min - 10,
                         f'{label} {confidence:.2f}',
                         color='g',
                         fontsize=12,
                         bbox=dict(facecolor='white', alpha=0.5))

                # Add OCR Result
                ocr_result = self._read_plate(image, *bounding_box)
                if ocr_result:
                    # label = f"Plate Number: {ocr_result[0][1]}, Confidence: {ocr_result[0][2]:.2f}"
                    label = f"Plate Number: {ocr_result[0][1]}"
                else:
                    label = 'Unable to read'
                plt.text(x_min - width/2,
                         y_max + height/2,
                         label,
                         color='black',
                         fontsize=12,
                         bbox=dict(facecolor='white', alpha=0.7))
                self._labels.append(label)
        if self.save_output:
            output_path = os.path.join('runs', file)
            plt.savefig(output_path)
        if not plate_detected:
            label = 'No Plate Detected'
            self._labels.append(label)
        return fig


class VideoPredictor(PrecictorBase):
    """
    Runs YOLOv8 inference on a video and either saves or returns
    the annotated video (for use in Gradio, etc.).

    Attributes
    ----------
        input: path to input test video
        model_path: path to saved YOLOv8 model
        output_name: name of saved output in avi format of final prediction
        frame: current video frame which is under prediction
        reader: easyocr.Reader

    Private Methods
    ---------------
        _check_input()
        _get_yolo_predictions()
        _visualize_predictions()
        _crop_plate()
        _read_plate()
    """

    def __init__(self,
                 input: str,
                 model_path: str,
                 output_name: str = "output.avi",
                 reader: easyocr.Reader = None,
                 save_output: bool = True):
        super().__init__(input, model_path, output_name, reader, save_output)
        self.model = YOLO(model_path)
        self.frame = None

    def run(self) -> Union[Path, bytes]:
        """Run YOLOv8 detection and optionally return the output video as bytes."""
        cap = cv2.VideoCapture(self.input)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {self.input}")

        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        fps = cap.get(cv2.CAP_PROP_FPS)

        if self.save_output:
            os.makedirs("runs", exist_ok=True)
            output_path = os.path.join("runs", self.output_name)
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        else:
            # Use a temporary file for in-memory result
            tmpfile = tempfile.NamedTemporaryFile(suffix=".webm", delete=False)
            out = cv2.VideoWriter(tmpfile.name,
                                  cv2.VideoWriter_fourcc(*'VP90'),
                                  fps, (frame_width, frame_height))

        while cap.isOpened():
            ret, self.frame = cap.read()
            if not ret:
                break
            self._get_yolo_predictions()
            out.write(self.frame)

        cap.release()
        out.release()

        if self.save_output:
            return output_path  # local file path
        else:
            with open(tmpfile.name, "rb") as f:
                video_bytes = f.read()
            os.remove(tmpfile.name)
            return video_bytes

    def _get_yolo_predictions(self) -> None:
        """Perform YOLOv8 inference on current frame."""
        results = self.model(self.frame)
        for result in results:
            for box in result.boxes:
                label = self._visualize_predictions(box)
                self._labels.append(label)

    def _visualize_predictions(self, box) -> str:
        """Draw bounding boxes and OCR results."""
        x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
        label = self.model.names[int(box.cls)]
        conf = box.conf.item()
        color = (0, 255, 0)

        cv2.rectangle(self.frame, (x_min, y_min), (x_max, y_max), color, 2)
        cv2.putText(self.frame, f"{label} {conf:.2f}", (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        ocr_result = self._read_plate(self.frame, x_min, y_min, x_max, y_max)
        label = ocr_result[0][1] if ocr_result else "Unable to read"
        cv2.putText(self.frame, label, (x_min, y_max + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        return label
