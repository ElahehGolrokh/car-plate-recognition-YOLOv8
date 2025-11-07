import cv2
import easyocr
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

import tempfile
from ultralytics import YOLO

from .base import PrecictorBase


class ImagePredictor(PrecictorBase):
    """
    Gets predictions and visualization of a yolo saved model on test images
    ...
    Attributes
    ----------
        input: path to input test image or directory
        model_path: path to saved YOLOv8 model
        output_name: name of plt saved figure of final prediction
        reader: easyocr.Reader which would be passed if the user wants to
                read the plate number

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
    """
    def __init__(self,
                 input: str,
                 model_path: str,
                 output_name: str,
                 reader: easyocr.Reader = None,
                 save_output: bool = True) -> None:
        super().__init__(input, model_path, output_name, reader, save_output)

    def run(self) -> list[str, Figure]:
        """
        Reads images and Write the prediction results on each one
        """
        if os.path.isdir(self.input):
            # The gradio app only is implemented for single image or video
            # So we dont need the label here
            label = None
            for file in os.listdir(self.input):
                file_path = os.path.join(self.input, file)
                _, fig = self._get_yolo_predictions(file_path, file)
        else:
            label, fig = self._get_yolo_predictions(self.input, self.output_name)
        # return label, fig
        return fig

    def _get_yolo_predictions(self, file_path: str, file: str) -> list[str, Figure]:
        """
        Gets YOLOv8 predictions for an image.

        :param file_path: path to test image or directory of images
        :param file: input file name. it will be used to save the output
        """
        # Get predictions
        results = self.model.predict(file_path)
        # Visualize the predictions
        label, fig = self._visualize_predictions(results, file_path, file)
        return label, fig

    def _visualize_predictions(self,
                               results: list,
                               file_path: str,
                               file: str) -> list[str, Figure]:
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
        if self.save_output:
            output_path = os.path.join('runs', file)
            plt.savefig(output_path)
        if not plate_detected:
            label = 'No Plate Detected'
        return label, fig


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
        reader: easyocr.Reader which would be passed if the user wants to
                read the plate number

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

    def run(self):
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
            tmpfile = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
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

    def _get_yolo_predictions(self):
        """Perform YOLOv8 inference on current frame."""
        results = self.model(self.frame)
        for result in results:
            for box in result.boxes:
                self._visualize_predictions(box)

    def _visualize_predictions(self, box):
        """Draw bounding boxes and OCR results."""
        x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
        label = self.model.names[int(box.cls)]
        conf = box.conf.item()
        color = (0, 255, 0)

        cv2.rectangle(self.frame, (x_min, y_min), (x_max, y_max), color, 2)
        cv2.putText(self.frame, f"{label} {conf:.2f}", (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        if self.reader:
            ocr_result = self._read_plate(self.frame, x_min, y_min, x_max, y_max)
            ocr_label = ocr_result[0][1] if ocr_result else "Unable to read"
            cv2.putText(self.frame, ocr_label, (x_min, y_max + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
