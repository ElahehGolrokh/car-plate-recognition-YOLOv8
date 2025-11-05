import os
import shutil
import subprocess

from omegaconf import OmegaConf

from .data_preparation import PrepareData, TrainTestSplit


class Pipeline:
    """
    The complete pipeline includes data preparation in the format accepted by
    yolo, training, and exporting the trained model.

    ...
    Attributes
    ----------
        config_path: path to config.yaml file
        remove_prev_runs: specifies whether you want to remove previous runs
        prepare: specifies whether you want to implement data preparation
        train: specifies whether you want to implement training
        export: specifies whether you want to export a saved model
        model_path: path to saved model to export default value is
                    runs/detect/train/weights/best.pt
        export_format: format for exporting saved model. See YOLO documentation
                       for more details about the acceptable formats.


    Public Methods
        run()
    """
    def __init__(self,
                 config_path: str,
                 remove_prev_runs: bool = True,
                 prepare: bool = True,
                 train: bool = True,
                 export: bool = False,
                 model_path: str = 'runs/detect/train/weights/best.pt',
                 export_format: str = 'torchscript') -> None:
        self.config_path = config_path
        self.remove_prev_runs = remove_prev_runs
        self.prepare = prepare
        self.train = train
        self.export = export
        self.model_path = model_path
        self.export_format = export_format

    def run(self):
        """Runs the entire pipeline based on user prefrences"""
        Config = OmegaConf.load(self.config_path)
        if self.remove_prev_runs:
            if os.path.isdir('runs'):
                shutil.rmtree('runs', )
            else:
                print('There is no runs directory.')
        if self.prepare:
            images_dir = Config.images_dir
            labels_dir = Config.labels_dir
            raw_annot = Config.raw_annot
            PrepareData(images_dir, labels_dir, raw_annot)
            TrainTestSplit().split()
        if self.train:
            image_size = Config.image_size
            epochs = Config.epochs
            try:
                print('Training ...')
                bashCommand = f"yolo train model=yolov8n.pt data={self.config_path} epochs={epochs} imgsz={image_size}"
                process = subprocess.Popen(bashCommand.split(),
                                           stdout=subprocess.PIPE)
                # You can also assign the process.communicate() to variables output, error
                process.communicate()
            except Exception as e:
                print(f"An error occurred: {e}")

        if self.export:
            try:
                bashCommand = f"yolo export model={self.model_path} format={self.export_format}"
                process = subprocess.Popen(bashCommand.split(),
                                           stdout=subprocess.PIPE)
                # You can also assign the process.communicate() to variables output, error
                process.communicate()
            except Exception as e:
                print(f"An error occurred: {e}")
