import cv2
import os
import random
import shutil

from bs4 import BeautifulSoup
from typing import List


class PrepareData:
    """
    Prepares data in the specific format defined by YOLO. The initial dataset
    must contain images and annotations folders (The names of these folders
    are defined in the config file). The annotations are first converted to
    plain text, then are saved in .txt format and finally all coordinates of
    the bounding boxes are normalized.

    ...
    Attributes
    ----------
        images_dir: path to images dir
        labels_dir: path to labels dir in which the converted labels are going
        to be stored
        raw_annot: path to initial annotations dir


    Private Methods
    ---------------
        _create_dirs()
        _modify_xmls()
        _xml_to_txt()
        _normalize_coordinates()
        _normalize_dataset()

    """
    def __init__(self, images_dir: str, labels_dir: str, raw_annot: str) -> None:
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.raw_annot = raw_annot
        self._normalize_dataset()

    def _create_dirs(self) -> None:
        """
        Creates images/ labels directories in the data root dir which is
        specified as 'data' by default in the config file
        """
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.labels_dir, exist_ok=True)

    def _modify_xmls(self) -> None:
        """
        Converts xml contents to a plain text containing only bbx class and
        locations
        """
        self._create_dirs()
        if len(os.listdir(self.labels_dir)) == 0:
            print('Convert xml contents to a plain text containing only bbx class and locations. \n',
                  'It may take a while ...')
            for file in os.listdir(self.raw_annot):
                filepath = os.path.join(self.raw_annot, file)
                bbxs = []
                with open(filepath, 'r') as f:
                    content = f.read()
                    bs_data = BeautifulSoup(content, "xml")
                    xmins = bs_data.find_all('xmin')
                    xmaxs = bs_data.find_all('xmax')
                    ymins = bs_data.find_all('ymin')
                    ymaxs = bs_data.find_all('ymax')
                    f.close()
                for i in range(len(xmins)):
                    xmin = int(xmins[i].text)
                    xmax = int(xmaxs[i].text)
                    ymin = int(ymins[i].text)
                    ymax = int(ymaxs[i].text)
                    object_class = 0
                    bbx_x_center = (xmax + xmin)/2
                    bbx_y_center = (ymax + ymin)/2
                    bbx_width = xmax - xmin
                    bbx_height = ymax - ymin
                    bbxs.append([bbx_x_center, bbx_y_center, bbx_width, bbx_height])

                filepath = os.path.join(self.labels_dir, file)
                with open(filepath, 'w') as f:
                    # for label in labels:
                    for bbx in bbxs:
                        f.write('{} '.format(object_class))
                        f.write('{} '.format(bbx[0]))
                        f.write('{} '.format(bbx[1]))
                        f.write('{} '.format(bbx[2]))
                        f.write('{}'.format(bbx[3]))
                        f.write('\n')
                    f.close()
            print('Done with modifying xmls!')
        else:
            print('The labels are already in plain texts as .xml files.')

    def _xml_to_txt(self) -> None:
        """Convert labels from xml to txt"""
        self._modify_xmls()
        labels = os.listdir(self.labels_dir)
        labels_check = [True if label.endswith('xml') else False for label in labels]
        if all(labels_check):
            for filename in labels:
                filepath = os.path.join(self.labels_dir, filename)
                if not os.path.isdir(filepath) and filepath.endswith('.xml'):
                    with open(filepath, 'r') as f:
                        content = f.read()
                        f.close()
                    txt_file = filepath.replace('xml', 'txt')
                    # !touch {txt_file}
                    with open(txt_file, 'w') as f:
                        f.write(content)
                    os.remove(filepath)
        else:
            print('The labels are already in txt formats.')

        # Test
        for i, filename in enumerate(os.listdir(self.labels_dir)):
            if filename.endswith('.xml'):
                raise ValueError('There are xml files in labels folder.')

        print(f'Number of label files with .txt format: {len(os.listdir(self.labels_dir))}')

    @staticmethod
    def _normalize_coordinates(image_path, bbox) -> List[float]:
        """
        Normalize bounding box coordinates.

        :param image_path: Path to the image file.
        :param bbox: Bounding box coordinates [x_min, y_min, x_max, y_max].
        :return: Normalized bounding box coordinates
        """
        image = cv2.imread(image_path)
        height, width = image.shape[:2]

        x_min, y_min, x_max, y_max = bbox
        x_min_norm = x_min / width
        y_min_norm = y_min / height
        x_max_norm = x_max / width
        y_max_norm = y_max / height

        # Ensure coordinates are within [0, 1]
        x_min_norm = min(max(x_min_norm, 0), 1)
        y_min_norm = min(max(y_min_norm, 0), 1)
        x_max_norm = min(max(x_max_norm, 0), 1)
        y_max_norm = min(max(y_max_norm, 0), 1)

        return [x_min_norm, y_min_norm, x_max_norm, y_max_norm]

    def _normalize_dataset(self, output_dir=None) -> None:
        """
        Normalize bounding box coordinates for the entire dataset.

        :param output_dir: Directory to save the normalized labels.
        """
        self._xml_to_txt()
        if not output_dir:
            output_dir = self.labels_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for label_file in os.listdir(self.labels_dir):
            label_path = os.path.join(self.labels_dir, label_file)
            image_path = os.path.join(self.images_dir,
                                      label_file.replace('.txt', '.png'))

            if not os.path.exists(image_path):
                continue

            with open(label_path, 'r') as file:
                lines = file.readlines()

            normalized_lines = []
            normalized = True
            for line in lines:
                class_id, x_min, y_min, x_max, y_max = map(float,
                                                           line.strip().split())
                norm_check = [True for item in [x_min, y_min, x_max, y_max] if item > 1]
                # If any of items are greater than 1 it means that they are not normalized yet
                if any(norm_check):
                    normalized_bbox = self._normalize_coordinates(image_path,
                                                                  [x_min, y_min, x_max, y_max])
                    normalized_lines.append(f"{class_id} {' '.join(map(str, normalized_bbox))}\n")
                else:
                    normalized = False
                    break
            if normalized:
                output_path = os.path.join(output_dir, label_file)
                with open(output_path, 'w') as file:
                    file.writelines(normalized_lines)
            else:
                print('The labels are already normalized.')
                break


class TrainTestSplit:
    """
    Randomly splits data into train, validation and test. This stage is
    totally optional.

    ...
    Attributes
    ----------
        image_train_path: path to train images. default is data/images/train.
        image_validation_path: path to validation images. default is data/images/validation
        image_test_path: path to test images. default is data/images/test
        train: defines the train portion for spliting
        val: defines the validation portion for spliting


    Private Methods
    ---------------
        _create_dirs()

    Public Methods
        split()
    """
    def __init__(self,
                 image_train_path: str = 'data/images/train',
                 image_validation_path: str = 'data/images/validation',
                 image_test_path: str = 'data/images/test',
                 train: float = .7,
                 val: float = .2) -> None:
        self.image_train_path = image_train_path
        self.image_validation_path = image_validation_path
        self.image_test_path = image_test_path
        self.train, self.val = train, val

    def _create_dirs(self) -> None:
        """
        Creates train/val/test directories in the data root dir for both
        images and labels
        """
        os.makedirs(self.image_train_path, exist_ok=True)
        os.makedirs(self.image_validation_path, exist_ok=True)
        os.makedirs(self.image_test_path, exist_ok=True)
        os.makedirs(self.image_train_path.replace("images", "labels"),
                    exist_ok=True)
        os.makedirs(self.image_validation_path.replace("images", "labels"),
                    exist_ok=True)
        os.makedirs(self.image_test_path.replace("images", "labels"),
                    exist_ok=True)

    def split(self, src_dir_path: str = 'data/images') -> None:
        """
        Splits data into train/validation/test partitions and move them to
        the final destinations.
        """
        random.seed(4)
        shuffled_files = os.listdir(src_dir_path)
        random.shuffle(shuffled_files)
        print(f'Total number of images: {len(shuffled_files)}')
        self._create_dirs()
        train_size = self.train * len(shuffled_files)
        val_size = (self.val+self.train) * len(shuffled_files)
        for i, filename in enumerate(shuffled_files):
            image_path = os.path.join(src_dir_path, filename)
            label_path = os.path.join(src_dir_path.replace('images', 'labels'),
                                      filename.replace('png', 'txt'))
            if not os.path.isdir(image_path):
                if i <= train_size:
                    dstpath = image_path.replace(src_dir_path,
                                                 self.image_train_path)
                elif i > train_size and i <= val_size:
                    dstpath = image_path.replace(src_dir_path,
                                                 self.image_validation_path)
                else:
                    dstpath = image_path.replace(src_dir_path,
                                                 self.image_test_path)
                shutil.move(image_path, dstpath)
                shutil.move(label_path, dstpath.replace('images', 'labels').replace('png', 'txt'))

        print('Number of train images:' +
              f'{len(os.listdir(self.image_train_path))}')
        print('Number of train labels:' +
              f'{len(os.listdir(self.image_train_path.replace("images","labels")))}')
        print('Number of validation images:' +
              f'{len(os.listdir(self.image_validation_path))}')
        print('Number of validation labels:' +
              f'{len(os.listdir(self.image_validation_path.replace("images", "labels")))}')
        print('Number of test images:' +
              f'{len(os.listdir(self.image_test_path))}')
        print('Number of test labels:' +
              f'{len(os.listdir(self.image_test_path.replace("images", "labels")))}')
