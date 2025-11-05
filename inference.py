import argparse
import sys
import types

try:
    # If bidi.algorithm already exists, do nothing
    import bidi.algorithm as _ba  # noqa: F401
except Exception:
    # Create a minimal module that re-exports bidi.get_display
    from bidi import get_display  # will raise if not installed
    mod = types.ModuleType("bidi.algorithm")
    mod.get_display = get_display
    sys.modules["bidi.algorithm"] = mod
import easyocr

from src.prediction import ImagePredictor, VideoPredictor


parser = argparse.ArgumentParser(description='Gets model path, file path and' +
                                             'a name for saving output.')

# Defining the parser arguments
parser.add_argument('-mp',
                    '--model_path',
                    default='runs/detect/train/weights/best.pt',
                    help='path to saved YOLOv8 model')
parser.add_argument('-ip',
                    '--image_path',
                    default=None,
                    help='path to jpg or png test image')
parser.add_argument('-vp',
                    '--video_path',
                    default=None,
                    help='path to mp4 test video')
parser.add_argument('-on',
                    '--output_name',
                    default=None,
                    help='name of the output saved figure or video')
parser.add_argument('-rf',
                    '--read_flag',
                    action='store_true',  # Default value is False
                    help='specifies whether to read car plates using OCR')
args = parser.parse_args()


def main(image_path, video_path, model_path, output_name, read_flag):
    if read_flag:
        reader = easyocr.Reader(['en'])
    else:
        reader = None
    if image_path:
        if not output_name:
            output_name = 'output.png'
        predictor = ImagePredictor(image_path,
                                   model_path,
                                   output_name,
                                   reader)

    elif video_path:
        if not output_name:
            output_name = 'output.avi'
        predictor = VideoPredictor(video_path,
                                   model_path,
                                   output_name,
                                   reader)
    predictor.run()


if __name__ == '__main__':
    main(args.image_path,
         args.video_path,
         args.model_path,
         args.output_name,
         args.read_flag)
