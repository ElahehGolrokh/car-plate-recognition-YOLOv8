import gradio as gr
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


Model_Path = "runs/detect/train/weights/best.pt"


def read_plate(input_type, input_file):
    input_name = input_file.name
    reader = easyocr.Reader(['en'])
    print(f'input_type = {input_type}')
    if input_type == "Image":
        predictor = ImagePredictor(input=input_name,
                                   model_path=Model_Path,
                                   reader=reader,
                                   output_name=None,
                                   save_output=False)

        # else:
        #     predictor = VideoPredictor(input=input_name,
        #                                model_path=Model_Path,
        #                                reader=reader,
        #                                output_name=None,
        #                                save_output=False)
        output_number, output_file = predictor.run()
    else:
        raise NotImplementedError("Right now just image files are implemented.")
    return output_number, output_file


input_type = gr.Radio(["Image", "Video"],
                              label="Input Type",
                              info="Select the type of input")
input_file = gr.File(
    file_types=[".jpg", ".jpeg", ".png", ".mp4", ".avi", ".mov"], 
    label="Upload image or video"
)
output_file = gr.Plot(label="Output")
output_number = gr.Text(label="Output Plate Number")


demo = gr.Interface(
    fn=read_plate,
    inputs=[input_type, input_file],
    outputs=[output_number, output_file],
    title="Car Plate Recognition App",
)

demo.launch()
