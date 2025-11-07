import gradio as gr
import os
import sys
import tempfile
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
from src.utils import get_plate_number, get_unique_plates


# Path to your YOLOv8 weights
MODEL_PATH = "runs/detect/train/weights/best.pt"


def save_temp(input_name, output_file):
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, input_name)
    temp_video_path = os.path.join(temp_dir, temp_path)
    with open(temp_video_path, "wb") as f:
        f.write(output_file)
    return temp_video_path


def process_image(input_file):
    input_name = input_file.name
    reader = easyocr.Reader(['en'])
    predictor = ImagePredictor(input=input_name,
                               model_path=MODEL_PATH,
                               reader=reader,
                               output_name=None,
                               save_output=False)
    output_file = predictor.run()
    labels = predictor._labels
    return labels, output_file


def process_video(input_file):
    input_name = input_file.name
    reader = easyocr.Reader(['en'])
    predictor = VideoPredictor(input=input_name,
                               model_path=MODEL_PATH,
                               reader=reader,
                               output_name=None,
                               save_output=False)
    output_file = predictor.run()
    output_path = save_temp(input_name, output_file)
    labels = predictor._labels
    return labels, output_path


with gr.Blocks() as demo:
    gr.Markdown("# 🚗 Car Plate Recognition (Image or Video)")

    with gr.Row():
        input_type = gr.Radio(
            ["Image", "Video"],
            label="Input Type",
            value="Image",
            info="Select whether to upload an image or video.",
        )

    file_input = gr.File(label="Upload Image or Video")

    # Two possible output components
    labels_output = gr.Text(label="Predicted Labels", visible=True)
    image_output = gr.Plot(label="Predicted Image", visible=True)
    video_output = gr.Video(label="Predicted Video", visible=False)

    def toggle_output(selected_type):
        return (
            gr.update(visible=selected_type == "Image"),
            gr.update(visible=selected_type == "Video"),
        )

    input_type.change(toggle_output, inputs=input_type, outputs=[image_output, video_output])

    with gr.Row():
        run_btn = gr.Button("Run Prediction", variant="primary")
        clear_btn = gr.Button("🧹 Clear", variant="secondary")

    def handle_run(selected_type, file_obj):
        if selected_type == "Image":
            labels, output = process_image(file_obj)
            labels = [get_plate_number(label) for label in labels]
            return labels, output, None
        else:
            labels, output = process_video(file_obj)
            unique_labels = get_unique_plates(labels)
            return unique_labels, None, output

    run_btn.click(
        fn=handle_run,
        inputs=[input_type, file_input],
        outputs=[labels_output, image_output, video_output]
    )
    clear_btn.click(
        fn=lambda: (None, None, None, None),
        inputs=None,
        outputs=[file_input, labels_output, image_output, video_output],
    )

demo.launch()
