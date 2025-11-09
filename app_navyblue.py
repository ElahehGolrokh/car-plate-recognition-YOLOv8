import json
import gradio as gr
import os
import shutil
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


def flag_plates(plates_text):
    """Save flagged plates to a file."""
    if not plates_text:
        return "⚠️ No plates to save."
    try:
        plates = json.loads(plates_text)
    except Exception:
        plates = [plates_text]

    os.makedirs("flag", exist_ok=True)
    save_path = os.path.join("flag", "plates_log.txt")
    with open(save_path, "a") as f:
        f.write("\n".join(plates) + "\n")
    return f"✅ Saved {len(plates)} plate(s) to {save_path}"


def download_output_file(file_path):
    """Prepare file for download by copying to a temp dir."""
    if not file_path:
        return None
    temp_dir = tempfile.mkdtemp()
    download_path = os.path.join(temp_dir, os.path.basename(file_path))
    shutil.copy(file_path, download_path)
    return download_path


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


with gr.Blocks(
    theme=gr.themes.Base(),
    css="""
        .gradio-container {
            width: 80vw !important;
            margin: auto;
            background: #f8faf9;
        }
        .header {
            background: linear-gradient(90deg, rgb(18, 65, 112), rgb(38, 102, 127));
            padding: 2.5em;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 1.5em;
        }
        .header h1 {
            margin: 0;
            font-size: 2rem;
            font-weight: 700;
            color: white !important;
        }
        .header p {
            margin: 0;
            font-size: 1.5rem;
            font-weight: 500;
            color: white !important;
        }
        .gr-button {
            border-radius: 6px !important;
            font-weight: 600 !important;
        }
        button.primary {
            background: #26667F !important;
            color: white !important;
        }
        button.primary:hover {
            background: #124170 !important;
            color: white !important;
        }
        input[type="radio"][aria-checked="true"] {
            border-color: #26667F !important;
            background: #26667F !important;
        }
        .output-card {
            background: white;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            padding: 1.2em;
            border: 1px solid #e2e8f0;
        }
        .status-box textarea {
            font-weight: 600;
        }
    """
) as demo:
    gr.HTML("<div class='header'><h1>🚗 Car Plate Recognition Dashboard</h1><p>Process and read vehicle plates from images or videos</p></div>")

# ) as demo:
#     gr.Markdown("# 🚗 Car Plate Recognition (Image or Video)")

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
    hidden_path = gr.Textbox(visible=False)

    def toggle_output(selected_type):
        return (
            gr.update(visible=selected_type == "Image"),
            gr.update(visible=selected_type == "Video"),
        )

    input_type.change(toggle_output, inputs=input_type, outputs=[image_output, video_output])

    with gr.Row():
        run_btn = gr.Button("Run Prediction", variant="primary")
        clear_btn = gr.Button("🧹 Clear", variant="secondary")
        flag_btn = gr.Button("🚩 Save Detected Plates", variant="secondary")
 
    with gr.Row():
        status_box = gr.Textbox(label="Status", interactive=False)

    # Connect flag button
    flag_btn.click(fn=flag_plates, inputs=labels_output, outputs=status_box)

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
        fn=lambda: (None, None, None, None, None),
        inputs=None,
        outputs=[file_input, labels_output, image_output, video_output, status_box],
    )

demo.launch()
