import json
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

from huggingface_hub import hf_hub_download
from matplotlib.figure import Figure
from omegaconf import OmegaConf
from pathlib import Path

from src.prediction import ImagePredictor, VideoPredictor
from src.utils import get_plate_number, get_unique_plates


# Path to your YOLOv8 weights
MODEL_PATH = "model.pt"


class GradioApp:
    def __init__(self,
                 config: OmegaConf):
        self.repo_id = config.repo_id
        self.file_name = config.file_name

    def build(self) -> None:
        """
        Builds the Gradio application.

        Loads model, builds the Gradio interface,
        and returns the app.

        Notes
        -----
        This method blocks execution until the server is stopped.
        """
        self._load_model()
        app = self._get_interface()
        return app
    
    def _load_model(self) -> None:
        """
        Download required artifacts from Hugging Face Hub.

        Returns
        -------
        dict
            Dictionary mapping artifact keys (e.g. "model", "scaler") to
            local file paths where artifacts are cached.
        """
        try:
            self.model_path = hf_hub_download(repo_id=self.repo_id, filename=self.file_name)
        except Exception as e:
            raise ValueError(f"⚠️ Could not download {self.file_name}: {e}")
    
    @staticmethod
    def _save_temp(input_name: str, output_file: bytes) -> Path:
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, input_name)
        temp_video_path = os.path.join(temp_dir, temp_path)
        with open(temp_video_path, "wb") as f:
            f.write(output_file)
        return temp_video_path

    @staticmethod
    def _flag_plates(plates_text: str) -> str:
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

    def _process_image(self, input_file: gr.File) -> tuple[list, Figure]:
        input_name = input_file.name
        reader = easyocr.Reader(['en'])
        predictor = ImagePredictor(input=input_name,
                                model_path=self.model_path,
                                reader=reader,
                                output_name=None,
                                save_output=False)
        output_file = predictor.run()
        labels = predictor._labels
        return labels, output_file

    def _process_video(self, input_file: gr.File) -> tuple[list, Path]:
        input_name = input_file.name
        reader = easyocr.Reader(['en'])
        predictor = VideoPredictor(input=input_name,
                                model_path=self.model_path,
                                reader=reader,
                                output_name=None,
                                save_output=False)
        output_file = predictor.run()
        output_path = self._save_temp(input_name, output_file)
        labels = predictor._labels
        return labels, output_path
    
    def _get_interface(self) -> gr.Blocks:
        with gr.Blocks(
            theme=gr.themes.Default(),
            css="""
                /* ---------- Gradio "default dark" colors & typography ---------- */
                :root {
                    --primary-50: #fff7ed;
                    --primary-100: #ffedd5;
                    --primary-200: #ffddb3;
                    --primary-300: #fdba74;
                    --primary-400: #fb923c;
                    --primary-500: #f97316;
                    --primary-600: #ea580c;
                    --primary-700: #c2410c;
                    --primary-800: #9a3412;
                    --primary-900: #7c2d12;
                    --primary-950: #6c2e12;
                    --secondary-50: #eff6ff;
                    --secondary-100: #dbeafe;
                    --secondary-200: #bfdbfe;
                    --secondary-300: #93c5fd;
                    --secondary-400: #60a5fa;
                    --secondary-500: #3b82f6;
                    --secondary-600: #2563eb;
                    --secondary-700: #1d4ed8;
                    --secondary-800: #1e40af;
                    --secondary-900: #1e3a8a;
                    --secondary-950: #1d3660;
                    --neutral-50: #fafafa;
                    --neutral-100: #f4f4f5;
                    --neutral-200: #e4e4e7;
                    --neutral-300: #d4d4d8;
                    --neutral-400: #bbbbc2;
                    --neutral-500: #71717a;
                    --neutral-600: #52525b;
                    --neutral-700: #3f3f46;
                    --neutral-800: #27272a;
                    --neutral-900: #18181b;
                    --neutral-950: #0f0f11;
                    --spacing-xxs: 1px;
                    --spacing-xs: 2px;
                    --spacing-sm: 4px;
                    --spacing-md: 6px;
                    --spacing-lg: 8px;
                    --spacing-xl: 10px;
                    --spacing-xxl: 16px;
                    --radius-xxs: 1px;
                    --radius-xs: 2px;
                    --radius-sm: 4px;
                    --radius-md: 6px;
                    --radius-lg: 8px;
                    --radius-xl: 12px;
                    --radius-xxl: 22px;
                    --text-xxs: 9px;
                    --text-xs: 10px;
                    --body-background-fill: #0f0f11;
                    --body-text-color: #f4f4f5;
                    --color-accent-soft: #3f3f46;
                    --background-fill-primary: #0f0f11;
                    --background-fill-secondary: var(--neutral-900);
                    --border-color-accent: #52525b;
                    --border-color-primary: var(--neutral-700);
                    --button-secondary-background-fill: var(--neutral-700)
                    --button-cancel-background-fill: #ef4444;
                    --button-cancel-background-fill-hover: #dc2626;
                    --button-cancel-border-color: var(--button-secondary-border-color);
                    --button-cancel-border-color-hover: var(--button-secondary-border-color-hover);
                    --button-cancel-text-color: white;
                    --button-cancel-text-color-hover: white;
                    --button-cancel-shadow: var(--button-secondary-shadow);
                    --button-cancel-shadow-hover: var(--button-secondary-shadow-hover);
                    --button-cancel-shadow-active: var(--button-secondary-shadow-active);
                    --button-transform-hover: none;
                    --button-transform-active: none;
                    --button-transition: all 0.2s 
                ease;
                    --button-large-padding: var(--spacing-lg) calc(2 * var(--spacing-lg));
                    --button-large-radius: var(--radius-md);
                    --button-large-text-size: var(--text-lg);
                    --button-large-text-weight: 600;
                    --button-primary-background-fill: var(--primary-600);
                    --button-primary-background-fill-hover: var(--primary-700);
                    --button-primary-border-color: var(--primary-600);
                    --button-primary-border-color-hover: var(--primary-600);
                    --button-primary-text-color: white;
                    --button-primary-text-color-hover: var(--button-primary-text-color);
                    --button-primary-shadow: none;
                    --button-primary-shadow-hover: var(--button-primary-shadow);
                    --button-primary-shadow-active: var(--button-primary-shadow);
                    --button-secondary-background-fill: var(--neutral-600);
                    --button-secondary-background-fill-hover: var(--neutral-700);
                    --button-secondary-border-color: var(--neutral-200);
                    --button-secondary-border-color-hover: var(--neutral-200);
                    --button-secondary-text-color: black;
                    --button-secondary-text-color-hover: var(--button-secondary-text-color);
                    --button-secondary-shadow: var(--button-primary-shadow);
                    --button-secondary-shadow-hover: var(--button-secondary-shadow);
                    --button-secondary-shadow-active: var(--button-secondary-shadow);
                    --button-small-padding: var(--spacing-sm) calc(1.5 * var(--spacing-sm));
                    --button-small-radius: var(--radius-md);
                    --button-small-text-size: var(--text-sm);
                    --button-small-text-weight: 400;
                    --button-medium-padding: var(--spacing-md) calc(2 * var(--spacing-md));
                    --button-medium-radius: var(--radius-md);
                    --button-medium-text-size: var(--text-md);
                    --button-medium-text-weight: 600;
                    --link-text-color-active: var(--secondary-500);
                    --link-text-color: var(--secondary-500);
                    --link-text-color-hover: var(--secondary-400);
                    --link-text-color-visited: var(--secondary-600);
                    --body-text-color-subdued: var(--neutral-400);
                    --accordion-text-color: var(--body-text-color);
                    --table-text-color: var(--body-text-color);
                    --shadow-spread: 1px;
                    --block-background-fill: var(--neutral-800);
                    --block-border-color: var(--border-color-primary);
                    --block_border_width: None;
                    --block-info-text-color: var(--body-text-color-subdued);
                    --block-label-background-fill: var(--background-fill-secondary);
                    --block-label-border-color: var(--border-color-primary);
                    --block_label_border_width: None;
                    --block-label-text-color: var(--neutral-200);
                    --block_shadow: None;
                    --block_title_background_fill: None;
                    --block_title_border_color: None;
                    --block_title_border_width: None;
                    --block-title-text-color: var(--neutral-200);
                    --panel-background-fill: var(--background-fill-secondary);
                    --panel-border-color: var(--border-color-primary);
                    --panel_border_width: None;
                    --border-color-accent-subdued: var(--border-color-accent);
                    --code-background-fill: var(--neutral-800);
                    --checkbox-background-color: var(--neutral-800);
                    --checkbox-background-color-focus: var(--checkbox-background-color);
                    --checkbox-background-color-hover: var(--checkbox-background-color);
                    --checkbox-background-color-selected: var(--color-accent);
                    --checkbox-border-color: var(--neutral-700);
                    --checkbox-border-color-focus: var(--color-accent);
                    --checkbox-border-color-hover: var(--neutral-600);
                    --checkbox-border-color-selected: var(--color-accent);
                    --checkbox-border-width: var(--input-border-width);
                    --checkbox-label-background-fill: var(--neutral-800);
                    --checkbox-label-background-fill-hover: var(--checkbox-label-background-fill);
                    --checkbox-label-background-fill-selected: var(--checkbox-label-background-fill);
                    --checkbox-label-border-color: var(--border-color-primary);
                    --checkbox-label-border-color-hover: var(--checkbox-label-border-color);
                    --checkbox-label-border-color-selected: var(--checkbox-label-border-color);
                    --checkbox-label-border-width: var(--input-border-width);
                    --checkbox-label-text-color: var(--body-text-color);
                }

                /* ---------- Container + base colors ---------- */
                .gradio-container {
                    width: 80vw !important;
                    margin: auto;
                    background: var(--body-background-fill) !important;
                    color: var(--body-text-color) !important;
                }

                .header {
                    padding: 1.5em;
                    border-radius: 10px;
                    text-align: center;
                    margin-bottom: 1.5em;
                    background: linear-gradient(90deg, var(--block-background-fill), var(--surface-2));
                    color: var(--body-text-color);
                }
                .header h1 {
                    margin: 0;
                    line-height: 2;
                    padding: 1rem
                    font-size: 2rem;
                    font-weight: 700;
                    color: var(--body-text-color);
                }
                .header p {
                    margin: 0;
                    font-size: 1.5rem;
                    font-weight: 500;
                    color: var(--body-text-color-subdued);
                }

                /* ---------- Buttons ---------- */
                .gr-button {
                    border-radius: 6px !important;
                    font-weight: 600 !important;
                }

                .gr-button-primary, button.primary, .gr-button.primary {
                    background: var(--button-primary-background-fill) !important;
                    color: white !important;
                    border: 1px solid var(--button-primary-background-fill) !important;
                }
                .gr-button-primary:hover,
                .gr-button-primary:focus,
                button.primary:hover,
                button.primary:focus,
                .gr-button.primary:hover,
                .gr-button.primary:focus {
                    background: var(--button-primary-background-fill-hover) !important;
                    outline: none !important;
                    box-shadow: 0 0 0 6px var(--focus-ring);
                }

                button.secondary, 
                .gr-button-secondary, 
                button.gr-button.secondary {
                    background-color: var(--button-secondary-background-fill) !important;
                    color: white !important;
                    border: 1px solid var(--button-secondary-background-fill) !important;
                    box-shadow: none !important;
                }
                .gr-button-secondary:hover,
                .gr-button-secondary:focus,
                button.secondary:hover,
                button.secondary:focus,
                .gr-button.secondary:hover,
                .gr-button.secondary:focus {
                    background: var(--button-secondary-background-fill-hover) !important;
                    color: var(--body-text-color) !important;
                }

                /* ---------- Output card / panels ---------- */
                .output-card {
                    border-radius: 12px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.5);
                    padding: 1.2em;
                    border: 1px solid var(--border-color);
                    background: var(--block-background-fill);
                    color: var(--body-text-color);
                }

                /* ---------- Status box ---------- */
                .status-box textarea {
                    font-weight: 600;
                    background: var(--surface-1) !important;
                    color: var(--body-text-color) !important;
                    border: 1px solid var(--border-color) !important;
                }

                /* ---------- Inputs, focus and radio visuals ---------- */
                input, textarea, select {
                    background: var(--surface-1) !important;
                    color: var(--body-text-color) !important;
                    border: 1px solid var(--border-color) !important;
                }
                input:focus, textarea:focus, select:focus {
                    outline: none !important;
                    box-shadow: 0 0 0 6px var(--focus-ring) !important;
                    border-color: var(--primary-500) !important;
                }

                /* radio/checkbox label visuals */
                input[type="radio"] + span,
                input[type="checkbox"] + span {
                    color: var(--body-text-color);
                    border-color: var(--border-color);
                    background: transparent;
                }
                input[type="radio"]:checked + span,
                input[type="checkbox"]:checked + span {
                    color: white !important;
                    border-color: var(--primary-600) !important;
                }
                input[type="radio"]:checked,
                input[type="radio"]:checked:hover {
                    border-color: var(--checkbox-border-color-selected) !important;
                    background-image: var(--radio-circle) !important;
                    background-color: var(--checkbox-background-color-selected) !important;
                }

                div.svelte-1rvzbk6 table tr {
                    background-color: var(--border-color-primary) !important;
                }

                /* additional classes */
                #output-row.unequal-height {
                    align-items: stretch !important;
                }

                .eval-box textarea {
                    font-family: monospace !important;
                    background-color: var(--block-background-fill) !important;
                    color: var(--body-text-color) !important;
                    padding: 0.6em !important;
                    border-radius: 8px !important;
                    border: 1px solid var(--border-color-primary) !important;
                    resize: none;
                }
            """
        ) as demo:
            gr.HTML("<div class='header'><h1>🚗 Car Plate Recognition Dashboard</h1><p>Process and read vehicle plates from images or videos</p></div>")

            with gr.Row():
                input_type = gr.Radio(
                    ["Image", "Video"],
                    label="Input Type",
                    value="Image",
                    info="Select whether to upload an image or video.",
                )

            file_input = gr.File(label="Upload Image or Video")

            with gr.Row(elem_id="output-row"):
                image_output = gr.Plot(label="Predicted Image", visible=True)
                video_output = gr.Video(label="Predicted Video", visible=False)
                labels_output = gr.Text(label="Predicted Labels", visible=True)
                hidden_path = gr.Textbox(visible=False)

            def toggle_output(selected_type):
                return (
                    gr.update(visible=selected_type == "Image"),
                    gr.update(visible=selected_type == "Video"),
                )

            input_type.change(toggle_output,
                              inputs=input_type,
                              outputs=[image_output, video_output])

            with gr.Row():
                run_btn = gr.Button("Run Prediction", variant="primary")
                clear_btn = gr.Button("🧹 Clear", variant="secondary")
                flag_btn = gr.Button("🚩 Save Detected Plates", variant="secondary")
        
            with gr.Row():
                status_box = gr.Textbox(label="Status", interactive=False)
            eval_box = gr.Textbox(
                label="Evaluation Results",
                interactive=False,
                lines=6,
                value=(
                    "Precision: 0.90\n"
                    "Recall:    0.90"
                    "mAP50:    0.94\n"
                    "mAP75:    0.60\n"
                    "mAP50-95: 0.55\n"
                ),
                elem_classes="eval-box")
            
            # Footer with author info
            gr.HTML("""
                <div style="text-align:center; margin-top:2rem; font-size:0.9rem; color: var(--body-text-color-subdued);">
                    Created by <a href='https://elahehgolrokh.github.io/' target='_blank'>Elaheh Golrokh</a> | 
                    <a href='https://linkedin.com/in/elahe-golrokh-736ab222a'>Contact</a>
                </div>
            """)

            # Connect flag button
            flag_btn.click(fn=self._flag_plates,
                           inputs=labels_output,
                           outputs=status_box)

            def handle_run(selected_type, file_obj):
                if selected_type == "Image":
                    labels, output = self._process_image(file_obj)
                    labels = [get_plate_number(label) for label in labels]
                    return labels, output, None
                else:
                    labels, output = self._process_video(file_obj)
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
            return demo
