import os
from omegaconf import OmegaConf

from src.serving import GradioApp


config = OmegaConf.load('private_settings.yaml')
# config = OmegaConf.create({
#     "repo_id": os.getenv("REPO_ID"),
#     "hf_token": os.getenv("HF_TOKEN"),
#     "file_name": 'model.pt'
# })
# config = OmegaConf.merge(base_conf, private_conf)

app = GradioApp(config)
demo = app.build()


if __name__ == "__main__":
    demo.launch(share=True)
