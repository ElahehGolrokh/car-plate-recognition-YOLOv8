
import argparse
from omegaconf import OmegaConf

from src.exporting import Exporter


parser = argparse.ArgumentParser(
    prog='export.py',
    description='Export model to Hugging Face Hub',
    epilog='Thanks for using.'
)

parser.add_argument("-ri",
                    "--repo_id",
                    type=str,
                    help="Hugging Face repo ID where model is stored")
parser.add_argument("-fn",
                    "--file_name",
                    type=str,
                    help="File name to save in repo (default = 'model.pt')")
parser.add_argument("--api_token", type=str, help="Hugging Face API token")

args = parser.parse_args()

config = OmegaConf.load('private_settings.yaml')
API_TOKEN = args.api_token if args.api_token else config.api_token
FILE_NAME = args.file_name if args.file_name else config.file_name
MODEL_PATH = config.file_name   # path to your trained model file
REPO_ID = args.repo_id if args.repo_id else config.repo_id


def main(repo_id: str, api_token: str, filename: str):
    exporter = Exporter(model_path=MODEL_PATH,
                        repo_id=repo_id,
                        api_token=api_token)
    exporter.run(filename=filename)


if __name__ == "__main__":
    main(repo_id=REPO_ID, api_token=API_TOKEN, filename=FILE_NAME)
