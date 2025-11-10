from huggingface_hub import HfApi, upload_file, login


class Exporter:
    """
    Exports object detection model to the Hugging Face Hub.
    """
    def __init__(self,
                 repo_id: str,
                 model_path: str,
                 api_token: str):
        self.repo_id = repo_id
        self.model_path = model_path
        login(api_token)

    def run(self, filename: str = "model.pt"):
        """
        Uploads a trained model file to Hugging Face Hub.

        Args:
            repo_id (str): Hugging Face repo id in format 'username/repo-name'.
            filename (str): File name to save in repo (default = 'model.pt').

        Returns:
            str: URL of the uploaded file.
        """
        api = HfApi()

        print(f"📤 Uploading {self.model_path} → {self.repo_id}/{filename} ...")
        url = upload_file(
            path_or_fileobj=self.model_path,
            path_in_repo=filename,
            repo_id=self.repo_id,
            repo_type="model"
        )
        print(f"✅ Uploaded successfully: {url}")
        return url
