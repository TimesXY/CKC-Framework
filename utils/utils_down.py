import os
import requests
from urllib.parse import urlparse


def download_checkpoint(url: str, save_dir: str) -> str:
    """
    Download pretrained weights. If the file already exists locally, the download is skipped.

    Args:
        url (str): URL to download the pretrained weights from.
        save_dir (str): Local directory to save the pretrained weights.

    Returns:
        str: Full local path to the pretrained weight file.
    """
    # Parse the URL to get the filename
    parsed_url = urlparse(url)
    filename = os.path.basename(parsed_url.path)

    # Create the save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Construct the full local file path
    local_path = os.path.join(save_dir, filename)

    # Download the file if it doesn't exist
    if not os.path.exists(local_path):
        print(f"Downloading pretrained weights to {local_path}...")
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            print("Pretrained weights download completed.")
        else:
            raise ValueError(f"Failed to download pretrained weights, status code: {response.status_code}")
    else:
        print(f"Pretrained weights already exist at {local_path}, skipping download.")

    return local_path
