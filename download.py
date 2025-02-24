import os
import shutil
from huggingface_hub import snapshot_download


def download_data(repo_id: str, repo_type: str = "dataset"):
    """ download data from huggingface hub"""
    snapshot_download(repo_id, 
                      repo_type=repo_type, 
                      allow_patterns=["data/*"],
                      local_dir="formatted_esci"
                      )


def change_dataset_location(src: str, dst: str):
    """ this is because hf downloads it in a weird location, so we are moving it """
    for root, dirs, files in os.walk(src):
        for file in files:
            src_file = os.path.join(root, file)
            dst_file = os.path.join(dst, file)
            print(f"Moving {src_file} to {dst_file}")
            shutil.move(src_file, dst_file)


if __name__ == "__main__":
    src_dir = "formatted_esci/data"
    dest_dir = "formatted_esci"
    download_data("tasksource/esci")
    change_dataset_location(src_dir, dest_dir)