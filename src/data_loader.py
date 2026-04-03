import kagglehub
import os
import shutil

def download_dataset():
    
    path = kagglehub.dataset_download("blastchar/telco-customer-churn")

    raw_dir = os.path.join("data", "raw")
    os.makedirs(raw_dir, exist_ok=True)

    # Copy all files from cache to data/raw/
    for file in os.listdir(path):
        src = os.path.join(path, file)
        dst = os.path.join(raw_dir, file)
        shutil.copy(src, dst)
        print(f"Copied: {file} → data/raw/")

    print("Dataset ready in data/raw/")
    return raw_dir

if __name__ == "__main__":
    download_dataset()