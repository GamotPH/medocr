import os
import tarfile
import urllib.request

def download_and_extract(url, output_dir, label="Model"):
    filename = url.split("/")[-1]
    model_flag = os.path.join(output_dir, "inference.pdiparams")  # Assume this file means downloaded

    print(f"\n📦 Checking {label}...")

    if os.path.exists(model_flag):
        print(f"✅ {label} already exists at '{output_dir}' — skipping download.")
        return

    os.makedirs(output_dir, exist_ok=True)
    temp_tar = os.path.join(output_dir, filename)

    try:
        print(f"⬇️ Downloading {filename}...")
        urllib.request.urlretrieve(url, temp_tar)

        with tarfile.open(temp_tar) as tar:
            tar.extractall(output_dir)

        os.remove(temp_tar)
        print(f"✅ Extracted {label} to '{output_dir}'")

    except Exception as e:
        print(f"❌ Failed to download {label}: {e}")


download_and_extract(
    url="https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_rec_infer.tar",
    output_dir="models/ppocrv4_rec",
    label="Recognition Model (v4, English)"
)

download_and_extract(
    url="https://paddleocr.bj.bcebos.com/PP-OCRv3/ch/ch_ppocr_mobile_v2.0_cls_infer.tar",
    output_dir="models/ppocrv4_cls",
    label="Angle Classifier (v3)"
)
