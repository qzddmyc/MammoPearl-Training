#!/usr/bin/env bash
# Download RadImageNet PyTorch pretrained models from Google Drive.
#
# Usage (run from repo root):
#   bash src/init/download_backbone.sh
#
# The zip contains: ResNet50.pt, DenseNet121.pt, InceptionResNetV2.pt, InceptionV3.pt
# Only ResNet50.pt is needed for bbox-train.py --medical-backbone-path.

set -euo pipefail

TARGET_FILE="models/raw/ResNet50.pt"
if [[ -f "${TARGET_FILE}" ]]; then
    echo "[Info] ${TARGET_FILE} already exists. Skipping download."
    exit 1
fi

FILE_ID="1RHt2GnuOYlc_gcoTETtBDSW73mFyRAtR"
ZIP_PATH="models/raw/RadImageNet_pytorch.zip"
EXTRACT_DIR="models/raw"

# 1. Ensure gdown is available
if ! command -v gdown &>/dev/null; then
    echo "[Info] gdown not found. Installing via pip..."
    pip install -q gdown
fi

# 2. Download the zip file from Google Drive
echo "[Info] Downloading RadImageNet PyTorch weights (File ID: ${FILE_ID})..."
gdown "https://drive.google.com/uc?id=${FILE_ID}" -O "${ZIP_PATH}"
echo "[Info] Download complete: ${ZIP_PATH}"

# 3. Unzip the downloaded file
echo "[Info] Extracting to ${EXTRACT_DIR}/ ..."
unzip -o "${ZIP_PATH}" -d "${EXTRACT_DIR}"
echo "[Info] Extraction complete."

# 4. Remove zip file
rm -f "${ZIP_PATH}"
echo "[Info] Removed zip file."

# 5. Move ResNet50.pt to target location and remove the extracted folder
mv "${EXTRACT_DIR}/RadImageNet_pytorch/ResNet50.pt" "${EXTRACT_DIR}/ResNet50.pt"
rm -rf "${EXTRACT_DIR}/RadImageNet_pytorch"
echo "[Info] Moved ResNet50.pt to ${EXTRACT_DIR}/ and cleaned up."
