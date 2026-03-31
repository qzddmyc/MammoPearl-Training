#!/bin/bash

echo "Start installing requirements for Python..."
pip install -r requirements.txt || exit 1

echo "Start downloading dataset..."
python ./src/init/download-dataset.py || exit 1

echo "Start moving files..."
mkdir -p ./data/raw/images_png || exit 1
mv ./tmp/images_png/* ./data/raw/images_png/ || exit 1

echo "Start removing tmp directory..."
rm -rf ./tmp || exit 1

echo "Checking sha-256 for dataset..."
cd ./data/raw/images_png
PRESET="9fca19ca3f463b7c13eb32d3b4717f59ff0239dc2832b0a20c72fefa31596281"
CURRENT_HASH=$(find . -type f ! -name "dataset.sha256" -print0 | sort -z | xargs -0 sha256sum | sha256sum | awk '{print $1}') || exit 1
if [ "$CURRENT_HASH" = "$PRESET" ]; then
    echo "Hash matched"
else
    echo "Hash did not match, you can use the following command to remove downloaded files and then try again."
    echo "   $ find ./data/raw/images_png/ -mindepth 1 ! -name "dataset.sha256" -exec rm -rf {} +"
    exit 1
fi

echo "Dataset build completed"