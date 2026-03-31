import kagglehub

kagglehub.dataset_download(
    "shantanughosh/vindr-mammogram-dataset-dicom-to-png",
    output_dir="./tmp"
)

print("Download success.")
