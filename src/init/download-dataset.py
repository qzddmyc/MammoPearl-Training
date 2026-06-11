import kagglehub

kagglehub.dataset_download(
    "shantanughosh/vindr-mammogram-dataset-dicom-to-png",
    output_dir="./tmp_for_download_dataset",
)

print("Download success.")
