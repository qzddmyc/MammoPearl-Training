"""Image segmentation utilities for MammoPearl-Training.

Provides a simple, robust segmentation pipeline (Otsu / adaptive)
that processes all preprocessed images and saves masks and base images
to `data/segmented` while preserving the original patient subfolder
structure: `data/segmented/base/<patient>/*` and
`data/segmented/mask/<patient>/*`.
"""

from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm


def segment_image(img: np.ndarray, method: str = "otsu") -> np.ndarray:
	"""Segment a single grayscale image and return a binary mask.

	Args:
		img: 2D uint8 grayscale image.
		method: 'otsu' or 'adaptive'.

	Returns:
		mask: uint8 binary mask (0 or 255).
	"""
	if img is None:
		raise ValueError("img is None")

	# ensure grayscale
	if img.ndim == 3:
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# denoise
	blur = cv2.GaussianBlur(img, (5, 5), 0)

	if method == "otsu":
		_, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	elif method == "adaptive":
		mask = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
									 cv2.THRESH_BINARY, 35, 5)
	else:
		raise ValueError(f"unknown method: {method}")

	# morphological clean-up
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
	mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
	mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

	# keep largest connected component (removes small specks)
	num_labels, labels = cv2.connectedComponents(mask)
	if num_labels <= 1:
		return mask

	# compute component sizes
	counts = np.bincount(labels.flatten())
	counts[0] = 0  # background
	largest = counts.argmax()
	mask = (labels == largest).astype("uint8") * 255

	return mask


def overlay_mask(img: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
	"""Return BGR overlay of mask on image."""
	if img.ndim == 2:
		img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
	else:
		img_bgr = img.copy()

	colored = img_bgr.copy()
	colored[mask > 0] = (0, 0, 255)  # red mask
	return cv2.addWeighted(img_bgr, 1 - alpha, colored, alpha, 0)


def process_all(method: str = "otsu"):
	"""Process all images under `data/processed/images_png`.

	Saves base (preprocessed grayscale image) to `data/segmented/base/<patient>`
	and masks to `data/segmented/mask/<patient>`, preserving the patient
	subfolder structure and keeping original filenames.
	"""
	repo_root = Path(__file__).resolve().parent.parent.parent
	input_root = repo_root / 'data' / 'processed' / 'images_png'
	out_base = repo_root / 'data' / 'segmented' / 'base'
	out_mask = repo_root / 'data' / 'segmented' / 'mask'

	for d in (out_base, out_mask):
		d.mkdir(parents=True, exist_ok=True)

	exts = ('*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff')
	files = []
	if input_root.exists():
		for ext in exts:
			files.extend(input_root.rglob(ext))

	files = sorted(files)
	for p in tqdm(files, desc='segmenting all processed images'):
		img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
		if img is None:
			continue

		if img.ndim == 3 and img.shape[2] > 3:
			img = img[:, :, :3]

		gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		mask = segment_image(gray, method=method)

		# preserve patient subfolder and original filename
		patient = p.parent.name
		patient_base_dir = out_base / patient
		patient_mask_dir = out_mask / patient
		patient_base_dir.mkdir(parents=True, exist_ok=True)
		patient_mask_dir.mkdir(parents=True, exist_ok=True)

		out_base_path = patient_base_dir / p.name
		out_mask_path = patient_mask_dir / p.name

		cv2.imencode('.png', gray)[1].tofile(str(out_base_path))
		cv2.imencode('.png', mask)[1].tofile(str(out_mask_path))


def main():
	process_all(method='otsu')


if __name__ == "__main__":
	main()
