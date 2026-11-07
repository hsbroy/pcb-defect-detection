from pathlib import Path

import cv2

import cv_io
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_MASK_DIR = BASE_DIR / "results" / "difference_images" / "defect_mask"
RESULTS_DIR = BASE_DIR / "results"
# [Experiment parameter] Kernel sizes to compare. 5 is the author's original setting.
KERNEL_SIZES = (3, 5, 7)
SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def clean_mask(raw_mask, ksize=5):
    """Remove small isolated regions and close small gaps in a binary mask."""
    binary = cv2.threshold(raw_mask, 127, 255, cv2.THRESH_BINARY)[1]
    if cv2.countNonZero(binary) > binary.size * 0.5:
        binary = cv2.bitwise_not(binary)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)
    return opened, cleaned


def run(ksize):
    output_dir = RESULTS_DIR / f"kernel_size{ksize}" / "cleaned_masks"
    preview_path = RESULTS_DIR / f"kernel_size{ksize}" / "morphology_preview.jpg"
    if not RAW_MASK_DIR.is_dir():
        raise FileNotFoundError(
            f"Raw defect mask folder not found: {RAW_MASK_DIR}. "
            "Run 05_difference.py first."
        )

    mask_paths = sorted(
        path
        for path in RAW_MASK_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )
    if not mask_paths:
        raise FileNotFoundError(f"No defect masks found in {RAW_MASK_DIR}")

    preview_by_folder = {}
    failures = []
    processed_count = 0
    for mask_path in mask_paths:
        relative_path = mask_path.relative_to(RAW_MASK_DIR)
        raw_mask = cv_io.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if raw_mask is None:
            failures.append(f"Could not read mask: {mask_path}")
            continue

        opened, cleaned = clean_mask(raw_mask, ksize)
        output_path = output_dir / relative_path.with_suffix(".png")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv_io.imwrite(str(output_path), cleaned):
            failures.append(f"Could not save cleaned mask: {output_path}")
            continue

        preview_by_folder.setdefault(relative_path.parent, (raw_mask, opened, cleaned))
        processed_count += 1
        print(f"Cleaned mask: {relative_path}")

    print(f"\nCleaned {processed_count} of {len(mask_paths)} mask(s).")
    print(f"Cleaned masks saved under: {output_dir}")

    if preview_by_folder:
        figure, axes = plt.subplots(
            len(preview_by_folder),
            3,
            figsize=(14, max(4, 3.5 * len(preview_by_folder))),
            squeeze=False,
        )
        for row, (folder, (raw_mask, opened, cleaned)) in enumerate(
            preview_by_folder.items()
        ):
            for axis, image, title in (
                (axes[row, 0], raw_mask, "Raw defect mask"),
                (axes[row, 1], opened, "Morphological opening"),
                (axes[row, 2], cleaned, "Cleaned mask"),
            ):
                axis.imshow(image, cmap="gray")
                axis.set_title(title)
                axis.axis("off")
            axes[row, 0].set_ylabel(folder.name)

        figure.tight_layout()
        figure.savefig(preview_path, dpi=150, bbox_inches="tight")
        print(f"Morphology preview saved to: {preview_path}")

    if failures:
        for failure in failures:
            print(f"ERROR: {failure}")
        raise RuntimeError(f"{len(failures)} mask(s) could not be cleaned.")


def main():
    for ksize in KERNEL_SIZES:
        print(f"\n=== Kernel size {ksize}x{ksize} ===")
        run(ksize)


if __name__ == "__main__":
    main()
