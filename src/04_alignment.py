# =========================================================================
# 04_alignment.py 概要（(筆記外):作者自加的前處理，不在 AOI 13 步內）
#
# 【在做什麼】把「傾斜（旋轉＋平移）的 PCB 圖」轉正，對齊到參考圖 reference.jpg 的位置與角度，
#            這樣 05 才能把兩張圖逐像素相減找差異。只處理名稱以 _Rotation 結尾的資料夾（60 張）。
#
# 【如何轉正】每張圖現場自己算，程式事先不知道要轉幾度：
#   1. 兩張圖各找最多 5000 個 ORB 特徵點（銲點、線條轉角等容易辨認的地標），每個點附一個「指紋」（描述子）。
#   2. 比對指紋：參考圖每個點到輸入圖找最像的前 2 名，最像的要明顯優於第二像（距離 < 0.75 倍）才留下配對。
#   3. 用配對點的座標，RANSAC 解出一個 3x3 矩陣 H：記錄「輸入圖座標 → 參考圖座標」的搬法，
#      旋轉角與平移都藏在 H 裡（程式沒有單獨算角度）；RANSAC 會剔除配錯的點（誤差 > 5 像素）。
#   4. warpPerspective 依 H 產生轉正後的圖：對「輸出圖每個位置」，用 H 的反矩陣
#      （OpenCV 內部自動做）去輸入圖查顏色；輸出尺寸＝參考圖尺寸。
#
# 【實測參考】（01_open_circuit_01 一張）：5000 組配對留下 2732 組、內點 2598、旋轉約 -9°。
#            60 張的旋轉角約 -10°～+9°，內點最少 2594。只看數字，沒有逐張肉眼檢查對位結果。
# 【輸入／輸出】讀 dataset/*_Rotation/*.jpg（彩色）→ 輸出 results/04_alignment/aligned_images/（彩色，供 05 讀）。
# =========================================================================
from pathlib import Path

import cv2

import cv_io
import matplotlib.pyplot as plt
import numpy as np

# -------------------------------------------------------------------------
# PATH SETUP
# -------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent       # 宣告BASE_DIR紀錄路徑pcb-defect-detection
DATASET_DIR = BASE_DIR / "dataset"                      # 宣告資料集的路徑，這裡假設資料集位於專案根目錄下的 "dataset" 資料夾
REFERENCE_PATH = BASE_DIR / "reference.jpg"             # 宣告參考圖像的路徑，這裡假設參考圖像位於專案根目錄下的 "reference.jpg" 檔案
RESULTS_DIR = BASE_DIR / "results" / "04_alignment"     # 宣告處理後圖像的輸出路徑，這裡假設輸出位於專案根目錄下的 "results/04_alignment" 資料夾
ALIGNED_DIR = RESULTS_DIR / "aligned_images"            # 宣告對齊後圖像的輸出路徑，這裡假設輸出位於專案根目錄下的 "results/04_alignment/aligned_images" 資料夾
PREVIEW_PATH = RESULTS_DIR / "alignment_preview.jpg"    # 宣告對齊預覽圖像的輸出路徑，這裡假設輸出位於專案根目錄下的 "results/04_alignment/alignment_preview.jpg"
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"} # 允許的圖片格式


def align_image(reference_gray, reference_keypoints, reference_descriptors, detector, image):
    """Align an input image to the reference using ORB features and RANSAC."""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 步驟2:轉灰階（只用來找特徵點，輸出的 aligned 仍是彩色）
    image_keypoints, image_descriptors = detector.detectAndCompute(gray_image, None)  # (筆記外)對位-1:找輸入圖的 ORB 特徵點與描述子
    if image_descriptors is None:
        raise RuntimeError("No ORB features found in image.")

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)                                          # (筆記外)對位-2:特徵比對，用 Hamming 距離（兩個描述子有幾個 bit 不同）
    pairs = matcher.knnMatch(reference_descriptors, image_descriptors, k=2)            # 參考圖每個特徵點，在輸入圖找最像的前 2 名
    good_matches = [                                                                   # 比例測試：最像的要明顯優於第二像（< 0.75 倍）才保留
        first
        for pair in pairs
        if len(pair) == 2
        for first, second in [pair]
        if first.distance < 0.75 * second.distance
    ]
    if len(good_matches) < 10:  # 好配對太少，矩陣不可靠，報錯（主程式會記錄這張失敗）
        raise RuntimeError(f"Not enough good matches ({len(good_matches)} found).")

    # (筆記外)對位-3:取出配對點的座標，準備算轉換矩陣
    reference_points = np.float32(
        [reference_keypoints[match.queryIdx].pt for match in good_matches]
    ).reshape(-1, 1, 2)
    image_points = np.float32(
        [image_keypoints[match.trainIdx].pt for match in good_matches]
    ).reshape(-1, 1, 2)
    homography, inlier_mask = cv2.findHomography(      # (筆記外)對位-4:RANSAC 算 3x3 轉換矩陣，誤差超過 5 像素的配對視為外點剔除
        image_points, reference_points, cv2.RANSAC, 5.0
    )
    inlier_count = int(inlier_mask.sum()) if inlier_mask is not None else 0  # 內點數 = 與矩陣吻合的配對數
    if homography is None or inlier_count < 10:  # 矩陣算不出來或內點太少 → 不可信，報錯
        raise RuntimeError(f"Could not compute reliable homography ({inlier_count} inliers).")

    height, width = reference_gray.shape  # 取參考圖的高、寬，當作輸出圖尺寸
    aligned = cv2.warpPerspective(image, homography, (width, height))  # (筆記外)對位-5:依矩陣把輸入圖轉到參考圖的位置，輸出尺寸＝參考圖尺寸
    return aligned, len(good_matches), inlier_count


def main():
    if not DATASET_DIR.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {DATASET_DIR}")

    reference = cv_io.imread(str(REFERENCE_PATH), cv2.IMREAD_COLOR)  # 步驟1:影像輸入（參考圖，彩色讀入）
    if reference is None:
        raise FileNotFoundError(f"Reference image not found: {REFERENCE_PATH}")

    reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)  # 步驟2:轉灰階（只用來找特徵點）
    detector = cv2.ORB_create(nfeatures=5000)  # (筆記外)對位-1:ORB 特徵偵測器，最多取 5000 個特徵點
    reference_keypoints, reference_descriptors = detector.detectAndCompute(  # 參考圖只算一次，之後每張輸入圖都拿來比對
        reference_gray, None
    )
    if reference_descriptors is None or len(reference_keypoints) < 4:  # 參考圖找不到特徵點就無法對位，直接停止
        raise RuntimeError(f"Could not find ORB features in reference: {REFERENCE_PATH}")

    # 只收集名稱以 _Rotation 結尾的資料夾（共 60 張旋轉圖）
    rotation_folders = sorted(
        path
        for path in DATASET_DIR.rglob("*")
        if path.is_dir() and path.name.casefold().endswith("_rotation")
    )
    image_paths = sorted(
        image_path
        for folder in rotation_folders
        for image_path in folder.rglob("*")
        if image_path.is_file()
        and image_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise FileNotFoundError(
            f"No supported images found in '*_Rotation' folders under {DATASET_DIR}"
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    previews = {}   # 每個資料夾只留第一張，最後畫預覽圖用
    failures = []   # 收集失敗訊息，跑完全部再統一回報
    for image_path in image_paths:
        relative_path = image_path.relative_to(DATASET_DIR)  # 相對於 dataset 的路徑，輸出時保持相同資料夾結構
        image = cv_io.imread(str(image_path), cv2.IMREAD_COLOR)  # 步驟1:影像輸入（彩色讀入）
        if image is None:
            failures.append(f"Could not read image: {image_path}")
            continue

        try:
            aligned, match_count, inlier_count = align_image(
                reference_gray,
                reference_keypoints,
                reference_descriptors,
                detector,
                image,
            )
            output_path = ALIGNED_DIR / relative_path  # 對位後的圖存到 aligned_images/，供 05 讀取
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv_io.imwrite(str(output_path), aligned):
                raise OSError(f"Could not save aligned image: {output_path}")

            previews.setdefault(relative_path.parent, (image_path.name, aligned))  # 同資料夾只記第一張（已存在就不覆蓋）
            print(
                f"Aligned: {relative_path} "
                f"({match_count} matches, {inlier_count} inliers)"
            )
        except (RuntimeError, OSError, cv2.error) as error:  # 對位失敗、存檔失敗、OpenCV 錯誤：記錄後繼續下一張
            failures.append(f"{relative_path}: {error}")
            print(f"ERROR aligning {relative_path}: {error}")

    print(f"\nAligned {len(image_paths) - len(failures)} of {len(image_paths)} image(s).")
    print(f"Aligned images saved under: {ALIGNED_DIR}")

    if previews:
        figure, axes = plt.subplots(
            len(previews),
            2,
            figsize=(14, max(5, 4 * len(previews))),
            squeeze=False,
        )
        for row, (folder, (image_name, aligned)) in enumerate(previews.items()):
            axes[row, 0].imshow(cv2.cvtColor(reference, cv2.COLOR_BGR2RGB))
            axes[row, 0].set_title("Reference: reference.jpg")
            axes[row, 1].imshow(cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB))
            axes[row, 1].set_title(f"Aligned: {folder.name} / {image_name}")
            axes[row, 0].axis("off")
            axes[row, 1].axis("off")

        figure.tight_layout()
        figure.savefig(PREVIEW_PATH, dpi=150, bbox_inches="tight")
        print(f"Reference/alignment preview saved to: {PREVIEW_PATH}")
        plt.show()

    if failures:
        for failure in failures:
            print(f"ERROR: {failure}")
        raise RuntimeError(
            f"{len(failures)} image(s) could not be aligned. "
            "See the errors printed above."
        )


if __name__ == "__main__":
    main()
