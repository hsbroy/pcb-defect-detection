# =========================================================================
# 05_difference.py 概要
#
# 【在做什麼】把 04 對位好的每張圖，和參考圖 reference.jpg 逐像素相減；差很多的地方就當作「疑似缺陷」，
#            輸出一張差異圖與一張黑白遮罩（白＝疑似缺陷）。
#            對應 AOI 筆記【步驟7 再次平滑】與【步驟9 二值化】；
#            「光照校正」與「與參考圖相減」不在你的 13 步內，標為 (筆記外)。
#
# 【為什麼能找缺陷】參考圖是沒有缺陷的 PCB（用 52 張非旋轉圖逐像素中位數合成）；
#                  對位後兩張圖大部分地方一樣，相減後只有缺陷處（與少數對位誤差、雜訊）會留下差異。
#
# 【流程】
#   讀 reference.jpg（灰階）與 results/04_alignment/aligned_images/*.jpg（灰階，60 張）   [步驟1+2]
#     └→ normalize_illumination：兩張圖各自除以背景亮度並正規化，降低亮度不均的影響     [(筆記外)光照校正]
#     └→ absdiff：兩圖逐像素相減取絕對值                                           [(筆記外)參考圖比對]
#     └→ GaussianBlur(3×3)：壓掉相減後的細碎雜訊                                    [步驟7]
#     └→ threshold(門檻 30)：差異 > 30 → 255（白＝疑似缺陷）                        [步驟9]
#     └→ 存 results/05_difference/difference/*.png 與 defect_mask/*.png（PNG 無損，遮罩只有 0/255）
#     └→ 存預覽圖 difference_preview.jpg（每個類別一列：參考圖、對位圖、差異圖、遮罩）
#
# 【實測】60 張遮罩白色比例：最小 0.01%、平均 0.05%、最大 0.38%（只看數字，沒有逐張檢視遮罩是否真是缺陷）。
# 【注意】
#   1. 預覽圖標題寫 "Adaptive"，但程式用的是固定門檻 30（THRESH_BINARY），不是自適應門檻。
#   2. 這個流程假設對位很準；對位若有偏差，線條邊緣也可能被當成差異（推測，未驗證）。
#   3. 遮罩白點只是「疑似」，沒有缺陷位置標註就不能算 Precision/Recall。
# 【輸入／輸出】輸入 reference.jpg、04 的 aligned_images；輸出 results/05_difference/，供 06 讀取。
# =========================================================================
from pathlib import Path

import cv2

import cv_io
import matplotlib.pyplot as plt

from illumination_utils import normalize_illumination

# -------------------------------------------------------------------------
# PATH SETUP
# -------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent                       # 宣告BASE_DIR紀錄路徑pcb-defect-detection
REFERENCE_PATH = BASE_DIR / "reference.jpg"                             # 宣告參考圖像的路徑，這裡假設參考圖像位於專案根目錄下的 "reference.jpg" 檔案
ALIGNED_DIR = BASE_DIR / "results" / "04_alignment" / "aligned_images"  # 宣告對齊後圖像的輸出路徑，這裡假設輸出位於專案根目錄下的 "results/04_alignment/aligned_images" 資料夾
RESULTS_DIR = BASE_DIR / "results" / "05_difference"                    # 宣告處理後圖像的輸出路徑，這裡假設輸出位於專案根目錄下的 "results/05_difference" 資料夾
PREVIEW_PATH = RESULTS_DIR / "difference_preview.jpg"                   # 宣告對齊預覽圖像的輸出路徑，這裡假設輸出位於專案根目錄下的 "results/05_difference/difference_preview.jpg"
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"} # 允許的圖片格式
DIFFERENCE_THRESHOLD = 30  # 步驟9:二值化門檻，差異超過 30 才算缺陷


def compute_difference(reference, aligned):
    """Return an illumination-normalized difference and binary defect mask."""
    if reference.shape != aligned.shape:
        raise ValueError(
            f"Aligned image size {aligned.shape} does not match reference "
            f"size {reference.shape}."
        )

    normalized_reference = normalize_illumination(reference)              # (筆記外)光照校正：除以背景亮度（同 03 的概念，見 illumination_utils.py）
    normalized_aligned = normalize_illumination(aligned)
    difference_blurred = cv2.GaussianBlur(                                # 步驟7:再次平滑（3x3 高斯），壓掉相減後的細碎雜訊
        cv2.absdiff(normalized_reference, normalized_aligned), (3, 3), 0  # (筆記外)參考圖比對：兩圖逐像素相減取絕對值
    )
    defect_mask = cv2.threshold(                                          # 步驟9:二值化，差異 > 門檻 → 255（白＝疑似缺陷）
        difference_blurred, DIFFERENCE_THRESHOLD, 255, cv2.THRESH_BINARY
    )[1]
    return difference_blurred, defect_mask # 回傳兩個圖像: difference_blurred 是經過光照校正後的差異圖像, defect_mask 是二值化的缺陷遮罩


def main():
    reference = cv_io.imread(str(REFERENCE_PATH), cv2.IMREAD_GRAYSCALE)   # 步驟1+2:影像輸入，直接以灰階讀入
    if reference is None:
        raise FileNotFoundError(f"Reference image not found: {REFERENCE_PATH}")
    if not ALIGNED_DIR.is_dir(): 
        raise FileNotFoundError(
            f"Aligned images folder not found: {ALIGNED_DIR}. "
            "Run 04_alignment.py first."
        ) # 如果aligned_images資料夾不存在，則拋出 FileNotFoundError

    image_paths = sorted(
        path
        for path in ALIGNED_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS #  path.is_file()只要檔案排除資料夾 & path.suffix.lower()取副檔名並轉小寫
    ) # 排序，讓每次的順序一致
    if not image_paths:
        raise FileNotFoundError(
            f"No supported aligned images found in {ALIGNED_DIR}. "
            "Run 04_alignment.py first."
        ) # not image_paths 在清單是空的時為真，這時主動丟出錯誤，避免後面對「0 張圖」默默跑完卻沒有輸出。

    preview_by_folder = {}  # 用來收集每個資料夾的預覽圖像，key 是資料夾名稱，value 是 tuple (image_name, aligned, difference, defect_mask)
    failures = []           # 用來收集失敗的訊息
    processed_count = 0     # 記錄處理的圖像數量
    for image_path in image_paths:
        relative_path = image_path.relative_to(ALIGNED_DIR)            # relative_to 取出「相對於 aligned_images 的路徑」，例如 Short/01_short_01.jpg。
        aligned = cv_io.imread(str(image_path), cv2.IMREAD_GRAYSCALE)  # 步驟1+2:讀入 04 對位後的圖（灰階）
        if aligned is None:
            failures.append(f"Could not read aligned image: {image_path}") # 讀不到時回傳 None並記下失敗訊息
            continue                                                       # 用 continue 跳過這張，繼續下一張，不讓整個程式中斷。

        try:
            difference, defect_mask = compute_difference(reference, aligned) # difference_blurred 是經過光照校正後的差異圖像, defect_mask 是二值化的缺陷遮罩
            relative_output = relative_path.with_suffix(".png")              # 把副檔名換成 .png,例如 open_circuit_Rotation/01_open_circuit_01.jpg → open_circuit_Rotation/01_open_circuit_01.png
            difference_path = RESULTS_DIR / "difference" / relative_output   # 接到輸出資料夾後面，所以輸出也保持同樣的資料夾結構：results/05_difference/difference/open_circuit_Rotation/...
            mask_path = RESULTS_DIR / "defect_mask" / relative_output        # 接到輸出資料夾後面，所以輸出也保持同樣的資料夾結構：results/05_difference/defect_mask/open_circuit_Rotation/...
            difference_path.parent.mkdir(parents=True, exist_ok=True)        # 先確保這張圖要放的子資料夾difference存在
            mask_path.parent.mkdir(parents=True, exist_ok=True)              # 先確保這張圖要放的子資料夾defect_mask存在 

            if not cv_io.imwrite(str(difference_path), difference):
                raise OSError(f"Could not save difference image: {difference_path}") # 存差異圖失敗一樣記錄並用continue跳過。
            if not cv_io.imwrite(str(mask_path), defect_mask):
                raise OSError(f"Could not save defect mask: {mask_path}")            # 存缺陷遮罩失敗一樣記錄並用continue跳過。

            preview_by_folder.setdefault(
                relative_path.parent,
                (relative_path.name, aligned, difference, defect_mask),
            ) # 每個資料夾只記第一張(圖名、對位圖、差異、遮罩),最後畫預覽圖用
            processed_count += 1                                # 成功才計數並印出
            print(f"Compared with reference: {relative_path}")
        except (OSError, ValueError, cv2.error) as error:       # 接住三種錯:存檔失敗(OSError)、尺寸不符(compute_difference 開頭丟的 ValueError)、OpenCV 錯誤。記錄後迴圈繼續下一張
            failures.append(f"{relative_path}: {error}")
            print(f"ERROR processing {relative_path}: {error}")

    print(f"\nProcessed {processed_count} of {len(image_paths)} aligned image(s).")
    print(f"Difference images saved under: {RESULTS_DIR / 'difference'}")
    print(f"Defect masks saved under: {RESULTS_DIR / 'defect_mask'}")

    # _______________畫預覽圖：每個資料夾一列、共 4 欄（參考圖、對位圖、差異圖、遮罩）_______________
    if preview_by_folder:  # 字典不是空的（至少有一張成功）才畫
        figure, axes = plt.subplots(
            len(preview_by_folder),                                  # 列數 = 資料夾數（本作業 3 個類別資料夾）
            4,                                                       # 欄數 = 4
            figsize=(16, max(4, 3.5 * len(preview_by_folder))),      # 圖的大小（英吋）：寬 16，高隨列數增加、至少 4
            squeeze=False,                                           # 即使只有 1 列也維持二維陣列，axes[row, col] 的寫法才不會出錯
        )
        for row, (folder, (image_name, aligned, difference, defect_mask)) in enumerate(
            preview_by_folder.items()                                # 取出每個資料夾那張代表圖；row 是列號 0、1、2…
        ):
            for axis, image, title in (                              # 4 組（格子、圖、標題）依序畫在同一列
                (axes[row, 0], reference, "Reference: reference.jpg"),
                (axes[row, 1], aligned, f"Aligned: {image_name}"),
                (axes[row, 2], difference, "Blurred absolute difference"),
                (axes[row, 3], defect_mask, "Adaptive defect mask"),  # 標題寫 Adaptive，但程式實際用固定門檻 30（cv2.THRESH_BINARY），不是自適應門檻
            ):
                axis.imshow(image, cmap="gray")                      # 用灰階色彩對應顯示
                axis.set_title(title)                                # 標題
                axis.axis("off")                                     # 隱藏座標軸
            axes[row, 0].set_ylabel(folder.name)                     # 在該列第一格標上資料夾名稱（但上一步 axis("off") 可能把標籤一起隱藏，我沒開預覽圖確認）

        figure.tight_layout()                                        # 自動調整間距，避免標題重疊
        figure.savefig(PREVIEW_PATH, dpi=150, bbox_inches="tight")   # 存成 difference_preview.jpg，dpi=150、裁掉多餘白邊
        print(f"Difference preview saved to: {PREVIEW_PATH}")

    # _______________最後統一回報失敗_______________
    if failures:                                                     # 有失敗才進來
        for failure in failures:
            print(f"ERROR: {failure}")                               # 逐筆印出錯誤原因
        raise RuntimeError(                                          # 丟出錯誤，讓你知道「不是全部都成功」（前面成功的圖已經存好了）
            f"{len(failures)} aligned image(s) could not be processed."
        )


if __name__ == "__main__":
    main()
