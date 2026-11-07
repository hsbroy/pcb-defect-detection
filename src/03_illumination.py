# =========================================================================
# 03_illumination.py 概要
#
# 【在做什麼】修正「整張圖亮度不均」（有的地方亮、有的地方暗），再用 CLAHE 局部增強對比，
#            每張圖輸出兩個版本：corrected（只校正光照）與 clahe_enhanced（校正後再增強對比）。
#            對應 AOI 筆記【步驟2 轉灰階】與【步驟4 對比增強(CLAHE)】；
#            中間的「光照校正」不在你的 13 步內，標為 (筆記外)。
#
# 【流程】
#   讀 dataset/ 底下所有圖                                                    [步驟1 影像輸入]
#     └→ 轉灰階                                                             [步驟2]
#     └→ 51×51 大 kernel 高斯模糊 → 得到「背景亮度」（細節被抹掉，只剩緩慢變化的亮度）   [(筆記外)光照校正-1]
#     └→ 原圖 ÷ 背景 × 255 → corrected（亮度不均被拉平，平坦處接近 255 即全白）       [(筆記外)光照校正-2]
#     └→ CLAHE（clipLimit=2.0、8×8 區塊）→ clahe_enhanced                       [步驟4]
#     └→ 存 results/03_illumination/corrected/ 與 clahe_enhanced/
#
# 【實測】（Open_circuit/01 一張）兩個版本有 57% 像素數值完全相同，其餘平均只差 7.7 灰階，肉眼很難分辨；
#        corrected 幾乎全白（平均亮度 232）。只測 1 張，不能推到全部。
# 【注意】這支是「支線」：輸出沒有被 04～10 使用。05 另外用 illumination_utils.py 的類似做法（scale=128＋正規化）。
# 【輸入／輸出】輸入 dataset/；輸出 results/03_illumination/corrected、clahe_enhanced。
# =========================================================================
from pathlib import Path

import cv2

import cv_io

# Resolve input and output locations relative to the project root.
BASE_DIR = Path(__file__).resolve().parent.parent           # 宣告BASE_DIR紀錄路徑pcb-defect-detection
DATASET_DIR = BASE_DIR / "dataset"                          # 宣告DATASET_DIR紀錄數據集路徑pcb-defect-detection/dataset
PROCESSED_DIR = BASE_DIR / "results" / "03_illumination"    # 宣告PROCESSED_DIR紀錄處理後圖片路徑pcb-defect-detection/results/03_illumination
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"} # 允許的圖片格式


def correct_illumination(image):
    """Return illumination-corrected and CLAHE-enhanced grayscale images."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)              # 步驟2: 將彩色圖像轉換為灰階圖像
    background = cv2.GaussianBlur(gray, (51, 51), 0)            # (筆記外)光照校正-1: 大 kernel 高斯模糊估計背景亮度（不是步驟3 去雜訊），(51, 51)是 kernel 大小，0表示自動計算標準差
    corrected = cv2.divide(gray, background, scale=255)         # (筆記外)光照校正-2: 原圖 ÷ 背景 × 255，讓亮度不均的地方變平均（平坦處≈255，所以幾乎全白）
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) # 步驟4: 建立 CLAHE（限制對比放大的局部直方圖均衡），clipLimit=2.0 限制放大幅度，8×8 是分塊數
    enhanced = clahe.apply(corrected)                           # 步驟4: 對 corrected 套用 CLAHE，使缺陷和背景的灰度差拉大
    return corrected, enhanced # 回傳兩個圖像: corrected 是照明校正後的灰階圖像, enhanced 是經過CLAHE增強後的圖像


def main():
    if not DATASET_DIR.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {DATASET_DIR}") # 如果資料集資料夾不存在，則拋出 FileNotFoundError

    image_paths = sorted(
        path
        for path in DATASET_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS #  path.is_file()只要檔案排除資料夾 & path.suffix.lower()取副檔名並轉小寫
    ) # 排序，讓每次的順序一致
    if not image_paths:
        raise FileNotFoundError(f"No supported images found in {DATASET_DIR}") # not image_paths 在清單是空的時為真，這時主動丟出錯誤，避免後面對「0 張圖」默默跑完卻沒有輸出。
    
    failures = []
    processed_count = 0
    for image_path in image_paths:
        #  _______________對每張圖，決定輸出位置 _______________
        relative_path = image_path.relative_to(DATASET_DIR) # relative_to 取出「相對於 dataset 的路徑」，例如 Short/01_short_01.jpg。
        # _______________讀圖並檢查_______________
        image = cv_io.imread(str(image_path), cv2.IMREAD_COLOR) # 彩色讀入（IMREAD_COLOR）。
        if image is None:
            failures.append(f"Could not read image: {image_path}") # # 讀不到時回傳 None並記下失敗訊息
            continue # 用 continue 跳過這張，繼續下一張，不讓整個程式中斷。
        # _______________處理並存檔_______________
        corrected, enhanced = correct_illumination(image) # corrected 是照明校正後的灰階圖像, enhanced 是經過CLAHE增強後的圖像
        for stage, processed in (("corrected", corrected), ("clahe_enhanced", enhanced)): # 這裡使用 tuple unpacking 將 stage 和 processed 分別對應到 "corrected" 與 corrected 圖像，以及 "clahe_enhanced" 與 enhanced 圖像
            output_path = PROCESSED_DIR / stage / relative_path # 接到輸出資料夾後面，所以輸出也保持同樣的資料夾結構：results/03_illumination/corrected/Short/01_short_01.jpg 或 results/03_illumination/clahe_enhanced/Short/01_short_01.jpg。
            output_path.parent.mkdir(parents=True, exist_ok=True) # 先確保這張圖要放的子資料夾存在
            if not cv_io.imwrite(str(output_path), processed): # imwrite 成功回傳 True，失敗是 False
                failures.append(f"Could not save processed image: {output_path}") # 失敗一樣記錄並用continue跳過。
                break
        else:
            processed_count += 1 # 這裡的 else 對應到 for 迴圈，表示如果沒有遇到 break，則表示兩個階段的圖像都成功保存，processed_count 才會增加。
            print(f"Processed: {relative_path}")

    print(f"\nProcessed {processed_count} of {len(image_paths)} image(s).") # 顯示處理結果
    print(f"Corrected and CLAHE-enhanced images saved under: {PROCESSED_DIR}") #　顯示輸出路徑
    if failures:
        for failure in failures:
            print(f"ERROR: {failure}") # 如果有失敗，把每一筆錯誤印出來，最後丟出 RuntimeError，讓你知道「不是全部都成功」。
        raise RuntimeError(f"{len(failures)} image output(s) could not be processed.")


if __name__ == "__main__":
    main()
