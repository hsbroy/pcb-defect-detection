# =========================================================================
# 02_preprocess.py 概要
#
# 【在做什麼】對資料集每張彩色圖做最基本的前處理：轉灰階 → 中值濾波 → 高斯濾波，把雜訊壓掉、畫面變平滑。
#            對應 AOI 筆記【步驟2 轉灰階】與【步驟3 去雜訊(平滑)】。
#
# 【流程】
#   讀 dataset/ 底下所有圖（含各缺陷類別資料夾，約 112 張）                 [步驟1 影像輸入]
#     └→ cvtColor：BGR 彩色 → 灰階                                         [步驟2]
#     └→ medianBlur(5)：中值濾波，去掉孤立亮/暗點（椒鹽雜訊）                 [步驟3]
#     └→ GaussianBlur(5×5, σ=1)：高斯濾波，再整體平滑                       [步驟3]
#     └→ 存 results/02_preprocess/（保持與 dataset 相同的資料夾結構）
#
# 【注意】這支是「支線」：輸出沒有被 04～10 使用（主線 04 直接讀原始彩色圖）。
#        kernel 越大越平滑，但小缺陷也越容易被抹掉（你的筆記步驟3 的提醒）。
# 【輸入／輸出】輸入 dataset/；輸出 results/02_preprocess/。
# =========================================================================
from pathlib import Path

import cv2

import cv_io

# Resolve input and output locations relative to the project root.
BASE_DIR = Path(__file__).resolve().parent.parent       # 宣告BASE_DIR紀錄路徑pcb-defect-detection
DATASET_DIR = BASE_DIR / "dataset"                      # 宣告DATASET_DIR紀錄數據集路徑pcb-defect-detection/dataset
PROCESSED_DIR = BASE_DIR / "results" / "02_preprocess"  # 宣告PROCESSED_DIR紀錄處理後圖片路徑pcb-defect-detection/results/02_preprocess
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"} # 支援的圖像檔案副檔名集合


def preprocess_image(image):
    """Convert an image to grayscale and apply median and Gaussian filters."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 步驟2:轉成灰階圖
    median = cv2.medianBlur(gray, 5)                # 步驟3:使用中值濾波器去除雜訊，濾波器大小為 5
    return cv2.GaussianBlur(median, (5, 5), 1)      # 步驟3:使用高斯濾波器進一步平滑圖像，濾波器大小為 5x5，標準差為 1


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

    # _______________建立輸出資料夾______________
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True) # 建立 results/02_preprocess/。parents=True 表示上層不存在也一併建立；exist_ok=True 表示已存在也不報錯。
    failures = [] # 是一個空清單，用來收集失敗的訊息    
    for image_path in image_paths:
        #  _______________對每張圖，決定輸出位置 _______________  
        relative_path = image_path.relative_to(DATASET_DIR) # relative_to 取出「相對於 dataset 的路徑」，例如 Short/01_short_01.jpg。
        output_path = PROCESSED_DIR / relative_path # 接到輸出資料夾後面，所以輸出也保持同樣的資料夾結構：results/02_preprocess/Short/01_short_01.jpg。
        # _______________讀圖並檢查_______________
        image = cv_io.imread(str(image_path), cv2.IMREAD_COLOR) # 彩色讀入（IMREAD_COLOR）。
        if image is None:
            failures.append(f"Could not read image: {image_path}") # 讀不到時回傳 None並記下失敗訊息
            continue # 用 continue 跳過這張，繼續下一張，不讓整個程式中斷。
        # _______________處理並存檔_______________
        processed = preprocess_image(image) # 灰階→中值→高斯
        output_path.parent.mkdir(parents=True, exist_ok=True) # 先確保這張圖要放的子資料夾存在
        if not cv_io.imwrite(str(output_path), processed): # imwrite 成功回傳 True，失敗是 False
            failures.append(f"Could not save processed image: {output_path}") # 失敗一樣記錄並用continue跳過。
            continue

        print(f"Processed: {relative_path}")

    processed_count = len(image_paths) - len(failures) # 成功數 = 總數 − 失敗數。這就是你看到的 Processed 112 of 112。
    print(f"\nProcessed {processed_count} of {len(image_paths)} image(s).")
    print(f"Processed images saved under: {PROCESSED_DIR}")
    if failures: # 如果有失敗，把每一筆錯誤印出來，最後丟出 RuntimeError，讓你知道「不是全部都成功」。
        for failure in failures:
            print(f"ERROR: {failure}")
        raise RuntimeError(f"{len(failures)} image(s) could not be processed.")


if __name__ == "__main__":
    main()
