# =========================================================================
# 08_features.py 概要
#
# 【在做什麼】對 06 清理後遮罩裡的每個連通元件（疑似缺陷）量測一整組特徵，存成 features.csv，
#            供 09 的 SVM 訓練使用；同時依面積算出一個「嚴重度」。
#            對應 AOI 筆記【步驟11 連通元件】、【步驟12 特徵量測】與【步驟13 判定規則】的一部分。
#
# 【流程】
#   讀 results/06_morphology/kernel_size5/cleaned_masks/*.png   （注意：寫死 kernel 5，不像 07 能換）
#     └→ 連通元件分析（同 07），面積 < 20 或 > 半張圖 的丟掉                [步驟11、步驟13]
#     └→ 對每個元件量測：                                                   [步驟12]
#          位置與外框（質心、bbox、寬高）、面積、長寬比（寬/高）、
#          周長、圓度（4π×面積/周長²，越接近 1 越圓）、
#          Hu 不變矩 7 個（描述形狀，與位置、旋轉、縮放較無關；取 log 壓縮數值範圍）
#     └→ 嚴重度 severity_pct = 面積/5（上限 100），<25 LOW、<65 MEDIUM、其餘 HIGH   [步驟13，作者自訂規則]
#     └→ 輸出：features.csv（每個元件一列）、feature_summary.csv（每張圖一列）
#
# 【注意】
#   1. label 由資料夾名稱推出，不是人工標註缺陷位置；沒有標註就不能算 Precision/Recall/F1。
#   2. 嚴重度規則（面積/5、25、65 的分界）是作者自訂，不是標準。
#   3. 這支沒有 OK/NG 判定。
# 【輸入／輸出】輸入 06 的 kernel_size5 cleaned_masks；輸出 results/08_features/。
# =========================================================================
from pathlib import Path

import cv2

import cv_io
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent                                          # 宣告BASE_DIR紀錄路徑pcb-defect-detection
# Reads the cleaned masks of the original setting (kernel 5x5).
CLEAN_MASK_DIR = BASE_DIR / "results" / "06_morphology" / "kernel_size5" / "cleaned_masks" # 宣告CLEAN_MASK_DIR紀錄形態學清理後的缺陷遮罩路徑pcb-defect-detection/results/06_morphology/kernel_size5/cleaned_masks
FEATURES_CSV_PATH = BASE_DIR / "results" / "08_features" / "features.csv"                  # 宣告FEATURES_CSV_PATH紀錄特徵量測結果的輸出路徑pcb-defect-detection/results/08_features/features.csv
SUMMARY_CSV_PATH = BASE_DIR / "results" / "08_features" / "feature_summary.csv"            # 宣告SUMMARY_CSV_PATH紀錄特徵量測摘要的輸出路徑pcb-defect-detection/results/08_features/feature_summary.csv
SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}            # 允許的圖片格式
FEATURE_COLUMNS = [                 # features.csv 的欄位順序
    "image_name",                   # 圖檔名
    "dataset_category",             # 所在資料夾名稱
    "label",                        # 由資料夾名稱推出的缺陷類別（非人工標註）
    "relative_path",                # 相對路徑
    "defect_id",                    # 元件編號（同一張圖內從 1 開始）
    "centroid_x",                   # 質心 x
    "centroid_y",                   # 質心 y
    "bbox_x",                       # 外框左上角 x
    "bbox_y",                       # 外框左上角 y
    "width",                        # 外框寬
    "height",                       # 外框高
    "area",                         # 面積（像素數）
    "aspect_ratio",                 # 長寬比 = 寬/高
    "perimeter",                    # 周長
    "circularity",                  # 圓度
    "hu1",                          # Hu 不變矩（共 7 個，取 log 後的值）
    "hu2",
    "hu3",
    "hu4",
    "hu5",
    "hu6",
    "hu7",
    "severity_pct",                 # 嚴重度百分比（面積/5，上限 100）
    "severity_label",               # 嚴重度等級 LOW / MEDIUM / HIGH
]
SUMMARY_COLUMNS = [                 # feature_summary.csv 的欄位順序
    "image_name",
    "dataset_category",
    "label",
    "relative_path",
    "detected_defect_count",        # 這張圖的缺陷（元件）數
    "highest_severity",             # 這張圖裡最高的嚴重度等級（沒有缺陷則 NONE）
    "inspection_status",            # SUCCESS 或 ERROR
    "error",                        # 失敗原因
]


def dataset_label(category):
    """Map a rotation-folder name to its base PCB defect class."""
    return category.casefold().removesuffix("_rotation")  # 例如 "open_circuit_Rotation" → "open_circuit"


def extract_features(mask):
    """Calculate geometry, shape descriptors, and severity for each component."""
    binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]         # 步驟9:確認是 0/255 二值圖
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(  # 步驟11:連通元件分析（同 07）
        binary, connectivity=8
    )
    image_area = mask.shape[0] * mask.shape[1]                           # 整張圖像素數（高×寬），用來判斷「太大」
    features = []                                                        # 收集每個元件的特徵

    for component_index in range(1, count):                              # 從 1 開始，0 是背景
        area = int(stats[component_index, cv2.CC_STAT_AREA])
        if area < 20 or area > image_area * 0.5:  # 步驟13:面積門檻（同 07）
            continue

        x = int(stats[component_index, cv2.CC_STAT_LEFT])  # 步驟12:特徵量測（位置、外框、面積）
        y = int(stats[component_index, cv2.CC_STAT_TOP])
        width = int(stats[component_index, cv2.CC_STAT_WIDTH])
        height = int(stats[component_index, cv2.CC_STAT_HEIGHT])
        center_x, center_y = centroids[component_index]
        component_mask = np.uint8(labels == component_index)             # 只留這一個元件（其餘為 0），才能單獨找輪廓
        contours, _ = cv2.findContours(                                  # 找這個元件的外輪廓（RETR_EXTERNAL 只要最外圈）
            component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE   # CHAIN_APPROX_SIMPLE 只存轉折點，省空間
        )
        if not contours:                                                 # 沒找到輪廓就跳過（保險）
            continue

        contour = max(contours, key=cv2.contourArea)                     # 若有多個輪廓，取面積最大的
        perimeter = float(cv2.arcLength(contour, True))  # 步驟12:周長（True＝封閉輪廓）
        circularity = (                                   # 步驟12:圓度 = 4π×面積 / 周長²，越接近 1 越圓
            (4.0 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0.0   # 周長為 0 時避免除以 0
        )
        hu_moments = cv2.HuMoments(cv2.moments(contour)).flatten()  # 步驟12:Hu 不變矩（形狀特徵，與位置/旋轉/縮放較無關），共 7 個
        hu_log = [                                                   # Hu 值通常非常小（如 1e-7），取 -sign×log10(|值|) 壓成容易比較的數值
            -np.sign(value) * np.log10(abs(value)) if value != 0 else 0.0
            for value in hu_moments
        ]
        severity_pct = round(min(100.0, area / 5.0), 2)  # 步驟13:嚴重度 = 面積/5，上限 100（作者自訂規則，非標準）
        if severity_pct < 25.0:                          # 分級：<25 LOW、<65 MEDIUM、其餘 HIGH
            severity_label = "LOW"
        elif severity_pct < 65.0:
            severity_label = "MEDIUM"
        else:
            severity_label = "HIGH"

        features.append(
            {
                "defect_id": len(features) + 1,                          # 編號只算通過過濾的，從 1 開始
                "centroid_x": round(float(center_x), 2),
                "centroid_y": round(float(center_y), 2),
                "bbox_x": x,
                "bbox_y": y,
                "width": width,
                "height": height,
                "area": area,
                "aspect_ratio": round(width / height, 4) if height else 0.0,   # 長寬比 = 寬/高，高為 0 時給 0
                "perimeter": round(perimeter, 2),
                "circularity": round(float(circularity), 5),
                **{                                                      # 展開成 hu1～hu7 七個欄位
                    f"hu{index + 1}": round(float(value), 7)
                    for index, value in enumerate(hu_log)
                },
                "severity_pct": severity_pct,
                "severity_label": severity_label,
            }
        )
    return features                                                      # 回傳這張圖所有元件的特徵清單


def main():
    if not CLEAN_MASK_DIR.is_dir():
        raise FileNotFoundError(
            f"Cleaned masks folder not found: {CLEAN_MASK_DIR}. "
            "Run 06_morphology.py first."
        ) # 06 的 kernel_size5 輸出不存在 → 提示先跑 06

    # _______________收集 06 輸出的遮罩檔_______________
    mask_paths = sorted(
        path
        for path in CLEAN_MASK_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    ) # 排序，讓每次的順序一致
    if not mask_paths:
        raise FileNotFoundError(f"No cleaned masks found in {CLEAN_MASK_DIR}")

    feature_rows = []   # 收集所有圖所有元件的特徵列
    summary_rows = []   # 收集每張圖的摘要
    failures = []       # 收集失敗訊息
    for mask_path in mask_paths:
        relative_image_path = mask_path.relative_to(CLEAN_MASK_DIR).with_suffix(".jpg")   # 相對路徑，並還原成原圖副檔名 .jpg
        category = relative_image_path.parent.name                                         # 所在資料夾名稱
        label = dataset_label(category)                                                    # 由資料夾名稱推出缺陷類別
        summary = {                                                                        # 先建立這張圖的摘要：預設 0 個缺陷、嚴重度 NONE、成功
            "image_name": relative_image_path.name,
            "dataset_category": category,
            "label": label,
            "relative_path": str(relative_image_path),
            "detected_defect_count": 0,
            "highest_severity": "NONE",
            "inspection_status": "SUCCESS",
            "error": "",
        }
        mask = cv_io.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)       # 步驟1:讀入 06 清理後的遮罩（灰階）
        if mask is None:
            summary["inspection_status"] = "ERROR"
            summary["error"] = f"Could not read cleaned mask: {mask_path}"
            failures.append(summary["error"])
            summary_rows.append(summary)
            continue

        features = extract_features(mask)                               # 步驟11～13:量測這張圖所有元件
        summary["detected_defect_count"] = len(features)
        if features:                                                    # 有缺陷才找最高嚴重度
            severity_rank = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}          # 等級排序用的數字
            summary["highest_severity"] = max(
                (feature["severity_label"] for feature in features),
                key=severity_rank.get,                                  # 依上面的數字比大小，取最嚴重的等級
            )
        feature_rows.extend(                                            # 把每個元件加上圖檔資訊後，加進總清單
            {
                "image_name": relative_image_path.name,
                "dataset_category": category,
                "label": label,
                "relative_path": str(relative_image_path),
                **feature,                                              # 展開元件的所有特徵欄位
            }
            for feature in features
        )
        summary_rows.append(summary)
        print(f"{relative_image_path}: extracted {len(features)} defect feature(s)")

    # _______________存兩份 CSV_______________
    FEATURES_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)         # 先確保 results/08_features 資料夾存在
    pd.DataFrame(feature_rows, columns=FEATURE_COLUMNS).to_csv(
        FEATURES_CSV_PATH, index=False, encoding="utf-8-sig"            # utf-8-sig：用 Excel 開啟中文不會亂碼
    )
    pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS).to_csv(
        SUMMARY_CSV_PATH, index=False, encoding="utf-8-sig"
    )
    print(
        f"\nExtracted features from {len(summary_rows) - len(failures)} "
        f"of {len(mask_paths)} image(s), producing {len(feature_rows)} defect row(s)."
    )
    print(f"Feature dataset saved to: {FEATURES_CSV_PATH}")
    print(f"Per-image feature summary saved to: {SUMMARY_CSV_PATH}")

    # _______________最後統一回報失敗_______________
    if failures:
        for failure in failures:
            print(f"ERROR: {failure}")
        raise RuntimeError(f"{len(failures)} mask(s) could not be analyzed.")


if __name__ == "__main__":
    main()
