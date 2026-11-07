# =========================================================================
# 07_components.py 概要
#
# 【在做什麼】把 06 清理後的黑白遮罩裡「相連的白色像素」圈成一個個物件（連通元件），
#            過濾太小或太大的，記下每個物件的位置、外框、面積，並輸出彩色標記圖與兩份 CSV。
#            對應 AOI 筆記【步驟11 輪廓/連通元件分析】，並用到步驟12（位置、外框、面積）與步驟13（面積門檻）。
#
# 【kernel 怎麼選】用命令列參數指定要讀 06 的哪一組：
#            python src/07_components.py 3   （讀 kernel_size3）；不給參數就是 5（作者原始設定）。
#
# 【流程】
#   讀 results/06_morphology/kernel_size{N}/cleaned_masks/*.png
#     └→ 二值化確認、connectedComponentsWithStats（8 鄰接）               [步驟11]
#     └→ 過濾：面積 < 20 像素 或 > 半張圖 的元件不算缺陷                  [步驟13]
#     └→ 記錄每個元件的質心、外框（bbox）、寬高、面積                      [步驟12]
#     └→ 輸出：labeled_masks/*.png（每個元件上不同顏色＋白色外框）
#             component_details.csv（每個元件一列）
#             component_summary.csv（每張圖一列，含元件數）
#
# 【注意】CSV 裡的 label（缺陷類別）是由「資料夾名稱」推出來的，不是人工標註缺陷位置，
#        所以不能拿來算 Precision/Recall，也不能把分類準確率說成檢出率。
# 【輸入／輸出】輸入 06 的 cleaned_masks；輸出 results/07_components/kernel_size{N}/。
# =========================================================================
import sys
from pathlib import Path

import cv2

import cv_io
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent                                          # 宣告BASE_DIR紀錄路徑pcb-defect-detection
# Usage: python src/07_components.py [kernel_size]   (default 5 = original setting)
KERNEL_SIZE = int(sys.argv[1]) if len(sys.argv) > 1 else 5                                 # 宣告KERNEL_SIZE紀錄形態學清理的kernel大小，預設為5（原始設定）
RESULT_DIR = BASE_DIR / "results" / "07_components" / f"kernel_size{KERNEL_SIZE}"          # 宣告RESULT_DIR紀錄處理後圖片路徑pcb-defect-detection/results/07_components/kernel_size{KERNEL_SIZE}
CLEAN_MASK_DIR = (
    BASE_DIR / "results" / "06_morphology" / f"kernel_size{KERNEL_SIZE}" / "cleaned_masks" # 宣告CLEAN_MASK_DIR紀錄形態學清理後的缺陷遮罩路徑pcb-defect-detection/results/06_morphology/kernel_size{KERNEL_SIZE}/cleaned_masks
)
OUTPUT_DIR = RESULT_DIR / "labeled_masks"                       # 彩色標記圖的輸出資料夾
DETAILS_CSV_PATH = RESULT_DIR / "component_details.csv"         # 每個元件一列的明細 CSV
SUMMARY_CSV_PATH = RESULT_DIR / "component_summary.csv"         # 每張圖一列的摘要 CSV
SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}  # 允許的圖片格式
DETAIL_COLUMNS = [                  # 明細 CSV 的欄位順序
    "image_name",                   # 圖檔名
    "dataset_category",             # 所在資料夾名稱（如 open_circuit_Rotation）
    "label",                        # 由資料夾名稱推出的缺陷類別（非人工標註）
    "relative_path",                # 相對路徑
    "component_id",                 # 元件編號（同一張圖內從 1 開始）
    "centroid_x",                   # 質心 x
    "centroid_y",                   # 質心 y
    "bbox_x",                       # 外框左上角 x
    "bbox_y",                       # 外框左上角 y
    "width",                        # 外框寬
    "height",                       # 外框高
    "area",                         # 面積（像素數）
]
SUMMARY_COLUMNS = [                 # 摘要 CSV 的欄位順序
    "image_name",
    "dataset_category",
    "label",
    "relative_path",
    "detected_component_count",     # 這張圖過濾後的元件數
    "inspection_status",            # SUCCESS 或 ERROR
    "error",                        # 失敗原因（成功則空字串）
]


def dataset_label(category):
    """Map a rotation-folder name to its base PCB defect class."""
    return category.casefold().removesuffix("_rotation")  # 例如 "open_circuit_Rotation" → "open_circuit_rotation" → "open_circuit"


def analyze_components(mask):
    """Label connected regions and return a color map and component records."""
    binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]         # 步驟9:確認是 0/255 二值圖
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(  # 步驟11:連通元件分析，connectivity=8 即 8 鄰接（斜角相連也算同一個）
        binary, connectivity=8                                           # count=元件數（含背景）、labels=每個像素屬於哪個元件、stats=每個元件的外框與面積、centroids=質心
    )
    labeled_image = cv2.applyColorMap(                                   # 每個元件上不同顏色，方便肉眼檢視
        np.uint8(labels % 256), cv2.COLORMAP_TURBO                       # 元件編號對 256 取餘數後上色
    )
    labeled_image[labels == 0] = 0                                       # 背景（編號 0）塗成黑色

    height, width = mask.shape                                           # 遮罩的高、寬
    image_area = height * width                                          # 整張圖的像素數，下面用來判斷「太大」
    components = []                                                      # 收集通過過濾的元件
    for component_index in range(1, count):                              # 從 1 開始，因為 0 是背景
        area = int(stats[component_index, cv2.CC_STAT_AREA])             # 這個元件的面積（像素數）
        if area < 20 or area > image_area * 0.5:  # 步驟13:面積門檻，< 20 像素或 > 半張圖的元件不算缺陷
            continue

        x = int(stats[component_index, cv2.CC_STAT_LEFT])  # 步驟12:取位置與外框（bbox）、質心、面積
        y = int(stats[component_index, cv2.CC_STAT_TOP])
        box_width = int(stats[component_index, cv2.CC_STAT_WIDTH])
        box_height = int(stats[component_index, cv2.CC_STAT_HEIGHT])
        center_x, center_y = centroids[component_index]
        components.append(
            {
                "component_id": len(components) + 1,                     # 編號只算通過過濾的，從 1 開始
                "centroid_x": round(float(center_x), 2),
                "centroid_y": round(float(center_y), 2),
                "bbox_x": x,
                "bbox_y": y,
                "width": box_width,
                "height": box_height,
                "area": area,
            }
        )
        cv2.rectangle(                                                   # 在標記圖上畫白色外框（粗 2）
            labeled_image,
            (x, y),
            (x + box_width, y + box_height),
            (255, 255, 255),
            2,
        )
    return labeled_image, components                                     # 回傳：彩色標記圖、元件資料清單


def main():
    if not CLEAN_MASK_DIR.is_dir():
        raise FileNotFoundError(
            f"Cleaned masks folder not found: {CLEAN_MASK_DIR}. "
            "Run 06_morphology.py first."
        ) # 06 的輸出資料夾不存在 → 提示先跑 06

    # _______________收集 06 輸出的遮罩檔_______________
    mask_paths = sorted(
        path
        for path in CLEAN_MASK_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    ) # 排序，讓每次的順序一致
    if not mask_paths:
        raise FileNotFoundError(f"No cleaned masks found in {CLEAN_MASK_DIR}")

    details = []    # 收集所有圖所有元件的明細
    summaries = []  # 收集每張圖的摘要
    failures = []   # 收集失敗訊息
    for mask_path in mask_paths:
        relative_mask_path = mask_path.relative_to(CLEAN_MASK_DIR)           # 相對於 cleaned_masks 的路徑
        relative_image_path = relative_mask_path.with_suffix(".jpg")         # 還原成原圖檔名（.png → .jpg），寫進 CSV 方便對應
        category = relative_mask_path.parent.name                            # 所在資料夾名稱
        label = dataset_label(relative_mask_path.parent.name)                # 由資料夾名稱推出缺陷類別
        summary = {                                                          # 先建立這張圖的摘要，預設成功、元件數 0
            "image_name": relative_image_path.name,
            "dataset_category": category,
            "label": label,
            "relative_path": str(relative_image_path),
            "detected_component_count": 0,
            "inspection_status": "SUCCESS",
            "error": "",
        }

        # _______________讀圖並檢查_______________
        mask = cv_io.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)            # 步驟1:讀入 06 清理後的遮罩（灰階）
        if mask is None:
            summary["inspection_status"] = "ERROR"                           # 失敗也要在摘要留一列，並標 ERROR
            summary["error"] = f"Could not read mask: {mask_path}"
            failures.append(summary["error"])
            summaries.append(summary)
            continue

        # _______________分析並存檔_______________
        labeled_image, components = analyze_components(mask)                 # 步驟11～13:連通元件分析＋量測＋面積過濾
        output_path = OUTPUT_DIR / relative_mask_path.with_suffix(".png")    # 彩色標記圖的輸出位置
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv_io.imwrite(str(output_path), labeled_image):
            summary["inspection_status"] = "ERROR"
            summary["error"] = f"Could not save component image: {output_path}"
            failures.append(summary["error"])
            summaries.append(summary)
            continue

        summary["detected_component_count"] = len(components)               # 記下這張圖通過過濾的元件數
        details.extend(                                                      # 把每個元件加上圖檔資訊後，加進明細清單
            {
                "image_name": relative_image_path.name,
                "dataset_category": category,
                "label": label,
                "relative_path": str(relative_image_path),
                **component,                                                 # 展開元件的位置、外框、面積等欄位
            }
            for component in components
        )
        summaries.append(summary)
        print(f"{relative_image_path}: {len(components)} component(s)")

    # _______________存兩份 CSV_______________
    pd.DataFrame(details, columns=DETAIL_COLUMNS).to_csv(
        DETAILS_CSV_PATH, index=False, encoding="utf-8-sig"                  # utf-8-sig：用 Excel 開啟中文不會亂碼
    )
    pd.DataFrame(summaries, columns=SUMMARY_COLUMNS).to_csv(
        SUMMARY_CSV_PATH, index=False, encoding="utf-8-sig"
    )
    print(f"\nAnalyzed {len(summaries) - len(failures)} of {len(mask_paths)} mask(s).")
    print(f"Component label images saved under: {OUTPUT_DIR}")
    print(f"Component details saved to: {DETAILS_CSV_PATH}")
    print(f"Per-image component summary saved to: {SUMMARY_CSV_PATH}")

    # _______________最後統一回報失敗_______________
    if failures:
        for failure in failures:
            print(f"ERROR: {failure}")
        raise RuntimeError(f"{len(failures)} mask(s) could not be analyzed.")


if __name__ == "__main__":
    main()
