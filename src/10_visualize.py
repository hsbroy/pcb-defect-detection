# =========================================================================
# 10_visualize.py 概要
#
# 【在做什麼】把 09 預測好的缺陷資料畫回「對位後的彩色圖」上：紅框框住缺陷、綠點標質心、文字寫編號與嚴重度，
#            再輸出每張圖的檢測摘要 CSV 與一張預覽圖。這是整個作者流程的最後一步。
#            對應 AOI 筆記【步驟13 標出缺陷位置】（部分）與【④輸出】。
#            ※ 步驟 13 的「OK/NG 判定」作者沒有實作：這支只畫出缺陷，沒有「這片板子是良品或不良品」的結論。
#
# 【流程】
#   讀 results/09_train_svm/features_with_severity.csv（每個缺陷一列，含預測類型）
#   讀 results/04_alignment/aligned_images/*.jpg（對位後的彩色圖，60 張）
#     └→ 依圖路徑把缺陷資料分組（groupby relative_path）
#     └→ 每張圖：統計缺陷數、最高嚴重度、預測類型（該圖元件預測中最常出現的）   [④輸出：摘要]
#     └→ annotate_image：在圖上畫紅框、綠點、文字「#編號 嚴重度 百分比」       [步驟13 標出缺陷位置]
#     └→ 存 results/10_visualize/annotated_images/*.jpg
#     └→ 存 inspection_summary.csv（每張圖一列）與 final_inspection_preview.jpg   [④輸出]
#
# 【注意】
#   1. 預測類型（predicted_defect_type）來自 09 用全部資料訓練的模型，在訓練資料上預測，不能當準確率；
#      label 又來自資料夾名稱，不是人工標註，所以這裡「看起來都對」不代表檢出率。
#   2. 沒有缺陷的圖，摘要的缺陷數為 0、預測類型為 NONE，圖上不會有任何框。
# 【輸入／輸出】輸入 04 的對位圖＋09 的特徵預測表；輸出 results/10_visualize/。
# =========================================================================
from pathlib import Path

import cv2

import cv_io
import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent                                        # 宣告BASE_DIR紀錄路徑pcb-defect-detection
ALIGNED_DIR = BASE_DIR / "results" / "04_alignment" / "aligned_images"                   # 宣告ALIGNED_DIR紀錄對齊後圖像的輸出路徑pcb-defect-detection/results/04_alignment/aligned_images
FEATURES_CSV_PATH = BASE_DIR / "results" / "09_train_svm" / "features_with_severity.csv" # 宣告FEATURES_CSV_PATH紀錄特徵量測結果與預測缺陷類型的輸出路徑pcb-defect-detection/results/09_train_svm/features_with_severity.csv
OUTPUT_DIR = BASE_DIR / "results" / "10_visualize" / "annotated_images"                  # 宣告OUTPUT_DIR紀錄標註後圖像的輸出路徑pcb-defect-detection/results/10_visualize/annotated_images
SUMMARY_CSV_PATH = BASE_DIR / "results" / "10_visualize" / "inspection_summary.csv"      # 宣告SUMMARY_CSV_PATH紀錄檢測摘要的輸出路徑pcb-defect-detection/results/10_visualize/inspection_summary.csv
PREVIEW_PATH = BASE_DIR / "results" / "10_visualize" / "final_inspection_preview.jpg"    # 宣告PREVIEW_PATH紀錄檢測預覽圖像的輸出路徑pcb-defect-detection/results/10_visualize/final_inspection_preview.jpg
REFERENCE_PATH = BASE_DIR / "reference.jpg"                                              # 宣告REFERENCE_PATH紀錄參考圖像的路徑pcb-defect-detection/reference.jpg
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}          # 允許的圖片格式
REQUIRED_FEATURE_COLUMNS = {        # 特徵表至少要有的欄位，缺了就停止
    "relative_path",                # 對應哪張圖
    "defect_id",                    # 缺陷編號
    "centroid_x",                   # 質心（畫綠點）
    "centroid_y",
    "bbox_x",                       # 外框（畫紅框）
    "bbox_y",
    "width",
    "height",
    "severity_pct",                 # 嚴重度百分比與等級（寫在文字裡）
    "severity_label",
    "predicted_defect_type",        # 09 預測的缺陷類型
}
SUMMARY_COLUMNS = [                 # inspection_summary.csv 的欄位順序
    "image_name",
    "dataset_category",
    "label",                        # 由資料夾名稱推出的類別（非人工標註）
    "relative_path",
    "detected_defect_count",        # 這張圖的缺陷數
    "highest_severity",             # 最高嚴重度（沒有缺陷則 NONE）
    "predicted_defect_type",        # 這張圖的預測缺陷類型（沒有缺陷則 NONE）
    "inspection_status",            # SUCCESS 或 ERROR
    "error",                        # 失敗原因
]


def annotate_image(aligned, image_features):
    """Draw detected defect boxes and labels on an aligned PCB image."""
    output = aligned.copy()  # 步驟13（部分）:標出缺陷位置；OK/NG 判定在此作者流程中未實作（複製一份，不改到原圖）
    for defect in image_features.itertuples(index=False):                     # 逐列取出這張圖的每個缺陷
        x = int(defect.bbox_x)                                                # 外框左上角 x
        y = int(defect.bbox_y)                                                # 外框左上角 y
        width = int(defect.width)                                             # 外框寬
        height = int(defect.height)                                           # 外框高
        center = (int(float(defect.centroid_x)), int(float(defect.centroid_y)))  # 質心座標（取整數）
        color = (0, 0, 255)                                                   # OpenCV 是 BGR 順序，(0,0,255) 是紅色
        cv2.rectangle(output, (x, y), (x + width, y + height), color, 3)  # 紅框＝缺陷外框（bbox）
        cv2.circle(output, center, 5, (0, 255, 0), -1)                    # 綠點＝質心
        text = (                                                              # 標註文字，例如「#3 LOW 8.4%」
            f"#{int(defect.defect_id)} {defect.severity_label} "
            f"{float(defect.severity_pct):.1f}%"
        )
        cv2.putText(                                                          # 把文字寫在外框上方
            output,
            text,
            (x, max(y - 8, 20)),                                              # 文字位置：框上方 8 像素，至少離上緣 20（避免寫出畫面）
            cv2.FONT_HERSHEY_SIMPLEX,                                         # 字型
            0.55,                                                             # 字大小
            color,                                                            # 紅色
            2,                                                                # 線寬
            cv2.LINE_AA,                                                      # 抗鋸齒，字邊緣較平滑
        )
    return output                                                             # 回傳畫好的圖


def main():
    # _______________檢查輸入是否齊全_______________
    if not ALIGNED_DIR.is_dir():
        raise FileNotFoundError(
            f"Aligned images folder not found: {ALIGNED_DIR}. "
            "Run 04_alignment.py first."
        ) # 04 的對位圖不存在 → 提示先跑 04
    if not FEATURES_CSV_PATH.is_file():
        raise FileNotFoundError(
            f"Trained feature report not found: {FEATURES_CSV_PATH}. "
            "Run 08_features.py and 09_train_svm.py first."
        ) # 09 的特徵預測表不存在 → 提示先跑 08、09
    reference = cv_io.imread(str(REFERENCE_PATH), cv2.IMREAD_COLOR)           # 步驟1:讀入參考圖（彩色），預覽圖與尺寸檢查用
    if reference is None:
        raise FileNotFoundError(f"Reference image not found: {REFERENCE_PATH}")

    # _______________讀特徵預測表並依圖分組_______________
    features = pd.read_csv(FEATURES_CSV_PATH)                                 # 讀 09 的 features_with_severity.csv（每個缺陷一列）
    missing_columns = sorted(REQUIRED_FEATURE_COLUMNS.difference(features.columns))
    if missing_columns:
        raise ValueError(
            f"Feature report is missing required columns: {', '.join(missing_columns)}"
        )
    features_by_image = {                                                     # 依圖路徑分組：key=圖路徑、value=該圖所有缺陷列
        relative_path: image_features
        for relative_path, image_features in features.groupby("relative_path")
    }

    # _______________收集對位圖_______________
    image_paths = sorted(
        path
        for path in ALIGNED_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    ) # 排序，讓每次的順序一致
    if not image_paths:
        raise FileNotFoundError(f"No aligned images found in {ALIGNED_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    preview_by_category = {}   # 每個類別的預覽圖：key=類別、value=(圖名, 對位圖, 標註圖)
    summaries = []             # 每張圖的摘要
    failures = []              # 失敗訊息
    processed_count = 0        # 成功張數
    for image_path in image_paths:
        relative_path = image_path.relative_to(ALIGNED_DIR)                   # 相對於 aligned_images 的路徑
        relative_key = str(relative_path)                                     # 轉成字串，用來和特徵表的 relative_path 對應
        category = relative_path.parent.name                                  # 所在資料夾名稱（類別）
        image_features = features_by_image.get(                               # 取出這張圖的缺陷資料；沒有缺陷的圖就給空表
            relative_key, features.iloc[0:0]
        )
        summary = {                                                           # 先建立這張圖的摘要：預設成功、最高嚴重度與預測類型 NONE
            "image_name": image_path.name,
            "dataset_category": category,
            "label": (                                                        # 有缺陷時用特徵表的 label，否則由資料夾名稱推出
                str(image_features["label"].iloc[0])
                if "label" in image_features.columns and not image_features.empty
                else category.casefold().removesuffix("_rotation")
            ),
            "relative_path": relative_key,
            "detected_defect_count": len(image_features),
            "highest_severity": "NONE",
            "predicted_defect_type": "NONE",
            "inspection_status": "SUCCESS",
            "error": "",
        }

        if not image_features.empty:                                          # 這張圖有缺陷才統計
            severity_rank = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}
            summary["highest_severity"] = max(                                # 最高嚴重度＝等級數字最大的
                image_features["severity_label"].astype(str),
                key=severity_rank.get,
            )
            summary["predicted_defect_type"] = str(                           # 預測類型＝該圖所有缺陷預測中最常出現的（眾數）
                image_features["predicted_defect_type"].mode().iloc[0]
            )

        # _______________讀對位圖並檢查_______________
        aligned = cv_io.imread(str(image_path), cv2.IMREAD_COLOR)             # 步驟1:讀入 04 的對位圖（彩色，畫紅框才看得到顏色）
        if aligned is None:
            summary["inspection_status"] = "ERROR"
            summary["error"] = f"Could not read aligned image: {image_path}"
            failures.append(summary["error"])
            summaries.append(summary)
            continue
        if aligned.shape[:2] != reference.shape[:2]:                          # 對位圖尺寸必須和參考圖相同，否則座標對不上
            summary["inspection_status"] = "ERROR"
            summary["error"] = (
                f"Aligned image dimensions {aligned.shape[:2]} do not match "
                f"reference dimensions {reference.shape[:2]}."
            )
            failures.append(f"{relative_key}: {summary['error']}")
            summaries.append(summary)
            continue

        # _______________畫標註並存檔_______________
        annotated = annotate_image(aligned, image_features)  # ④輸出:畫上缺陷框、編號、嚴重度後存檔
        output_path = OUTPUT_DIR / relative_path                              # 輸出位置，保持相同資料夾結構
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv_io.imwrite(str(output_path), annotated):
            summary["inspection_status"] = "ERROR"
            summary["error"] = f"Could not save annotated image: {output_path}"
            failures.append(summary["error"])
            summaries.append(summary)
            continue

        preview_by_category.setdefault(                                       # 每個類別只記第一張，畫預覽圖用
            category, (image_path.name, aligned, annotated)
        )
        summaries.append(summary)
        processed_count += 1
        print(
            f"Visualized {relative_key}: {summary['detected_defect_count']} "
            f"defect(s), predicted class {summary['predicted_defect_type']}"
        )

    # ④輸出:每張圖的檢測摘要（缺陷數、最高嚴重度、預測類型）存成 CSV
    pd.DataFrame(summaries, columns=SUMMARY_COLUMNS).to_csv(
        SUMMARY_CSV_PATH, index=False, encoding="utf-8-sig"
    )
    print(f"\nCreated annotated output for {processed_count} of {len(image_paths)} images.")
    print(f"Annotated images saved under: {OUTPUT_DIR}")
    print(f"Per-image inspection summary saved to: {SUMMARY_CSV_PATH}")

    # _______________畫預覽圖：每個類別一列、3 欄（參考圖、對位圖、標註圖）_______________
    if preview_by_category:
        figure, axes = plt.subplots(
            len(preview_by_category),                                         # 列數 = 類別數
            3,                                                                # 欄數 = 3
            figsize=(15, max(4, 3.5 * len(preview_by_category))),             # 圖大小（英吋）
            squeeze=False,                                                    # 只有 1 列時也維持二維陣列
        )
        for row, (category, (image_name, aligned, annotated)) in enumerate(
            preview_by_category.items()
        ):
            for axis, image, title in (
                (axes[row, 0], reference, "Reference: reference.jpg"),
                (axes[row, 1], aligned, f"Aligned: {image_name}"),
                (axes[row, 2], annotated, "Detected defects"),
            ):
                axis.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))           # matplotlib 用 RGB，OpenCV 是 BGR，所以要轉色
                axis.set_title(title)
                axis.axis("off")
            axes[row, 0].set_ylabel(category)                                 # 類別名稱（axis off 之後可能不顯示，我沒開圖確認）

        figure.tight_layout()
        figure.savefig(PREVIEW_PATH, dpi=150, bbox_inches="tight")
        print(f"Final inspection preview saved to: {PREVIEW_PATH}")

    # _______________最後統一回報失敗_______________
    if failures:
        for failure in failures:
            print(f"ERROR: {failure}")
        raise RuntimeError(f"{len(failures)} image(s) could not be visualized.")


if __name__ == "__main__":
    main()
