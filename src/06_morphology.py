# =========================================================================
# 06_morphology.py 概要
#
# 【在做什麼】05 產生的缺陷遮罩（白＝疑似缺陷）還有很多細碎雜點與小缺口，
#            這支用「形態學」清理：opening 去掉小雜點、closing 補起小洞與斷線，得到乾淨遮罩給 07 使用。
#            對應 AOI 筆記【步驟10 二值化形態學清理】。
#
# 【本作業的實驗】kernel（結構元素）大小 = 3、5、7 三種，作者原始設定是 5。
#            每種大小各自輸出一個資料夾，之後比較「清理後的連通元件數」：
#            （你之前的實測）3×3＝334、5×5＝235、7×7＝226。
#
# 【流程】
#   讀 results/05_difference/defect_mask/*.png（60 張，0/255 二值圖）
#     └→ 確保是 0/255，必要時反轉（白色過多代表前景背景顛倒）         [步驟9 之後的保險]
#     └→ opening（先侵蝕再膨脹）：去掉比 kernel 小的白色雜點             [步驟10]
#     └→ closing（先膨脹再侵蝕）：補起小黑洞、接起細小斷線               [步驟10]
#     └→ 存 results/06_morphology/kernel_size{3,5,7}/cleaned_masks/*.png
#     └→ 存預覽圖 morphology_preview.jpg（每個類別一列：原遮罩、opening 後、清理後）
#
# 【輸入／輸出】輸入 05 的 defect_mask；輸出 06_morphology/kernel_size{N}/，供 07 讀取。
# =========================================================================
from pathlib import Path

import cv2

import cv_io
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent                     # 宣告BASE_DIR紀錄路徑pcb-defect-detection
RAW_MASK_DIR = BASE_DIR / "results" / "05_difference" / "defect_mask" # 宣告RAW_MASK_DIR紀錄原始缺陷遮罩路徑pcb-defect-detection/results/05_difference/defect_mask
RESULTS_DIR = BASE_DIR / "results" / "06_morphology"                  # 宣告RESULTS_DIR紀錄處理後圖片路徑pcb-defect-detection/results/06_morphology
# [Experiment parameter] Kernel sizes to compare. 5 is the author's original setting.
KERNEL_SIZES = (3, 5, 7) # 實驗參數kernel=3,5,7，5是作者原始設定
SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"} # 允許的圖片格式


def clean_mask(raw_mask, ksize=5):
    """Remove small isolated regions and close small gaps in a binary mask."""
    binary = cv2.threshold(raw_mask, 127, 255, cv2.THRESH_BINARY)[1]           # 步驟9:確保是 0/255 的二值圖（>127 → 255，其餘 0）
    if cv2.countNonZero(binary) > binary.size * 0.5:                           # 白色超過一半 → 前景/背景可能顛倒，反轉
        binary = cv2.bitwise_not(binary)                                       # 0 與 255 對調
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))      # 步驟10:橢圓形 kernel，大小 = ksize（本作業的實驗參數）
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)    # 步驟10:opening 去雜點（先侵蝕再膨脹，比 kernel 小的白點會消失）
    cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)  # 步驟10:closing 補洞接線（先膨脹再侵蝕，小黑洞與細縫會被填起來）
    return opened, cleaned                                                     # 回傳兩張：opening 之後、再 closing 之後（最終清理結果）


def run(ksize):
    # _______________決定這個 kernel 大小的輸出位置_______________
    output_dir = RESULTS_DIR / f"kernel_size{ksize}" / "cleaned_masks"                # 例如 results/06_morphology/kernel_size5/cleaned_masks
    preview_path = RESULTS_DIR / f"kernel_size{ksize}" / "morphology_preview.jpg"      # 例如 results/06_morphology/kernel_size5/morphology_preview.jpg
    if not RAW_MASK_DIR.is_dir():
        raise FileNotFoundError(
            f"Raw defect mask folder not found: {RAW_MASK_DIR}. "
            "Run 05_difference.py first."
        ) # 05 的遮罩資料夾不存在 → 提示先跑 05

    # _______________收集 05 輸出的遮罩檔_______________
    mask_paths = sorted(
        path
        for path in RAW_MASK_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS  # 只要檔案、且副檔名是圖片
    ) # 排序，讓每次的順序一致
    if not mask_paths:
        raise FileNotFoundError(f"No defect masks found in {RAW_MASK_DIR}") # 空清單時主動報錯，避免默默跑完卻沒有輸出

    preview_by_folder = {}  # 每個資料夾的預覽圖：key=資料夾、value=(原遮罩, opening 後, 清理後)
    failures = []           # 收集失敗訊息，最後統一回報
    processed_count = 0     # 成功處理的張數
    for mask_path in mask_paths:
        relative_path = mask_path.relative_to(RAW_MASK_DIR)                   # 相對於 defect_mask 的路徑，例如 open_circuit_Rotation/01_open_circuit_01.png
        raw_mask = cv_io.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)         # 步驟1:讀入 05 的遮罩（灰階讀入，內容是 0/255）
        if raw_mask is None:
            failures.append(f"Could not read mask: {mask_path}") # 讀不到就記錄
            continue                                             # 跳過這張，繼續下一張

        # _______________清理並存檔_______________
        opened, cleaned = clean_mask(raw_mask, ksize)                         # 步驟10:形態學清理（本作業實驗的核心）
        output_path = output_dir / relative_path.with_suffix(".png")          # 輸出存成 PNG（無損，遮罩的 0/255 才不會被壓縮改變）
        output_path.parent.mkdir(parents=True, exist_ok=True)                 # 先確保子資料夾存在
        if not cv_io.imwrite(str(output_path), cleaned):                      # 存的是「最終清理結果」cleaned（不是 opened）
            failures.append(f"Could not save cleaned mask: {output_path}")
            continue

        preview_by_folder.setdefault(relative_path.parent, (raw_mask, opened, cleaned))  # 每個資料夾只記第一張，畫預覽圖用
        processed_count += 1
        print(f"Cleaned mask: {relative_path}")

    print(f"\nCleaned {processed_count} of {len(mask_paths)} mask(s).")
    print(f"Cleaned masks saved under: {output_dir}")

    # _______________畫預覽圖：每個資料夾一列、3 欄（原遮罩、opening、清理後）_______________
    if preview_by_folder:
        figure, axes = plt.subplots(
            len(preview_by_folder),                                   # 列數 = 資料夾數
            3,                                                        # 欄數 = 3
            figsize=(14, max(4, 3.5 * len(preview_by_folder))),       # 圖大小（英吋）
            squeeze=False,                                            # 只有 1 列時也維持二維陣列
        )
        for row, (folder, (raw_mask, opened, cleaned)) in enumerate(
            preview_by_folder.items()
        ):
            for axis, image, title in (
                (axes[row, 0], raw_mask, "Raw defect mask"),          # 05 輸出的原始遮罩
                (axes[row, 1], opened, "Morphological opening"),      # 只做 opening（去雜點）之後
                (axes[row, 2], cleaned, "Cleaned mask"),              # 再 closing（補洞接線）之後 = 最終輸出
            ):
                axis.imshow(image, cmap="gray")
                axis.set_title(title)
                axis.axis("off")
            axes[row, 0].set_ylabel(folder.name)                      # 資料夾名稱（axis off 之後可能不顯示，我沒開圖確認）

        figure.tight_layout()
        figure.savefig(preview_path, dpi=150, bbox_inches="tight")
        print(f"Morphology preview saved to: {preview_path}")

    # _______________最後統一回報失敗_______________
    if failures:
        for failure in failures:
            print(f"ERROR: {failure}")
        raise RuntimeError(f"{len(failures)} mask(s) could not be cleaned.")


def main():
    for ksize in KERNEL_SIZES:                       # 依序跑 kernel = 3、5、7（每種各產生一組輸出資料夾）
        print(f"\n=== Kernel size {ksize}x{ksize} ===")
        run(ksize)


if __name__ == "__main__":
    main()
