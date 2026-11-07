"""主題 1：Mathematical Morphology（Kernel Size）主流程。

AOI 步驟與函式對照（步驟編號為個人學習筆記的 13 步）：
  ① 取像        現成資料集 dataset/              （本檔 all_paths）
  步驟 1、2     影像輸入、轉灰階                  aoi/loader.py     load_gray_images
  （筆記沒有）  合成參考圖（52 張中位數）          aoi/reference.py  build_reference
  步驟 3、4、9  去雜訊、照明校正、與參考圖相減後二值化   aoi/detect.py  make_defect_mask
  步驟 10       二值化後形態學清理（實驗參數 ksize）     aoi/detect.py  clean_mask
  步驟 11、12   連通元件分析、面積量測              aoi/detect.py     count_components
  步驟 13       缺陷判定與輸出（OK/NG）            尚未實作
"""
from pathlib import Path

import cv2
import numpy as np

import aoi.loader as loader
import aoi.reference as reference
import aoi.detect as detect

# 專案根目錄（main.py 就在根目錄，所以只需要一個 .parent）
BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "results" / "main_test"          # results/ 已被 .gitignore 排除

# ---- 資料夾（只用非旋轉的 PCB 01 圖，尺寸皆為 1586x3034，不需要對位） ----
open_circuit_path = BASE_DIR / "dataset" / "Open_circuit" / "Open_circuit"
short_path = BASE_DIR / "dataset" / "Short"
spurious_copper_path = BASE_DIR / "dataset" / "Spurious_copper"
dir_list = [open_circuit_path, short_path, spurious_copper_path]

all_paths = []   # 所有圖像檔的路徑
for d in dir_list:
    if not d.is_dir():
        raise FileNotFoundError(f"找不到資料夾：{d}")
    all_paths += sorted(d.glob("*.jpg"))              # sorted：順序固定，實驗才能重現
if not all_paths:
    raise FileNotFoundError("沒有找到任何 .jpg 圖片")


def save_image(name, img):
    """用 imencode + tofile 存圖，避免 cv2.imwrite 在中文路徑失敗。"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise OSError(f"無法編碼圖片：{name}")
    buf.tofile(str(OUT_DIR / name))


def main():
    # AOI 步驟 1、2：影像輸入與轉灰階
    images = loader.load_gray_images(all_paths)
    print("載入張數:", len(images), "形狀:", images[0].shape, images[0].dtype)

    # 參考圖：52 張逐像素中位數
    ref = reference.build_reference(images)
    save_image("reference.png", ref)

    # 取一張測試圖，用原始設定（ksize=5）跑完整流程
    test = images[0]
    diff, mask = detect.make_defect_mask(ref, test)            # 【步驟 3、4、9】照明校正、相減、去雜訊、二值化
    cleaned = detect.clean_mask(mask, ksize=5)                 # 【步驟 10】形態學清理（ksize=5 為原始設定）
    n_before, _, _ = detect.count_components(mask)             # 【步驟 11、12】清理前的元件數（觀察用）
    n_after, areas, boxes = detect.count_components(cleaned)   # 【步驟 11、12】清理後的元件數與面積
    print(f"{all_paths[0].name}: 清理前元件 {n_before}，清理後元件 {n_after}")
    print("清理後各元件面積:", sorted(areas, reverse=True)[:10])

    save_image("test.png", test)
    save_image("diff.png", diff)
    save_image("mask_raw.png", mask)
    save_image("mask_cleaned.png", cleaned)


if __name__ == "__main__":
    main()
