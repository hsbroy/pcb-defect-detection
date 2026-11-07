"""AOI 步驟 3、4、9、10、11、12：差異偵測、形態學清理、連通元件計數。

各函式與作者程式的關係（X-2 程式修改用）：
  normalize_illumination  [步驟 4 ] 照抄作者 src/illumination_utils.py
  make_defect_mask        [步驟 3、4、9] 改寫自作者 src/05_difference.py 的 compute_difference()
                          （作者輸入的是「對位後的圖」，這裡直接用已對齊的圖，不做 04 對位）
  clean_mask              [步驟 10] 改寫自作者 src/06_morphology.py 的 clean_mask()，
                          修改處：kernel 大小 (5,5) 改成參數 ksize，並省略作者的
                          「前景超過 50% 就反相」判斷（本資料的遮罩前景極少，用不到）
  count_components        [步驟 11、12] 自行新增，用法參考作者 src/07_components.py
                          （作者濾掉面積 < 20 的元件；這裡預設不濾，才看得出殘留雜點）
主題 1 的實驗參數是 clean_mask() 的 ksize（kernel 大小）。
"""
import cv2
import numpy as np

DIFF_THRESHOLD = 30      # 與作者 05_difference.py 相同的固定門檻（作者註解寫 adaptive，實際是固定值）


def normalize_illumination(gray):
    """AOI 步驟 4（對比／照明校正）：除以大範圍模糊後的背景，消除照明不均。
    與作者 illumination_utils.normalize_illumination 相同。"""
    background = cv2.GaussianBlur(gray, (51, 51), 0)
    corrected = cv2.divide(gray, background, scale=128)
    return cv2.normalize(corrected, None, 0, 255, cv2.NORM_MINMAX)


def make_defect_mask(reference, test):
    """AOI 步驟 3、4、9：與參考圖比對，得到二值缺陷遮罩（白=差異）。"""
    ref_n = normalize_illumination(reference)                          # 【步驟 4】參考圖照明校正
    test_n = normalize_illumination(test)                              # 【步驟 4】測試圖照明校正
    diff = cv2.absdiff(ref_n, test_n)                                  # （筆記沒有）與參考圖相減，取絕對值
    diff = cv2.GaussianBlur(diff, (3, 3), 0)                           # 【步驟 3】去雜訊（高斯 3×3）
    mask = cv2.threshold(diff, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)[1]   # 【步驟 9】二值化（固定門檻 30）
    return diff, mask


def clean_mask(mask, ksize=5):
    """AOI 步驟 10：二值化後的形態學清理（本題要改的 Kernel Size）。

    先 opening（侵蝕再膨脹）去掉小雜點，再 closing（膨脹再侵蝕）補小洞。
    ksize=5 且橢圓 kernel = 作者 06_morphology.py clean_mask() 的原始設定。
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))        # 【步驟 10】建立橢圓 kernel（大小 = 實驗參數）
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)         # 【步驟 10】opening：去雜點
    return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)        # 【步驟 10】closing：補洞接線


def count_components(mask, min_area=1):
    """AOI 步驟 11、12：連通元件分析，回傳 (元件數, 各元件面積清單, 元件框清單)。

    8 鄰接；面積小於 min_area 的忽略（預設 1 = 全部都算，才能看出「殘留雜點」）。
    """
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)   # 【步驟 11】把相連白點標成一個物件（8 鄰接）
    areas, boxes = [], []
    for i in range(1, count):                                          # 0 是背景，從 1 開始
        area = int(stats[i, cv2.CC_STAT_AREA])                         # 【步驟 12】量測面積與外框
        if area < min_area:
            continue
        areas.append(area)
        boxes.append((int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]),
                      int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])))
    return len(areas), areas, boxes
