"""AOI 步驟 1、2：影像輸入與轉灰階。

【來源：自行新增】作者的 src/02、03、04 用 cv2.imread 讀「Final Dataset」資料夾（實際不存在），
這裡改成讀 dataset/ 並用 np.fromfile + imdecode，以支援含中文的路徑。
"""
import cv2
import numpy as np


def read_gray(path):
    """讀一張圖並直接轉成灰階（回傳 uint8 的 2 維陣列）。

    為什麼不用 cv2.imread：專案路徑含中文時 imread 會回傳 None（實測）。
    做法：由 numpy 開檔（Python 能處理中文路徑），OpenCV 只在記憶體內解碼。
    這個寫法在純英文路徑同樣可用。
    """
    data = np.fromfile(path, dtype=np.uint8)           # 先用 numpy 把檔案讀成位元組（此時還不是圖）
    return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)    # 再由 OpenCV 解碼成灰階圖；失敗時回傳 None


def load_gray_images(paths):
    """AOI 步驟 1、2：把路徑清單讀成灰階圖清單。

    任何一張讀不到就直接報錯，而不是跳過，
    因為三組實驗必須使用完全相同的測試圖。
    """
    images = []
    for p in paths:
        img = read_gray(p)
        if img is None:
            raise FileNotFoundError(f"讀不到圖片：{p}")
        images.append(img)
    return images
