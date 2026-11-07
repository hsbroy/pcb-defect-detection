"""參考圖（golden image）合成。

【來源：自行新增，作者程式沒有這個步驟】（筆記步驟沒有，屬「與參考圖比對」的前置準備）
作者的 src/04、05 都讀 reference.jpg，但 repo 沒有附這個檔案，所以這裡自己合成（X-2 程式修改）：
資料集同屬 PCB 01，缺陷位置每張都不同，對多張圖「逐像素取中位數」，
多數張數在該像素是正常的，缺陷會被多數蓋掉，得到近似無缺陷的板子。
"""
import numpy as np


def build_reference(images):
    """逐像素中位數合成參考圖。

    images: 灰階圖清單，每張尺寸必須相同。
    回傳: uint8 的 2 維陣列。
    """
    stack = np.stack(images, axis=0)               # (N, H, W)：把 N 張疊成一個三維陣列
    median = np.median(stack, axis=0)              # axis=0：沿「第幾張」這個方向，對每個像素位置取中位數
    return median.astype(np.uint8)                 # 中位數是浮點數，轉回 uint8 才是正常的圖
