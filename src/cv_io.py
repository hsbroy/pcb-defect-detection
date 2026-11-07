"""讀寫圖片的小工具（本專案新增，不是作者原本的檔案）。

原因：專案路徑含中文（例如「專案開發區」）時，cv2.imread 讀不到圖（回傳 None），
cv2.imwrite 也會寫入失敗。這裡改成由 numpy / Python 開檔，OpenCV 只負責編碼與解碼。
介面與 cv2.imread / cv2.imwrite 相同，所以各腳本只要把 cv2.imread 換成 cv_io.imread 即可。
"""
import cv2
import numpy as np


def imread(path, flags=cv2.IMREAD_COLOR):
    """等同 cv2.imread，但支援含中文的路徑；讀不到時回傳 None。"""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def imwrite(path, image):
    """等同 cv2.imwrite，但支援含中文的路徑；成功回傳 True。"""
    suffix = str(path).rsplit(".", 1)[-1] if "." in str(path) else "png"
    ok, buffer = cv2.imencode("." + suffix, image)
    if not ok:
        return False
    buffer.tofile(str(path))
    return True
