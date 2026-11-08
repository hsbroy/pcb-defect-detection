# 作業二 主題 A：Mathematical Morphology（Kernel Size）

> 機器視覺與 AI 演算法應用｜作業二。原作者的說明保留在 [README_原作者.md](README_原作者.md)。

## 1. 這份作業做了什麼

這是一套「與無缺陷參考圖相減 → 形態學清理 → 連通元件分析 → SVM 分類」的 PCB 缺陷檢測流程。
作業要求修改 **形態學（opening／closing）的 Kernel Size**，做三組實驗並記錄「殘留雜點數」與「指定缺陷是否保留」。

| 設定 | Kernel Size | 說明 |
|---|---|---|
| 1 | **5×5** | 原作者設定（原始） |
| 2 | 3×3 | 實驗 |
| 3 | 7×7 | 實驗 |

只改 `src/06_morphology.py` 的 kernel 大小，其餘條件固定；`src/07_components.py` 以命令列參數選擇要分析哪一組。

> 資料集沒有缺陷位置標註，所以所有數字都是**觀察指標**，不是檢出率或準確率。`09_train_svm.py` 的標籤是由「資料夾名稱」推出來的，其 accuracy 不能當成缺陷檢出率。

## 2. 環境需求

- Python 3.12（作者測試版本：3.12.6，Windows 11；程式用 `pathlib`，macOS／Linux 預期可執行，但未實測）
- 套件見 `requirements.txt`：`opencv-python`、`numpy`、`pandas`、`scikit-learn`、`matplotlib`、`joblib`、`scipy`、`openpyxl`
  （測試版本：opencv-python 5.0.0、numpy 2.5.3、pandas 3.1.1、scikit-learn 1.8.0、matplotlib 3.11.2、joblib 1.5.3、scipy 1.17.1；`openpyxl` 程式沒用到，測試環境未安裝，也能正常執行）

```bash
pip install -r requirements.txt
```

## 3. 資料夾內容

```text
pcb-defect-detection/
├── dataset/               資料集共 112 張（PCB 01）
│   ├── Open_circuit/Open_circuit/        20 張（未旋轉）
│   ├── Short/                            20 張（未旋轉）
│   ├── Spurious_copper/                  12 張（未旋轉）
│   ├── open_circuit_Rotation/ Short_Rotation/ Spurious_copper_rotation/   各 20 張，共 60 張（有旋轉，需對位）
├── reference.jpg          參考圖（作者未附；以 52 張未旋轉圖的逐像素中位數合成，已附在專案內）
├── src/                   作者的流程 01～10（已修正為可執行，並新增 Kernel Size 實驗）
│   ├── cv_io.py           本作業新增：讀寫圖片（專案路徑含中文時 cv2.imread／imwrite 會失敗）
│   └── ...
├── models/                09 訓練出的 SVM 模型
├── results/               執行後自動產生（已被 .gitignore 排除）
├── requirements.txt
├── README.md              本檔
└── README_原作者.md        原作者的 README
```

## 4. 如何重現結果

**請先把終端機移到本資料夾。** 所有路徑都由程式檔位置推算，不依賴你放在哪裡，也不受中文路徑影響。

```bash
cd pcb-defect-detection
```

依序執行（後面的步驟會讀前一步的輸出，**不能跳著跑**）：

| 步驟 | 指令 | 在做什麼 | 輸出（`results/` 下） | 耗時* |
|---|---|---|---|---|
| 04 | `python src/04_alignment.py` | 用 ORB＋RANSAC 把 60 張旋轉圖轉正，對位到 `reference.jpg` | `04_alignment/` | 約 16 秒 |
| 05 | `python src/05_difference.py` | 與參考圖相減、高斯模糊、門檻 30 二值化，得到缺陷遮罩 | `05_difference/` | 約 12 秒 |
| 06 | `python src/06_morphology.py` | **本作業實驗**：kernel 3、5、7 各做 opening＋closing | `06_morphology/kernel_size{3,5,7}/` | 約 17 秒 |
| 07 | `python src/07_components.py 3` <br> `python src/07_components.py 5` <br> `python src/07_components.py 7` | 連通元件分析，**參數選要讀哪一組 kernel**（不給參數＝5） | `07_components/kernel_size{N}/` | 各約 10～15 秒 |
| 08 | `python src/08_features.py` | 量測每個元件的特徵（**只讀 kernel 5 的結果**） | `08_features/` | 約 5 秒 |
| 09 | `python src/09_train_svm.py` | 訓練 SVM（48 張訓練、12 張驗證），並覆寫 `models/` 裡的模型檔 | `09_train_svm/` | 約 5 秒 |
| 10 | `python src/10_visualize.py` | 把預測畫回對位後的圖 | `10_visualize/` | 約 13 秒 |

\*作者電腦上的量測值，全部跑完約 2 分鐘。

選用：

- `python src/02_preprocess.py`、`python src/03_illumination.py`：作者的前處理「支線」（灰階／濾波、照明校正＋CLAHE），輸出沒有被 04～10 使用。
- **不要跑 `src/01_view_image.py`**：這是作者的範例檔，讀的 `data/sample.jpg` 不存在，會直接報錯，與作業結果無關。

> `04_alignment.py` 最後會呼叫 `plt.show()`，在有螢幕的環境會跳出預覽視窗，**關掉視窗才會結束**。
> 想完全不跳視窗：Linux／macOS 在指令前加 `MPLBACKEND=Agg`；Windows PowerShell 先執行 `$env:MPLBACKEND="Agg"`。

## 5. 預期結果（用來核對你重現得對不對）

### 5-1 07 的連通元件總數（60 張遮罩，每張圖都有統計）

打開 `results/07_components/kernel_size{N}/component_details.csv`，資料列數應為：

| Kernel | 元件總數 | open_circuit | short | spurious_copper |
|---|---|---|---|---|
| 3×3 | **334** | 113 | 98 | 123 |
| 5×5（原始） | **235** | 60 | 75 | 100 |
| 7×7 | **226** | 60 | 78 | 88 |

作業的兩個指標：

- **指定缺陷**＝`open_circuit` 每張 3 個實心白塊，共 60 個；**殘留雜點數**＝open_circuit 的元件數扣掉這 60 個：3×3＝**53**、5×5＝**0**、7×7＝**0**。
- 60 個指定缺陷在三種 kernel 下都保留。

### 5-2 其他可核對的終端機輸出

- 04：`Aligned 60 of 60 image(s).`，第一張 `open_circuit_Rotation\01_open_circuit_01.jpg (2732 matches, 2598 inliers)`
- 06、07：`Cleaned 60 of 60 mask(s).`／`Analyzed 60 of 60 mask(s).`
- 09：`Trained on 48 image(s); evaluated on 12 held-out image(s).`（驗證集共 46 個元件，accuracy 約 0.826，**僅供核對，不是檢出率**）

### 5-3 重現驗證

在一份全新複本（只複製程式與模型，資料集以資料夾連結共用）依上表順序、使用 `MPLBACKEND=Agg` 全部重跑，結果與作者原本的結果比對如下，皆**逐位元組相同**：

- `04_alignment/aligned_images`：60／60 張
- `05_difference/defect_mask`：60／60 張
- `06_morphology/kernel_size5/cleaned_masks`：60／60 張
- `07_components/kernel_size{3,5,7}` 的 `component_details.csv`、`component_summary.csv`
- `08_features/`、`09_train_svm/` 的所有 CSV

## 6. 程式修改摘要（修改前 → 修改後）

| 項目 | 修改前（作者） | 修改後 |
|---|---|---|
| `06_morphology.py` | kernel 寫死 `(5, 5)`，只跑一組 | `clean_mask(raw_mask, ksize=5)`，`KERNEL_SIZES = (3, 5, 7)` 逐組執行 |
| `07_components.py` | 固定讀一組清理後遮罩 | 命令列參數選 kernel，預設 5 |
| 讀寫圖片 | `cv2.imread`／`cv2.imwrite` | 改用 `cv_io.imread`／`cv_io.imwrite`（中文路徑也能用） |
| 資料夾與參考圖 | `Final Dataset`（不存在）、沒有 `reference.jpg` | `dataset/`；以 52 張未旋轉圖的中位數合成 `reference.jpg` |
| 輸出位置 | `results/` 下平面存放 | `results/NN_腳本名/`，06、07 再分 `kernel_size3／5／7` |

## 7. 常見問題

- **`FileNotFoundError: ... folder not found`**（例如 06 的 `Raw defect mask folder not found`、07 的 `Cleaned masks folder not found`）：前一步沒先跑。依 §4 順序執行。
- **`07` 找不到 `kernel_size3`**：要先執行 06（它會一次產生 3、5、7 三組）。
- **04 跑完卡住不結束**：關掉跳出的預覽視窗，或使用 `MPLBACKEND=Agg`。
- **想只看 kernel 3 或 7 的結果**：`python src/07_components.py 3`（或 7）即可；08～10 只讀 5×5，不隨 kernel 改變。
