# =========================================================================
# 09_train_svm.py 概要
#
# 【在做什麼】拿 08 的特徵表（features.csv）訓練一個 SVM 分類器，判斷每個缺陷元件屬於哪一類缺陷，
#            並輸出評估報告、混淆矩陣、模型檔，以及「每個元件的預測類型」給 10 使用。
#            (筆記外)：這是機器學習分類，不在你的 AOI 13 步內；它分的是「缺陷類型」，不是 OK/NG。
#
# 【流程】
#   讀 results/08_features/features.csv，檢查欄位、缺值、標籤是否衝突
#     └→ 以「整張圖」為單位切分：80% 訓練、20% 驗證（stratify 讓每類比例相近，random_state=42 固定可重現）
#     └→ 用訓練集訓練 → 在驗證集預測 → 輸出分類報告與混淆矩陣     （這是評估用）
#     └→ 再用「全部資料」重新訓練最終模型，存 models/svm_defect_classifier.joblib
#     └→ 用最終模型對全部元件預測 predicted_defect_type，存 features_with_severity.csv（給 10）
#
# 【模型】Pipeline：StandardScaler（特徵標準化）→ SVC（RBF 核，C=1.0）
#        使用 13 個特徵：area、width、height、aspect_ratio、perimeter、circularity、hu1～hu7
#
# 【注意（紅線）】
#   1. 標籤（label）來自「資料夾名稱」，不是人工標註的缺陷位置，所以這裡的準確率是「缺陷類型分類準確率」，
#      不能說成檢出率；沒有缺陷標註也不能算 Precision/Recall/F1 來當檢測成效。
#   2. predicted_defect_type 是用「全部資料」訓練的最終模型去預測同一批資料，
#      等於在訓練資料上預測，不能拿來當準確率（讀碼判斷，沒有另外驗證）。
#   3. 驗證集只有約 20% 的圖（你之前的實測：48 張訓練、12 張驗證），樣本很少，結果不穩定。
# 【輸入／輸出】輸入 08 的 features.csv；輸出 results/09_train_svm/（報告、混淆矩陣、含預測的特徵表）與 models/（模型檔）。
# =========================================================================
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

BASE_DIR = Path(__file__).resolve().parent.parent                           # 宣告BASE_DIR紀錄路徑pcb-defect-detection
FEATURES_CSV_PATH = BASE_DIR / "results" / "08_features" / "features.csv"   # 宣告FEATURES_CSV_PATH紀錄特徵量測結果的輸出路徑pcb-defect-detection/results/08_features/features.csv
RESULTS_DIR = BASE_DIR / "results" / "09_train_svm"                         # 宣告RESULTS_DIR紀錄處理後圖片路徑pcb-defect-detection/results/09_train_svm
MODEL_DIR = BASE_DIR / "models"                                             # 宣告MODEL_DIR紀錄模型輸出路徑pcb-defect-detection/models
MODEL_PATH = MODEL_DIR / "svm_defect_classifier.joblib"                     # 宣告MODEL_PATH紀錄SVM缺陷分類器模型的輸出路徑pcb-defect-detection/models/svm_defect_classifier.joblib
PREDICTIONS_CSV_PATH = RESULTS_DIR / "features_with_severity.csv"           # 宣告PREDICTIONS_CSV_PATH紀錄特徵量測結果與預測缺陷類型的輸出路徑pcb-defect-detection/results/09_train_svm/features_with_severity.csv
CLASSIFICATION_REPORT_PATH = RESULTS_DIR / "svm_classification_report.csv"  # 宣告CLASSIFICATION_REPORT_PATH紀錄SVM缺陷分類器的分類報告輸出路徑pcb-defect-detection/results/09_train_svm/svm_classification_report.csv
CONFUSION_MATRIX_PATH = RESULTS_DIR / "svm_confusion_matrix.csv"            # 宣告CONFUSION_MATRIX_PATH紀錄SVM缺陷分類器的混淆矩陣輸出路徑pcb-defect-detection/results/09_train_svm/svm_confusion_matrix.csv
FEATURE_COLUMNS = [     # 餵給 SVM 的 13 個特徵欄位（不含位置、嚴重度、標籤）
    "area",             # 面積
    "width",            # 外框寬
    "height",           # 外框高
    "aspect_ratio",     # 長寬比
    "perimeter",        # 周長
    "circularity",      # 圓度
    "hu1",              # Hu 不變矩 1～7
    "hu2",
    "hu3",
    "hu4",
    "hu5",
    "hu6",
    "hu7",
]


def create_model():
    """Build the scaled RBF support-vector classifier."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),                              # (筆記外)機器學習:特徵標準化（平均 0、標準差 1），避免面積這類大數值蓋過其他特徵
            ("svm", SVC(kernel="rbf", C=1.0, random_state=42)),        # (筆記外)機器學習:SVM 分類「缺陷類型」，不是 OK/NG；rbf 核＝非線性分界，C=1.0 控制容錯程度
        ]
    )


def main():
    # _______________讀特徵表並做檢查_______________
    if not FEATURES_CSV_PATH.is_file():
        raise FileNotFoundError(
            f"Feature dataset not found: {FEATURES_CSV_PATH}. "
            "Run 08_features.py first."
        ) # 08 的特徵表不存在 → 提示先跑 08

    features = pd.read_csv(FEATURES_CSV_PATH)                                      # 讀 08 的 features.csv（每個元件一列）
    required_columns = {"relative_path", "label", *FEATURE_COLUMNS}                # 必要欄位：圖路徑、標籤、13 個特徵
    missing_columns = sorted(required_columns.difference(features.columns))        # 找出缺少的欄位
    if missing_columns:
        raise ValueError(
            f"Feature dataset is missing required columns: {', '.join(missing_columns)}"
        )
    if features.empty:
        raise ValueError(
            "No defect features are available for training. "
            "Check the generated masks and component filtering."
        ) # 一個元件都沒有，無法訓練
    if features[FEATURE_COLUMNS].isna().any().any():
        raise ValueError("Feature dataset contains missing classifier values.")   # 特徵有缺值就停止

    image_labels = features[["relative_path", "label"]].drop_duplicates()          # 每張圖一列（圖路徑＋標籤），切分時以「圖」為單位
    conflicting_labels = image_labels["relative_path"].duplicated(keep=False)      # 同一張圖若出現兩種標籤就是衝突
    if conflicting_labels.any():
        raise ValueError("An image has conflicting defect-class labels.")
    label_counts = image_labels["label"].value_counts()                            # 每個類別有幾張圖
    if len(label_counts) < 2 or label_counts.min() < 2:                            # 至少要有 2 類、每類至少 2 張，才能切訓練/驗證
        raise ValueError(
            "Training requires at least two labeled images in each of two classes."
        )

    # _______________以「整張圖」為單位切分訓練集與驗證集_______________
    # 注意:label 來自「資料夾名稱」，不是人工標註的缺陷位置；這裡的準確率是「缺陷類型分類」，不能說成檢出率
    # 以「整張圖」為單位切分 80% 訓練 / 20% 驗證，避免同一張圖的元件同時出現在兩邊
    train_images, test_images = train_test_split(
        image_labels,
        test_size=0.2,                                  # 20% 的圖當驗證集
        random_state=42,                                # 固定亂數種子，每次切分結果相同
        stratify=image_labels["label"],                 # 依標籤分層，讓訓練/驗證裡各類比例相近
    )
    train_paths = set(train_images["relative_path"])    # 訓練集的圖路徑集合
    test_paths = set(test_images["relative_path"])      # 驗證集的圖路徑集合
    train_rows = features[features["relative_path"].isin(train_paths)]   # 屬於訓練圖的所有元件列
    test_rows = features[features["relative_path"].isin(test_paths)]     # 屬於驗證圖的所有元件列
    if train_rows.empty or test_rows.empty:
        raise ValueError(
            "Image-level train/test split produced an empty feature partition."
        )

    # _______________評估:只用訓練集訓練，在驗證集預測_______________
    evaluation_model = create_model()  # 先只用訓練集訓練、在驗證集評估（產生分類報告與混淆矩陣）
    evaluation_model.fit(train_rows[FEATURE_COLUMNS], train_rows["label"])         # 訓練：特徵 → 標籤
    predictions = evaluation_model.predict(test_rows[FEATURE_COLUMNS])             # 對驗證集預測類別
    classes = sorted(features["label"].unique())                                   # 所有類別名稱（排序後固定欄位順序）

    report = classification_report(                                                # 分類報告：每類的 precision / recall / f1 與整體 accuracy
        test_rows["label"],
        predictions,
        labels=classes,
        output_dict=True,
        zero_division=0,                                                           # 某類沒被預測到時，指標記 0 而不是報警告
    )
    confusion = confusion_matrix(                                                  # 混淆矩陣：列＝實際類別、欄＝預測類別
        test_rows["label"], predictions, labels=classes
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(report).transpose().to_csv(                                       # 報告轉置後存成 CSV
        CLASSIFICATION_REPORT_PATH, encoding="utf-8-sig"
    )
    pd.DataFrame(
        confusion,
        index=[f"actual_{label}" for label in classes],                            # 列名：actual_類別
        columns=[f"predicted_{label}" for label in classes],                       # 欄名：predicted_類別
    ).to_csv(CONFUSION_MATRIX_PATH, encoding="utf-8-sig")

    # _______________最終模型:用全部資料重新訓練並存檔_______________
    final_model = create_model()  # 再用全部資料重新訓練最終模型，存成 joblib，並預測每個元件的類型（供 10 標註）
    final_model.fit(features[FEATURE_COLUMNS], features["label"])
    features["predicted_defect_type"] = final_model.predict(                       # 在「訓練過的同一批資料」上預測，不是獨立驗證，不能當準確率
        features[FEATURE_COLUMNS]
    )
    joblib.dump(final_model, MODEL_PATH)                                           # 把模型存成檔案，之後可以直接載入使用
    features.to_csv(PREDICTIONS_CSV_PATH, index=False, encoding="utf-8-sig")       # 特徵表加上 predicted_defect_type，給 10 標註用

    print(
        f"Trained on {len(train_paths)} image(s); evaluated on {len(test_paths)} "
        f"held-out image(s)."
    )
    print(f"Classifier classes: {', '.join(classes)}")
    print(f"Evaluation report saved to: {CLASSIFICATION_REPORT_PATH}")
    print(f"Confusion matrix saved to: {CONFUSION_MATRIX_PATH}")
    print(f"Final model saved to: {MODEL_PATH}")
    print(f"Features and predictions saved to: {PREDICTIONS_CSV_PATH}")


if __name__ == "__main__":
    main()
