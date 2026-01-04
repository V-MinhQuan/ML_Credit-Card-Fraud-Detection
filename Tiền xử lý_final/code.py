import pandas as pd
import numpy as np

# ============================
# 1. Load dataset
# ============================
data = pd.read_csv("data.csv")

# ============================
# 2. Feature engineering
# ============================
# 2.1 Tạo cột Hour từ thời gian
data["Hour"] = (data["Time"] % 86400) // 3600

# 2.2 (Optional) Bỏ cột Time vì ít ý nghĩa
data = data.drop(columns=["Time"])

# ============================
# 3. Xử lý cột Amount
# ============================
from sklearn.preprocessing import RobustScaler

amount_scaler = RobustScaler()

data["Amount"] = amount_scaler.fit_transform(data[["Amount"]])

# ============================
# 4. Chia train/test (KHÔNG scale/oversample trước)
# ============================
from sklearn.model_selection import train_test_split

X = data.drop("Class", axis=1)
y = data["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# ============================
# 5. Scale FEATURES (chỉ fit train → transform test)
# ============================
from sklearn.preprocessing import StandardScaler

# Chỉ scale các cột PCA + Hour (Amount đã scale trước)
features_to_scale = [col for col in X.columns if col not in ["Amount"]]

scaler = StandardScaler()

X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[features_to_scale] = scaler.fit_transform(X_train[features_to_scale])
X_test_scaled[features_to_scale] = scaler.transform(X_test[features_to_scale])

# ============================
# 6. SMOTE – chỉ áp dụng trên TRAIN
# ============================
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

# ============================
# 7. (Optional) Xóa outliers bằng Isolation Forest
# ============================
from sklearn.ensemble import IsolationForest

clf = IsolationForest(contamination=0.001, random_state=42)
isof_labels = clf.fit_predict(X_train_resampled)

mask = isof_labels == 1  # 1 = inlier, -1 = outlier

X_train_final = X_train_resampled[mask]
y_train_final = y_train_resampled[mask]

# ============================
# 8. Dữ liệu sẵn sàng để huấn luyện model
# ============================
print("Train shape:", X_train_final.shape)
print("Test shape:", X_test_scaled.shape)
print("Fraud ratio (train):", y_train_final.mean())
print("Fraud ratio (test):", y_test.mean())


train_df = pd.DataFrame(X_train_final, columns=X.columns)
train_df["Class"] = y_train_final.values

print("\n== 10 dòng đầu của tập train sau xử lý ==")
print(train_df.head(10))

train_df.to_csv("train_processed.csv", index=False)

# ============================
# 9. Xuất tập TEST đã xử lý
# ============================

test_df = pd.DataFrame(X_test_scaled, columns=X.columns)
test_df["Class"] = y_test.values

print("\n== 10 dòng đầu của tập test sau xử lý ==")
print(test_df.head(10))

test_df.to_csv("test_processed.csv", index=False)

print("\nĐã lưu file test_processed.csv")

