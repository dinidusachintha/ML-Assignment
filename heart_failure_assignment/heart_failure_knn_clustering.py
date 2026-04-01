  import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    silhouette_score
)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# =========================
# 1. LOAD DATA
# =========================
file_path = "heart_failure_clinical_records_dataset.csv"
df = pd.read_csv(file_path)

print("First 5 rows:")
print(df.head())
print("\nDataset shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())
print("\nMissing values:")
print(df.isnull().sum())

# =========================
# 2. BASIC EXPLORATION
# =========================
print("\nTarget distribution:")
print(df["DEATH_EVENT"].value_counts())

print("\nSummary statistics:")
print(df.describe())

# =========================
# 3. DEFINE FEATURES/TARGET
# =========================
X = df.drop("DEATH_EVENT", axis=1)
y = df["DEATH_EVENT"]

# =========================
# 4. TRAIN-TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# 5. FEATURE SCALING
# IMPORTANT for KNN and K-Means
# =========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_scaled_all = scaler.fit_transform(X)

# =========================
# 6. KNN CLASSIFICATION
# =========================
print("\n=========================")
print("KNN CLASSIFICATION")
print("=========================")

# Hyperparameter tuning
param_grid = {
    "n_neighbors": list(range(1, 21)),
    "weights": ["uniform", "distance"],
    "metric": ["euclidean", "manhattan"]
}

knn = KNeighborsClassifier()

grid_search = GridSearchCV(
    estimator=knn,
    param_grid=param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

grid_search.fit(X_train_scaled, y_train)

best_knn = grid_search.best_estimator_

print("Best Parameters:", grid_search.best_params_)

# Predictions
y_pred = best_knn.predict(X_test_scaled)
y_prob = best_knn.predict_proba(X_test_scaled)[:, 1]

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print("\nKNN Results:")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Predicted Survived", "Predicted Died"],
            yticklabels=["Actual Survived", "Actual Died"])
plt.title("KNN Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("KNN ROC Curve")
plt.legend()
plt.tight_layout()
plt.show()

# =========================
# 7. K-MEANS CLUSTERING
# =========================
print("\n=========================")
print("K-MEANS CLUSTERING")
print("=========================")

# Find best k using silhouette score
silhouette_scores = []
k_values = range(2, 7)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled_all)
    score = silhouette_score(X_scaled_all, cluster_labels)
    silhouette_scores.append(score)
    print(f"k = {k}, Silhouette Score = {score:.4f}")

best_k = k_values[np.argmax(silhouette_scores)]
print("\nBest number of clusters based on silhouette score:", best_k)

# Train final KMeans
kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
clusters = kmeans_final.fit_predict(X_scaled_all)

df["Cluster"] = clusters

# Cluster summary
cluster_summary = df.groupby("Cluster").mean(numeric_only=True)
cluster_counts = df["Cluster"].value_counts().sort_index()

print("\nCluster counts:")
print(cluster_counts)

print("\nCluster mean summary:")
print(cluster_summary)

# Mortality rate per cluster
cluster_mortality = df.groupby("Cluster")["DEATH_EVENT"].mean() * 100
print("\nMortality rate (%) by cluster:")
print(cluster_mortality)

# =========================
# 8. PCA VISUALIZATION
# =========================
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled_all)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap="viridis")
plt.title("Patient Clusters Visualized with PCA")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar(scatter, label="Cluster")
plt.tight_layout()
plt.show()

# =========================
# 9. ELBOW + SILHOUETTE PLOT
# =========================
inertias = []
for k in range(1, 7):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled_all)
    inertias.append(km.inertia_)

plt.figure(figsize=(6, 5))
plt.plot(range(1, 7), inertias, marker="o")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method for K-Means")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 5))
plt.plot(list(k_values), silhouette_scores, marker="o")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Scores for K-Means")
plt.tight_layout()
plt.show()

# =========================
# 10. SIMPLE INTERPRETATION
# =========================
print("\n=========================")
print("INTERPRETATION")
print("=========================")

highest_risk_cluster = cluster_mortality.idxmax()
lowest_risk_cluster = cluster_mortality.idxmin()

print(f"Highest risk cluster: Cluster {highest_risk_cluster} with mortality rate {cluster_mortality[highest_risk_cluster]:.2f}%")
print(f"Lowest risk cluster : Cluster {lowest_risk_cluster} with mortality rate {cluster_mortality[lowest_risk_cluster]:.2f}%")

print("\nDone successfully.")