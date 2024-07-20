import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Membaca data dari file Excel
file_path = 'data.xlsx'  # Ganti dengan path file Excel Anda
df = pd.read_excel(file_path)

# Encoding categorical variables
df['Status Gizi Ibu'] = df['Status Gizi Ibu'].map({'Baik': 1, 'Kurang': 0})
df['Riwayat Penyakit'] = df['Riwayat Penyakit'].map({'Tidak Ada': 1, 'Ada': 0})
df['Kondisi Lingkungan'] = df['Kondisi Lingkungan'].map({'Baik': 2, 'Sedang': 1, 'Kurang': 0})
df['Akses Layanan Kesehatan'] = df['Akses Layanan Kesehatan'].map({'Baik': 2, 'Sedang': 1, 'Kurang': 0})

# Features for clustering
features = ['Usia (bulan)', 'Tinggi Badan (cm)', 'Berat Badan (kg)', 'Status Gizi Ibu', 'Riwayat Penyakit', 'Kondisi Lingkungan', 'Akses Layanan Kesehatan']
X = df[features]

# Handling infinite values
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.dropna(inplace=True)

# Standardizing the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine optimal number of clusters using silhouette score
best_n_components = 5
best_silhouette_score = -1

for n_components in range(2, 10):
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(X_scaled)
    cluster_labels = gmm.predict(X_scaled)
    silhouette_avg = silhouette_score(X_scaled, cluster_labels)
    print(f"Number of clusters: {n_components}, Silhouette Score: {silhouette_avg}")
    
    if silhouette_avg > best_silhouette_score:
        best_silhouette_score = silhouette_avg
        best_n_components = n_components

print(f"Optimal number of clusters determined: {best_n_components}")

# Generate dynamic cluster labels
default_cluster_labels = ['Sangat Buruk', 'Buruk', 'Cukup', 'Baik', 'Sangat Baik']
cluster_labels = [f'Cluster {i+1}' for i in range(best_n_components)]

# Loop until all clusters are separated properly
iteration = 0
while True:
    iteration += 1
    # Gaussian Mixture Model
    gmm = GaussianMixture(n_components=best_n_components, random_state=42)
    gmm.fit(X_scaled)
    df['Cluster'] = gmm.predict(X_scaled)

    # Determine cluster labels
    cluster_centers = gmm.means_
    sorted_cluster_indices = np.argsort(np.sum(cluster_centers, axis=1))

    cluster_map = {sorted_cluster_indices[i]: cluster_labels[i] for i in range(best_n_components)}

    df['Cluster Label'] = df['Cluster'].map(cluster_map)

    # Log the iteration
    unique_clusters = df['Cluster Label'].unique()
    print(f"Iteration {iteration}: Found {len(unique_clusters)} unique clusters: {unique_clusters}")

    # Check if clusters are properly separated
    if len(unique_clusters) == best_n_components:
        print("All clusters are properly separated.")
        break

# Menyimpan hasil ke dalam file Excel
output_excel_path = 'clustered_data.xlsx'
df.to_excel(output_excel_path, index=False)

# Plotting the clusters and saving to PNG
plt.figure(figsize=(12, 8))

# Plot scatter points with smaller size for larger data
sns.scatterplot(data=df, x='Tinggi Badan (cm)', y='Berat Badan (kg)', hue='Cluster Label', palette='viridis', s=5, alpha=0.6)

# Plot KDE for each cluster
for label in cluster_labels:
    sns.kdeplot(data=df[df['Cluster Label'] == label], x='Tinggi Badan (cm)', y='Berat Badan (kg)', fill=True, alpha=0.1)

plt.title('Clustering of Balita Data Using GMM with Cluster Areas')
plt.xlabel('Tinggi Badan (cm)')
plt.ylabel('Berat Badan (kg)')
plt.legend(title='Cluster')

# Menyimpan grafik dalam bentuk file PNG
output_png_path = 'clustered_data_plot.png'
plt.savefig(output_png_path)
plt.close()

# Pair plot for additional visualization and saving to PNG
pairplot = sns.pairplot(df, hue='Cluster Label', palette='viridis', diag_kind='kde')
pairplot_path = 'pairplot_clustered_data.png'
pairplot.savefig(pairplot_path)
plt.close()
