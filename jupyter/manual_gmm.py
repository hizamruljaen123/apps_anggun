from flask import Flask, render_template, send_file
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)

# Fungsi untuk menghitung PDF Gaussian
def gaussian_pdf(X, mean, cov):
    n = X.shape[1]
    diff = X - mean
    exp_term = np.exp(-0.5 * np.sum(np.dot(diff, np.linalg.inv(cov)) * diff, axis=1))
    return exp_term / np.sqrt((2 * np.pi) ** n * np.linalg.det(cov))

# Fungsi untuk inisialisasi parameter GMM
def initialize_gmm(X, k):
    n, d = X.shape
    weights = np.ones(k) / k
    means = X[np.random.choice(n, k, replace=False)]
    covariances = np.array([np.cov(X, rowvar=False)] * k)
    return weights, means, covariances

# Fungsi E-step
def e_step(X, weights, means, covariances):
    k = len(weights)
    n = X.shape[0]
    resp = np.zeros((n, k))

    for i in range(k):
        resp[:, i] = weights[i] * gaussian_pdf(X, means[i], covariances[i])
    
    resp /= resp.sum(axis=1, keepdims=True)
    return resp

# Fungsi M-step
def m_step(X, resp):
    n, d = X.shape
    k = resp.shape[1]
    
    weights = resp.sum(axis=0) / n
    means = np.dot(resp.T, X) / resp.sum(axis=0)[:, np.newaxis]
    covariances = np.zeros((k, d, d))
    
    for i in range(k):
        diff = X - means[i]
        covariances[i] = np.dot(resp[:, i] * diff.T, diff) / resp[:, i].sum()
    
    return weights, means, covariances

# Fungsi log-likelihood
def log_likelihood(X, weights, means, covariances):
    k = len(weights)
    n = X.shape[0]
    log_likelihood = 0
    
    for i in range(n):
        tmp = 0
        for j in range(k):
            tmp += weights[j] * gaussian_pdf(X[i].reshape(1, -1),  means[j], covariances[j])
        log_likelihood += np.log(tmp)
    
    return log_likelihood

# Fungsi untuk GMM
def gmm(X, k, max_iters=100, tol=1e-6):
    weights, means, covariances = initialize_gmm(X, k)
    log_likelihoods = []
    
    for i in range(max_iters):
        print(f"Iteration {i+1}")
        resp = e_step(X, weights, means, covariances)
        weights, means, covariances = m_step(X, resp)
        current_log_likelihood = log_likelihood(X, weights, means, covariances)
        log_likelihoods.append(current_log_likelihood)
        print(f"Log Likelihood: {current_log_likelihood}")
        
        if i > 0 and abs(log_likelihoods[-1] - log_likelihoods[-2]) < tol:
            print("Converged")
            break
    
    return weights, means, covariances, resp, log_likelihoods

@app.route('/')
def index():
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
    X = df[features].values

    # Handling infinite values
    X = np.nan_to_num(X)

    # Standarisasi data
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X = (X - X_mean) / X_std

    # Menentukan jumlah klaster
    k = 5

    # Melakukan GMM
    weights, means, covariances, resp, log_likelihoods = gmm(X, k)

    # Menentukan klaster untuk setiap data
    clusters = resp.argmax(axis=1)

    # Menambahkan hasil klaster ke DataFrame
    df['Cluster'] = clusters

    # Menyimpan hasil ke dalam file Excel
    output_excel_path = 'clustered_data_manual.xlsx'
    df.to_excel(output_excel_path, index=False)

    # Plotting the clusters and saving to PNG
    plt.figure(figsize=(12, ))

    # Plot scatter points with smaller size for larger data
    sns.scatterplot(x=X[:, 1], y=X[:, 2], hue=clusters, palette='viridis', s=5, alpha=0.6)

    plt.title('Clustering of Balita Data Using Manual GMM')
    plt.xlabel('Tinggi Badan (normalized)')
    plt.ylabel('Berat Badan (normalized)')
    plt.legend(title='Cluster')

    # Menyimpan grafik dalam bentuk file PNG
    output_png_path = 'static/clustered_data_plot_manual.png'
    plt.savefig(output_png_path)
    plt.close()

    # Render index.html with the path to the image
    return render_template('index.html', image_path='clustered_data_plot_manual.png')

if __name__ == '__main__':
    app.run(debug=True)
