from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.manifold import TSNE
import plotly.graph_objs as go

app = Flask(__name__)

# Fungsi untuk menghitung PDF Gaussian
def gaussian_pdf(X, mean, cov, epsilon=1e-6):
    diff = X - mean
    cov += np.eye(cov.shape[0]) * epsilon
    exp_term = np.exp(-0.5 * np.einsum('ij,jk,ik->i', diff, np.linalg.inv(cov), diff))
    return exp_term / np.sqrt((2 * np.pi) ** X.shape[1] * np.linalg.det(cov))

# Inisialisasi parameter GMM
def initialize_gmm(X, k):
    n, d = X.shape
    weights = np.ones(k) / k
    means = X[np.random.choice(n, k, replace=False)]
    covariances = np.array([np.cov(X, rowvar=False) for _ in range(k)])
    return weights, means, covariances

# Langkah E dalam GMM
def e_step(X, weights, means, covariances):
    k = len(weights)
    resp = np.array([weights[i] * gaussian_pdf(X, means[i], covariances[i]) for i in range(k)]).T
    resp /= resp.sum(axis=1, keepdims=True)
    return resp

# Langkah M dalam GMM
def m_step(X, resp):
    n, d = X.shape
    k = resp.shape[1]

    weights = resp.sum(axis=0) / n
    means = np.dot(resp.T, X) / resp.sum(axis=0)[:, np.newaxis]
    covariances = []
    for i in range(k):
        diff = X - means[i]
        cov = np.dot(resp[:, i] * diff.T, diff) / resp[:, i].sum()
        covariances.append(cov)
    return weights, means, np.array(covariances)

# Fungsi utama GMM
def gmm(X, k, max_iters=100, tol=1e-6):
    weights, means, covariances = initialize_gmm(X, k)
    prev_log_likelihood = None
    for i in range(max_iters):
        resp = e_step(X, weights, means, covariances)
        weights, means, covariances = m_step(X, resp)
        current_log_likelihood = log_likelihood(X, weights, means, covariances)
        if prev_log_likelihood is not None and abs(current_log_likelihood - prev_log_likelihood) < tol:
            break
        prev_log_likelihood = current_log_likelihood
    return weights, means, covariances, resp

# Log-likelihood GMM
def log_likelihood(X, weights, means, covariances):
    likelihood = np.array([weights[k] * gaussian_pdf(X, means[k], covariances[k]) for k in range(len(weights))]).sum(axis=0)
    return np.sum(np.log(likelihood))

@app.route('/')
def index():
    output_excel_path = 'static/clustered_data_manual.xlsx'

    if os.path.exists(output_excel_path):
        df = pd.read_excel(output_excel_path)

        # Validasi apakah kolom 'Cluster' ada
        if 'Cluster' not in df.columns:
            return "Error: Kolom 'Cluster' tidak ditemukan dalam data clustering.", 500
    else:
        return "Error: File hasil clustering tidak ditemukan. Jalankan proses clustering terlebih dahulu.", 404

    # Tambahkan label stunting dalam bentuk string berdasarkan cluster
    cluster_mapping = {
        0: 'Tidak Stunting',
        1: 'Potensi Stunting'
    }
    df['Stunting Label'] = df['Cluster'].map(cluster_mapping).fillna('Cluster Tidak Dikenal')

    # Visualisasi clustering menggunakan Plotly berdasarkan nilai normalisasi hasil GMM
    fig_gmm = go.Figure()
    clusters = df['Cluster'].values
    for cluster in np.unique(clusters):
        cluster_label = cluster_mapping.get(cluster, f'Cluster {cluster}')
        cluster_data = df[df['Cluster'] == cluster]
        fig_gmm.add_trace(go.Scatter(x=cluster_data['Tinggi Badan (cm)'], 
                                      y=cluster_data['Berat Badan (kg)'], 
                                      mode='markers', name=cluster_label))

    fig_gmm.update_layout(
        title="Clustering of Balita Data Using Manual GMM",
        xaxis_title="Tinggi Badan (cm) (normalized)",
        yaxis_title="Berat Badan (kg) (normalized)",
        showlegend=True,
        height=850
    )

    # Visualisasi t-SNE
    features = ['Tinggi Badan (cm)', 'Berat Badan (kg)']
    X = df[features].values
    X_normalized = (X - X.mean(axis=0)) / X.std(axis=0)
    tsne_results = TSNE(n_components=2, random_state=42).fit_transform(X_normalized)

    fig_tsne = go.Figure()
    for cluster in np.unique(clusters):
        cluster_label = cluster_mapping.get(cluster, f'Cluster {cluster}')
        cluster_data_tsne = tsne_results[df['Cluster'] == cluster]
        fig_tsne.add_trace(go.Scatter(x=cluster_data_tsne[:, 0], y=cluster_data_tsne[:, 1], 
                                       mode='markers', name=cluster_label))

    fig_tsne.update_layout(
        title="t-SNE Visualization of Clustering",
        xaxis_title="t-SNE Dimension 1",
        yaxis_title="t-SNE Dimension 2",
        showlegend=True,
        height=850
    )

    # Tampilkan tabel data
    table_content = df[['ID Balita', 'Nama', 'Usia (bulan)', 'Tinggi Badan (cm)', 'Berat Badan (kg)', 'Stunting Label']].head(10).to_html(classes='table table-bordered', index=False)

    return render_template('index.html', 
                           graph_html_gmm=fig_gmm.to_html(full_html=False),
                           graph_html_tsne=fig_tsne.to_html(full_html=False),
                           table_content=table_content)

@app.route('/data', methods=['GET'])
def data_json():
    file_path = '../data.xlsx'
    if os.path.exists(file_path):
        return pd.read_excel(file_path).to_json(orient='records'), 200, {'Content-Type': 'application/json'}
    return {"error": "File not found"}, 404

@app.route('/data_result', methods=['GET'])
def data_result():
    output_excel_path = 'static/clustered_data_manual.xlsx'
    if os.path.exists(output_excel_path):
        df = pd.read_excel(output_excel_path)
        return jsonify(df.to_dict(orient='records'))
    return {"error": "No results available."}, 404

@app.route('/training_data', methods=['GET'])
def training_data():
    file_path = '../data.xlsx'
    model_path = 'gmm_model.pkl'

    if not os.path.exists(file_path):
        return jsonify({"error": "Data file not found."}), 404

    df = pd.read_excel(file_path)

    # Mapping untuk kategori stunting (label)
    stunting_mapping = {'Baik': 0, 'Kurang': 1}
    df['Label Stunting'] = df['Status Gizi Ibu'].map(stunting_mapping)

    # Fitur untuk pelatihan
    features = ['Usia (bulan)', 'Tinggi Badan (cm)', 'Berat Badan (kg)', 
                'Kondisi Lingkungan', 'Akses Layanan Kesehatan']

    mappings = {
        'Kondisi Lingkungan': {'Baik': 2, 'Sedang': 1, 'Kurang': 0},
        'Akses Layanan Kesehatan': {'Baik': 2, 'Sedang': 1, 'Kurang': 0}
    }
    for col, mapping in mappings.items():
        df[col] = df[col].map(mapping)

    X = df[features].values
    X = (X - X.mean(axis=0)) / X.std(axis=0)

    k = 2  # Dua kategori: Stunting (1) dan Tidak Stunting (0)
    weights, means, covariances, _ = gmm(X, k)

    model = {
        'weights': weights, 
        'means': means, 
        'covariances': covariances
    }

    with open(model_path, 'wb') as model_file:
        pickle.dump(model, model_file)

    return jsonify({"message": "Model trained and saved successfully."})

@app.route('/predict', methods=['GET'])
def predict():
    model_path = 'gmm_model.pkl'
    test_data_path = '../data.xlsx'

    if not os.path.exists(model_path):
        return jsonify({"error": "Model not found. Please train the model first."}), 404

    if not os.path.exists(test_data_path):
        return jsonify({"error": "Test data file not found."}), 404

    df = pd.read_excel(test_data_path)

    features = ['Usia (bulan)', 'Tinggi Badan (cm)', 'Berat Badan (kg)', 
                'Kondisi Lingkungan', 'Akses Layanan Kesehatan']

    mappings = {
        'Kondisi Lingkungan': {'Baik': 2, 'Sedang': 1, 'Kurang': 0},
        'Akses Layanan Kesehatan': {'Baik': 2, 'Sedang': 1, 'Kurang': 0}
    }
    for col, mapping in mappings.items():
        df[col] = df[col].map(mapping)

    X = df[features].values
    X = (X - X.mean(axis=0)) / X.std(axis=0)

    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)

    weights, means, covariances = model['weights'], model['means'], model['covariances']
    clusters = e_step(X, weights, means, covariances).argmax(axis=1)

    df['Predicted Cluster'] = clusters
    df['Stunting Label'] = df['Predicted Cluster'].apply(lambda x: 'Potensi Stunting' if x == 1 else 'Tidak Stunting')
    output_path = 'static/predicted_data.xlsx'
    df.to_excel(output_path, index=False)

    return jsonify({"message": "Prediction completed and results saved.", "output_file": output_path})


if __name__ == '__main__':
    app.run(debug=True)
