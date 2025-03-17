from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import os
import pickle
from datetime import datetime
from sklearn.manifold import TSNE
import plotly.graph_objs as go

app = Flask(__name__)

# Konstanta untuk file data dan sheet yang digunakan
DATA_FILE = "main_data.xlsx"
SHEET_NAME = "Sheet1"

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

# Fungsi untuk menghitung usia (bulan) dari Tgl Lahir
def calculate_age_in_months(birth_date):
    today = datetime.today()
    return (today.year - birth_date.year) * 12 + today.month - birth_date.month

@app.route('/')
def index():
    output_excel_path = 'static/predicted_data.xlsx'
    if not os.path.exists(output_excel_path):
        return "Error: File hasil prediksi tidak ditemukan. Jalankan proses prediksi terlebih dahulu.", 404

    df = pd.read_excel(output_excel_path)
    
    # Validasi kolom yang diperlukan
    required_columns = ['t-SNE 1', 't-SNE 2', 'Predicted Cluster',
                        'Tinggi Badan (cm)', 'Berat Badan (kg)', 
                        'JK', 'Posyandu', 'Desa/Kel']
    if not all(col in df.columns for col in required_columns):
        return "Error: File hasil prediksi tidak valid. Kolom yang diperlukan tidak ditemukan.", 500

    # Mapping label stunting
    cluster_mapping = {
        0: 'Tidak Stunting',
        1: 'Potensi Stunting'
    }
    df['Stunting Label'] = df['Predicted Cluster'].map(cluster_mapping).fillna('Cluster Tidak Dikenal')

    # ----------------
    # GRAFIK GMM
    # ----------------
    fig_gmm = go.Figure()
    clusters = df['Predicted Cluster'].values
    for cluster in np.unique(clusters):
        cluster_label = cluster_mapping.get(cluster, f'Cluster {cluster}')
        cluster_data = df[df['Predicted Cluster'] == cluster]
        fig_gmm.add_trace(go.Scatter(
            x=cluster_data['Tinggi Badan (cm)'],
            y=cluster_data['Berat Badan (kg)'],
            mode='markers',
            name=cluster_label
        ))
    fig_gmm.update_layout(
        title="Clustering of Balita Data Using Manual GMM",
        xaxis_title="Tinggi Badan (cm)",
        yaxis_title="Berat Badan (kg)",
        showlegend=True,
        height=850
    )

    # ----------------
    # GRAFIK t-SNE
    # ----------------
    fig_tsne = go.Figure()
    for cluster in np.unique(clusters):
        cluster_label = cluster_mapping.get(cluster, f'Cluster {cluster}')
        cluster_data_tsne = df[df['Predicted Cluster'] == cluster]
        fig_tsne.add_trace(go.Scatter(
            x=cluster_data_tsne['t-SNE 1'],
            y=cluster_data_tsne['t-SNE 2'],
            mode='markers',
            name=cluster_label
        ))
    fig_tsne.update_layout(
        title="t-SNE Visualization of Clustering",
        xaxis_title="t-SNE Dimension 1",
        yaxis_title="t-SNE Dimension 2",
        showlegend=True,
        height=850
    )

    # ======================================================
    # FUNGSI BANTUAN UNTUK MEMBUAT TABEL FREKUENSI & PERSEN
    # ======================================================
    def create_freq_table(df_grouped, category_name):
        """
        df_grouped: DataFrame dengan kolom [category, 'Stunting Label', 'count']
        category_name: Nama kolom kategori (mis. 'JK', 'Posyandu', dll.)
        """
        # Hitung total keseluruhan
        total = df_grouped['count'].sum()
        # Hitung persentase
        df_grouped['percentage'] = (df_grouped['count'] / total * 100).round(2)
        # Ubah nama kolom agar lebih rapi
        df_grouped.rename(columns={
            category_name: 'Category',
            'Stunting Label': 'Label',
            'count': 'Frequency',
            'percentage': 'Percentage (%)'
        }, inplace=True)
        # Konversi DataFrame ke HTML
        return df_grouped.to_html(classes='table table-bordered', index=False)

    # ----------------
    # GRAFIK GENDER
    # ----------------
    gender_counts = df.groupby(['JK', 'Stunting Label']).size().reset_index(name='count')
    fig_gender = go.Figure()
    for label in gender_counts['Stunting Label'].unique():
        data = gender_counts[gender_counts['Stunting Label'] == label]
        fig_gender.add_trace(go.Bar(
            x=data['JK'],
            y=data['count'],
            name=label
        ))
    fig_gender.update_layout(
        title="Stunting Category by Gender",
        xaxis_title="Jenis Kelamin",
        yaxis_title="Jumlah Balita",
        barmode='group',
        height=850
    )
    # Buat tabel frekuensi & persentase untuk gender
    table_freq_gender_df = gender_counts.copy()
    freq_table_gender = create_freq_table(table_freq_gender_df, 'JK')

    # ----------------
    # GRAFIK POSYANDU (stacked, horizontal)
    # ----------------
    posyandu_counts = df.groupby(['Posyandu', 'Stunting Label']).size().reset_index(name='count')
    fig_posyandu = go.Figure()
    for label in posyandu_counts['Stunting Label'].unique():
        data = posyandu_counts[posyandu_counts['Stunting Label'] == label]
        fig_posyandu.add_trace(go.Bar(
            y=data['Posyandu'],
            x=data['count'],
            name=label,
            orientation='h'
        ))
    fig_posyandu.update_layout(
        title="Stunting Category by Posyandu ",
        xaxis_title="Jumlah Balita",
        yaxis_title="Posyandu",
        barmode='stack',
        height=850
    )
    # Buat tabel frekuensi & persentase untuk posyandu
    table_freq_posyandu_df = posyandu_counts.copy()
    freq_table_posyandu = create_freq_table(table_freq_posyandu_df, 'Posyandu')

    # ----------------
    # GRAFIK DESA (stacked, horizontal)
    # ----------------
    desa_counts = df.groupby(['Desa/Kel', 'Stunting Label']).size().reset_index(name='count')
    fig_desa = go.Figure()
    for label in desa_counts['Stunting Label'].unique():
        data = desa_counts[desa_counts['Stunting Label'] == label]
        fig_desa.add_trace(go.Bar(
            y=data['Desa/Kel'],
            x=data['count'],
            name=label,
            orientation='h'
        ))
    fig_desa.update_layout(
        title="Stunting Category by Desa/Kel ",
        xaxis_title="Jumlah Balita",
        yaxis_title="Desa/Kel",
        barmode='stack',
        height=850
    )
    # Buat tabel frekuensi & persentase untuk desa
    table_freq_desa_df = desa_counts.copy()
    freq_table_desa = create_freq_table(table_freq_desa_df, 'Desa/Kel')

    # ----------------
    # GRAFIK RENTANG USIA (5 KATEGORI)
    # ----------------
    df['Age Range'] = pd.cut(df['Usia (bulan)'], bins=5, include_lowest=True)
    age_counts = df.groupby(['Age Range', 'Stunting Label']).size().reset_index(name='count')
    fig_age = go.Figure()
    for label in age_counts['Stunting Label'].unique():
        data = age_counts[age_counts['Stunting Label'] == label]
        fig_age.add_trace(go.Bar(
            x=data['Age Range'].astype(str),
            y=data['count'],
            name=label
        ))
    fig_age.update_layout(
        title="Stunting Category by Age Range (bulan)",
        xaxis_title="Rentang Usia (bulan)",
        yaxis_title="Jumlah Balita",
        barmode='group',
        height=850
    )
    # Buat tabel frekuensi & persentase untuk rentang usia
    table_freq_age_df = age_counts.copy()
    freq_table_age = create_freq_table(table_freq_age_df, 'Age Range')

    # Tampilkan tabel data detail
    table_columns = [
        'ID Balita', 'Nama', 'Usia (bulan)',
        'Tinggi Badan (cm)', 'Berat Badan (kg)',
        'Desa/Kel', 'Posyandu', 'Stunting Label'
    ]
    table_content = df[table_columns].to_html(classes='table table-bordered', index=False)

    return render_template(
        'index.html',
        # Grafik
        graph_html_gmm=fig_gmm.to_html(full_html=False),
        graph_html_tsne=fig_tsne.to_html(full_html=False),
        graph_html_gender=fig_gender.to_html(full_html=False),
        graph_html_posyandu=fig_posyandu.to_html(full_html=False),
        graph_html_desa=fig_desa.to_html(full_html=False),
        graph_html_age=fig_age.to_html(full_html=False),

        # Tabel utama
        table_content=table_content,

        # Tabel frekuensi & persentase
        freq_table_gender=freq_table_gender,
        freq_table_posyandu=freq_table_posyandu,
        freq_table_desa=freq_table_desa,
        freq_table_age=freq_table_age
    )


@app.route('/data', methods=['GET'])
def data_json():
    # Mengembalikan data asli dari file DATA_FILE (Sheet1)
    if os.path.exists(DATA_FILE):
        df = pd.read_excel(DATA_FILE, sheet_name=SHEET_NAME)
        return df.to_json(orient='records'), 200, {'Content-Type': 'application/json'}
    return {"error": "File not found"}, 404

@app.route('/data_result', methods=['GET'])
def data_result():
    output_excel_path = 'static/predicted_data.xlsx'
    if os.path.exists(output_excel_path):
        df = pd.read_excel(output_excel_path)
        return jsonify(df.to_dict(orient='records'))
    return {"error": "No results available."}, 404

@app.route('/training_data', methods=['GET'])
def training_data():
    model_path = 'gmm_model.pkl'
    if not os.path.exists(DATA_FILE):
        return jsonify({"error": "Data file not found."}), 404

    # Baca data terbaru dari file DATA_FILE, Sheet1
    df = pd.read_excel(DATA_FILE, sheet_name=SHEET_NAME)
    # Hitung usia dalam bulan dari kolom Tgl Lahir
    df['Usia (bulan)'] = pd.to_datetime(df['Tgl Lahir']).apply(calculate_age_in_months)

    # Fitur untuk pelatihan: Gunakan Usia, Tinggi, dan Berat
    features = ['Usia (bulan)', 'Tinggi', 'Berat']
    X = df[features].values.astype(float)
    X_normalized = (X - X.mean(axis=0)) / X.std(axis=0)

    k = 2  # Dua kategori: Tidak Stunting dan Potensi Stunting
    weights, means, covariances, _ = gmm(X_normalized, k)
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
    if not os.path.exists(model_path):
        return jsonify({"error": "Model not found. Please train the model first."}), 404
    if not os.path.exists(DATA_FILE):
        return jsonify({"error": "Data file not found."}), 404

    # Baca data terbaru dari file DATA_FILE, Sheet1
    df = pd.read_excel(DATA_FILE, sheet_name=SHEET_NAME)
    # Hitung usia dalam bulan dari Tgl Lahir
    df['Usia (bulan)'] = pd.to_datetime(df['Tgl Lahir']).apply(calculate_age_in_months)

    # Fitur untuk prediksi: Gunakan Usia, Tinggi, dan Berat
    features = ['Usia (bulan)', 'Tinggi', 'Berat']
    X = df[features].values.astype(float)
    X_normalized = (X - X.mean(axis=0)) / X.std(axis=0)

    # Load model GMM
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
    weights, means, covariances = model['weights'], model['means'], model['covariances']
    clusters = e_step(X_normalized, weights, means, covariances).argmax(axis=1)
    df['Predicted Cluster'] = clusters

    # Mapping label stunting
    df['Stunting Label'] = df['Predicted Cluster'].apply(lambda x: 'Potensi Stunting' if x == 1 else 'Tidak Stunting')

    # Hitung t-SNE untuk visualisasi
    tsne_results = TSNE(n_components=2, random_state=42).fit_transform(X_normalized)
    df['t-SNE 1'] = tsne_results[:, 0]
    df['t-SNE 2'] = tsne_results[:, 1]

    # Ubah nama kolom untuk tampilan konsisten
    df.rename(columns={"Tinggi": "Tinggi Badan (cm)", "Berat": "Berat Badan (kg)"}, inplace=True)
    # Tambahkan ID Balita berdasarkan indeks dan pastikan kolom Nama tetap ada
    df.insert(0, "ID Balita", range(1, len(df) + 1))
    
    # Simpan hasil prediksi ke file Excel
    output_path = 'static/predicted_data.xlsx'
    df.to_excel(output_path, index=False)

    return jsonify({"message": "Prediction completed and results saved.", "output_file": output_path})

if __name__ == '__main__':
    app.run(debug=True)
