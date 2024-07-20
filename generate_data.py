import pandas as pd
import numpy as np

def generate_dummy_data(num_samples):
    np.random.seed(42)  # for reproducibility

    # Generate IDs
    ids = [f"{i:03d}" for i in range(1, num_samples + 1)]
    
    # Generate Names (dummy names Balita 1, Balita 2, ...)
    names = [f"Balita {i}" for i in range(1, num_samples + 1)]
    
    # Generate Usia (months) with normal distribution
    usia = np.random.normal(loc=24, scale=10, size=num_samples).astype(int)
    usia = np.clip(usia, 6, 60)  # Clamp values to be within realistic range
    
    # Generate Tinggi Badan (cm) with normal distribution
    tinggi_badan = np.random.normal(loc=80, scale=10, size=num_samples)
    tinggi_badan = np.clip(tinggi_badan, 50, 110)  # Clamp values to be within realistic range
    
    # Generate Berat Badan (kg) with normal distribution
    berat_badan = np.random.normal(loc=10, scale=2, size=num_samples)
    berat_badan = np.clip(berat_badan, 4, 20)  # Clamp values to be within realistic range
    
    # Generate Status Gizi Ibu (0: Kurang, 1: Baik)
    status_gizi_ibu = np.random.choice(['Kurang', 'Baik'], size=num_samples)
    
    # Generate Riwayat Penyakit (0: Ada, 1: Tidak Ada)
    riwayat_penyakit = np.random.choice(['Ada', 'Tidak Ada'], size=num_samples)
    
    # Generate Kondisi Lingkungan (0: Kurang, 1: Sedang, 2: Baik)
    kondisi_lingkungan = np.random.choice(['Kurang', 'Sedang', 'Baik'], size=num_samples)
    
    # Generate Akses Layanan Kesehatan (0: Kurang, 1: Sedang, 2: Baik)
    akses_layanan_kesehatan = np.random.choice(['Kurang', 'Sedang', 'Baik'], size=num_samples)
    
    # Create DataFrame
    data = {
        'ID Balita': ids,
        'Nama': names,
        'Usia (bulan)': usia,
        'Tinggi Badan (cm)': tinggi_badan,
        'Berat Badan (kg)': berat_badan,
        'Status Gizi Ibu': status_gizi_ibu,
        'Riwayat Penyakit': riwayat_penyakit,
        'Kondisi Lingkungan': kondisi_lingkungan,
        'Akses Layanan Kesehatan': akses_layanan_kesehatan
    }
    
    df = pd.DataFrame(data)
    
    return df

# Generate 100 samples of dummy data
num_samples = 10
df_dummy = generate_dummy_data(num_samples)

# Save the DataFrame to an Excel file
file_path = 'data_1.xlsx'
df_dummy.to_excel(file_path, index=False)