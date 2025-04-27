import csv
import pandas as pd

def analyze_stunting(csv_file):
    """
    Analyzes stunting risk based on multiple anthropometric factors from a CSV file,
    incorporating medical knowledge.

    Args:
        csv_file (str): Path to the CSV file containing child anthropometric data.

    Returns:
        list: A list of dictionaries, where each dictionary represents a child
              and their stunting classification ("Not Stunted", "Potential Stunting", or "Stunted").
    """

    results = []
    with open(csv_file, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file, delimiter=';')
        for row in reader:
            try:
                zs_tb_u = float(row['ZS TB/U'])
                zs_bb_u = float(row['ZS BB/U'])
                zs_bb_tb = float(row['ZS BB/TB'])
                bb_lahir = float(row['BB Lahir'].replace(',', '.'))  # Handle comma as decimal separator
                tb_lahir = float(row['TB Lahir'].replace(',', '.'))  # Handle comma as decimal separator
                usia_saat_ukur = row['Usia Saat Ukur']
                nama = row['Nama']

                # Extract age in years from "Usia Saat Ukur"
                age_years = int(usia_saat_ukur.split(' ')[0])  # e.g., "2 Tahun" -> 2

                # Define thresholds for low birth weight and height
                low_birth_weight = 2.5
                low_birth_height = 45

                # Convert age to months
                age_months = int(usia_saat_ukur.split(' ')[0]) * 12 if 'Tahun' in usia_saat_ukur else int(usia_saat_ukur.split(' ')[0])
                
                # Evaluate birth parameters using official tables
                birth_status = []
                # Detailed birth parameter evaluation using official tables
                if age_months == 0:
                    # Check against newborn standards (0 months)
                    # Evaluasi berat badan lahir berdasarkan Permenkes
                    if bb_lahir < 2.1:  # -3 SD
                        birth_status.append("BB Lahir Sangat Rendah")
                    elif 2.1 <= bb_lahir < 2.5:  # -3 SD sampai -2 SD
                        birth_status.append("BB Lahir Rendah")
                    elif 2.5 <= bb_lahir < 3.3:  # -2 SD sampai median
                        birth_status.append("BB Lahir Normal Bawah")
                    elif 2.5 <= bb_lahir < 3.3:  # -2 SD to median
                        birth_status.append("BB Lahir Normal Bawah")
                        
                    # Evaluasi panjang badan lahir berdasarkan Permenkes
                    if tb_lahir < 44.2:  # -3 SD
                        birth_status.append("PB Lahir Sangat Pendek")
                    elif 44.2 <= tb_lahir < 46.1:  # -3 SD sampai -2 SD
                        birth_status.append("PB Lahir Pendek")
                    elif 46.1 <= tb_lahir < 49.9:  # -2 SD sampai median
                        birth_status.append("PB Lahir Normal Bawah")
                
                # 1. Evaluasi TB/U sesuai Permenkes No.2/2020
                if zs_tb_u < -2:
                    stunting_status = "Potensi Stunting"
                else:
                    stunting_status = "Tidak Stunting"
                
                # Evaluasi BB/TB (Wasting/Overweight)
                if zs_bb_tb < -3:
                    nutritional_status = "Gizi Buruk"
                elif -3 <= zs_bb_tb < 1:
                    nutritional_status = "Normal"
                else:
                    nutritional_status = "Berlebih"
                
                # Combined risk assessment
                final_status = "Normal"
                if stunting_status == "Potensi Stunting":
                    final_status = stunting_status
                else:
                    final_status = "Tidak Stunting"

                results.append({
                    'Nama': nama,
                    'Stunting Status': final_status,
                    'Nutritional Status': nutritional_status,
                    'Birth Risk Factors': ", ".join(birth_status) if birth_status else "None"
                })
            except ValueError:
                print(f"Skipping row due to invalid data: {row}")
            except KeyError as e:
                print(f"Skipping row due to missing key: {e}")

    return results


if __name__ == "__main__":
    csv_file = 'main_data.csv'
    stunting_analysis = analyze_stunting(csv_file)

    import pandas as pd

    # Create a Pandas DataFrame from the results
    df = pd.DataFrame(stunting_analysis)

    # Define the order of columns
    column_order = ['Nama', 'Stunting Status', 'Nutritional Status', 'Birth Risk Factors']
    df = df[column_order]

    # Save the DataFrame to an Excel file
    output_excel_file = 'stunting_analysis_result.xlsx'
    df.to_excel(output_excel_file, index=False)

    print(f"Analisis stunting telah disimpan ke {output_excel_file}")