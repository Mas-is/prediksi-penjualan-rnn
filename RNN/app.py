import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model
import io
from fpdf import FPDF

# =========================
# KONFIGURASI HALAMAN
# =========================
st.set_page_config(
    page_title="Prediksi Penjualan RNN",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# FUNGSI BANTUAN
# =========================
def format_angka(x):
    try:
        return f"{float(x):,.0f}".replace(",", ".")
    except:
        return "0"

def format_rupiah(x):
    return f"Rp {format_angka(x)}"

@st.cache_resource
def load_saved_model():
    model = load_model("model_rnn_sales.keras")
    scaler = joblib.load("scaler.pkl")
    with open("metadata.json", "r") as f:
        metadata = json.load(f)
    try:
        with open("metrics.json","r") as f:
            metrics = json.load(f)
    except:
        metrics = {"mae":0,"rmse":0}
    return model, scaler, metadata, metrics

def detect_column(df, candidates):
    lower_cols = {col.lower().strip(): col for col in df.columns}
    for c in candidates:
        key = c.lower().strip()
        if key in lower_cols:
            return lower_cols[key]
    return None

def prepare_daily_data(df):
    date_col = detect_column(df, ["date","tanggal","order date","transaction date","invoice date"])
    sales_col = detect_column(df, ["sales","penjualan","total amount","total_amount","revenue","amount","price","quantity"])
    if date_col is None or sales_col is None:
        st.error("Kolom tanggal atau penjualan tidak ditemukan.")
        st.stop()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[sales_col] = pd.to_numeric(df[sales_col], errors="coerce")
    df = df.dropna(subset=[date_col,sales_col])
    daily = df.groupby(date_col)[sales_col].sum().reset_index().sort_values(date_col)
    daily = daily.set_index(date_col).asfreq("D")
    daily[sales_col] = daily[sales_col].fillna(0)
    daily = daily.reset_index()
    daily.columns=["tanggal","penjualan"]
    return daily

# =========================
# GENERATE PDF
# =========================
def generate_pdf(df, metrics, description=""):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial",'B',16)
    pdf.cell(0,10,txt="Prediksi Penjualan 10 Hari",ln=True,align='C')
    pdf.ln(5)
    
    pdf.set_font("Arial", '', 12)
    if description:
        pdf.multi_cell(0,8,description)
        pdf.ln(5)
    
    pdf.set_font("Arial",'B',12)
    pdf.cell(50,10,"Tanggal",border=1)
    pdf.cell(50,10,"Prediksi",border=1)
    pdf.ln()
    
    pdf.set_font("Arial", '', 12)
    for i,row in df.iterrows():
        pdf.cell(50,10,str(row["tanggal"]),border=1)
        pdf.cell(50,10,format_rupiah(row["prediksi_penjualan"]),border=1)
        pdf.ln()
        
    pdf.ln(5)
    pdf.set_font("Arial",'B',12)
    pdf.cell(0,10,"Evaluasi Model",ln=True)
    pdf.set_font("Arial", '', 12)
    pdf.cell(0,8,f"MAE: {format_rupiah(metrics['mae'])}")
    pdf.ln()
    pdf.cell(0,8,f"RMSE: {format_rupiah(metrics['rmse'])}")
    pdf_bytes = pdf.output(dest='S').encode('latin1')
    pdf_buffer = io.BytesIO(pdf_bytes)
    pdf_buffer.seek(0)
    return pdf_buffer

# =========================
# HEADER
# =========================
st.markdown("""
<div style="background:linear-gradient(135deg,#3f7df6,#73a4ff);padding:1.5rem;border-radius:15px;color:white;">
<h1 style="margin:0;">📈 Prediksi Penjualan 10 Hari</h1>
<p style="margin:0.5rem 0 0 0;">Menggunakan 30 hari terakhir untuk memprediksi 10 hari ke depan dengan Stacked RNN.</p>
</div>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR
# =========================
st.sidebar.title("Pengaturan Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV",type=["csv"])
st.sidebar.markdown("---")
st.sidebar.info("File harus ada kolom tanggal dan penjualan.")

# =========================
# LOAD MODEL & DATA
# =========================
model, scaler, metadata, metrics = load_saved_model()

if uploaded_file:
    raw_df = pd.read_csv(uploaded_file)
else:
    raw_df = pd.read_csv("retail_sales.csv")
daily = prepare_daily_data(raw_df)

if len(daily) < 30:
    st.error("Data kurang dari 30 hari.")
    st.stop()

# =========================
# RINGKASAN DATA
# =========================
total_hari = len(daily)
total_penjualan = daily["penjualan"].sum()
rata_rata = daily["penjualan"].mean()
penjualan_terakhir = daily["penjualan"].iloc[-1]
tanggal_awal = daily["tanggal"].min().strftime("%Y-%m-%d")
tanggal_akhir = daily["tanggal"].max().strftime("%Y-%m-%d")

col1,col2,col3,col4 = st.columns(4)
col1.metric("Total Hari",total_hari)
col2.metric("Total Penjualan",format_rupiah(total_penjualan))
col3.metric("Rata-rata Harian",format_rupiah(rata_rata))
col4.metric("Penjualan Terakhir",format_rupiah(penjualan_terakhir))

st.markdown(f"""
<div style='background:#f0f8ff;padding:0.5rem;border-radius:8px;color:#1f3b64;'>
<b>Rentang data:</b> {tanggal_awal} s.d {tanggal_akhir}<br>
<b>Input model:</b> 30 hari terakhir<br>
<b>Output model:</b> 10 hari prediksi
</div>
""", unsafe_allow_html=True)

# =========================
# PREDIKSI
# =========================
last_30 = daily[["penjualan"]].tail(30)
scaled_last_30 = scaler.transform(last_30)
X_input = scaled_last_30.reshape(1,30,1)
prediction_scaled = model.predict(X_input,verbose=0)
prediction = scaler.inverse_transform(prediction_scaled.reshape(-1,1)).flatten()
prediction = np.maximum(prediction,0)
last_date = daily["tanggal"].max()
future_dates = pd.date_range(start=last_date+pd.Timedelta(days=1),periods=10,freq="D")
forecast_df = pd.DataFrame({"tanggal":future_dates,"prediksi_penjualan":prediction})

# =========================
# TABS
# =========================
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard","📄 Data Harian","🔮 Hasil Prediksi","ℹ️ Info Project"])

with tab1:
    combined_chart = pd.concat([daily.tail(90).set_index("tanggal"), forecast_df.set_index("tanggal")], axis=1)
    st.line_chart(combined_chart)

with tab2:
    daily_rp = daily.copy()
    daily_rp["penjualan"] = daily_rp["penjualan"].apply(lambda x: format_rupiah(x))
    st.dataframe(daily_rp)

with tab3:
    forecast_rp = forecast_df.copy()
    forecast_rp["prediksi_penjualan"] = forecast_rp["prediksi_penjualan"].apply(lambda x: format_rupiah(x))
    st.dataframe(forecast_rp)
    
    csv_buffer = forecast_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv_buffer, "prediksi_penjualan.csv", "text/csv")
    
    pdf_buffer = generate_pdf(forecast_df, metrics, description="Prediksi penjualan menggunakan Stacked RNN berdasarkan 30 hari terakhir.")
    st.download_button("Download PDF", pdf_buffer, "prediksi_penjualan.pdf", "application/pdf")

with tab4:
    st.markdown(f"""
    <div style="padding:1rem;background:#f0f8ff;border-radius:10px;color:#1f3b64;">
    <h3>Info Project</h3>
    <p>Project ini menggunakan data penjualan harian untuk memprediksi 10 hari ke depan menggunakan <b>Stacked RNN</b>. Model menganalisis pola temporal dari 30 hari terakhir.</p>
    <ul>
        <li><b>Total Hari Data:</b> {total_hari}</li>
        <li><b>Total Penjualan:</b> {format_rupiah(total_penjualan)}</li>
        <li><b>Rata-rata Harian:</b> {format_rupiah(rata_rata)}</li>
        <li><b>Penjualan Terakhir:</b> {format_rupiah(penjualan_terakhir)}</li>
    </ul>
    <p><b>MAE:</b> {format_rupiah(metrics['mae'])} &nbsp;&nbsp; <b>RMSE:</b> {format_rupiah(metrics['rmse'])}</p>
    <p>Interpretasi: Nilai MAE dan RMSE menunjukkan seberapa jauh prediksi menyimpang dari data aktual. Semakin kecil nilainya, semakin akurat model.</p>
    </div>
    """, unsafe_allow_html=True)