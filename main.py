import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="Analisis Mall Customers", layout="wide")

# ============================================================
# MEMBACA DATASET
# ============================================================
@st.cache_data
def load_data():
    return pd.read_csv("Mall_Customers.csv")

df = load_data()

# ============================================================
# SIDEBAR (NAVIGASI)
# ============================================================
st.sidebar.title("Menu")
menu = st.sidebar.radio(
    "Pilih halaman:",
    ["1. Elemen Teks", "2. Menampilkan Data", "3. Menampilkan Grafik", "4. Pemodelan K-Means"],
)

# ============================================================
# 1. ELEMEN TEKS
# ============================================================
if menu == "1. Elemen Teks":
    st.title("Aplikasi Analisis Mall Customers")
    st.write("Selamat datang di aplikasi analisis data Mall Customers.")
    st.header("Header: Tentang Dataset")
    st.subheader("Subheader: Informasi Singkat")
    st.markdown("Dataset ini berisi **200 pelanggan** mall dengan kolom *Gender*, *Age*, *Annual Income*, dan *Spending Score*.")
    st.caption("Caption: data digunakan untuk keperluan praktikum Machine Learning.")
    st.text("Teks biasa: aplikasi dibuat menggunakan Streamlit.")
    st.code("df = pd.read_csv('Mall_Customers.csv')", language="python")
    st.latex(r"d(x,c)=\sqrt{\sum_{i=1}^{n}(x_i-c_i)^2}")

# ============================================================
# 2. MENAMPILKAN DATA
# ============================================================
elif menu == "2. Menampilkan Data":
    st.title("Menampilkan Data")

    st.subheader("Ringkasan")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Jumlah Pelanggan", len(df))
    c2.metric("Rata-rata Usia", f"{df['Age'].mean():.1f}")
    c3.metric("Rata-rata Income (k$)", f"{df['Annual Income (k$)'].mean():.1f}")
    c4.metric("Rata-rata Spending Score", f"{df['Spending Score (1-100)'].mean():.1f}")

    st.subheader("Tabel Data (st.dataframe)")
    st.dataframe(df, width="stretch", height=300)

    st.subheader("Statistik Deskriptif")
    st.table(df.describe().round(2))

# ============================================================
# 3. MENAMPILKAN GRAFIK
# ============================================================
elif menu == "3. Menampilkan Grafik":
    st.title("Menampilkan Grafik")
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Garis & Area", "Batang", "Histogram", "Altair", "Plotly"]
    )

    with tab1:
        st.subheader("Grafik Garis (st.line_chart)")
        st.line_chart(df.set_index("CustomerID")[["Annual Income (k$)", "Spending Score (1-100)"]])
        st.subheader("Grafik Area (st.area_chart)")
        st.area_chart(df.set_index("CustomerID")[["Age"]])

    with tab2:
        st.subheader("Jumlah Pelanggan Berdasarkan Gender (st.bar_chart)")
        st.bar_chart(df["Gender"].value_counts())
        st.subheader("Rata-rata Spending Score per Gender")
        st.bar_chart(df.groupby("Gender")["Spending Score (1-100)"].mean())

    with tab3:
        st.subheader("Histogram Usia (Matplotlib)")
        fig, ax = plt.subplots()
        ax.hist(df["Age"], bins=15, color="#4C78A8", edgecolor="white")
        ax.set_xlabel("Usia")
        ax.set_ylabel("Jumlah Pelanggan")
        st.pyplot(fig)

    with tab4:
        st.subheader("Scatter Plot Altair: Income vs Spending Score")
        chart = (
            alt.Chart(df)
            .mark_circle(size=70)
            .encode(
                x="Annual Income (k$)",
                y="Spending Score (1-100)",
                color="Gender",
                tooltip=["CustomerID", "Gender", "Age", "Annual Income (k$)", "Spending Score (1-100)"],
            )
            .interactive()
        )
        st.altair_chart(chart, width="stretch")

    with tab5:
        st.subheader("Scatter Plot Plotly: Usia vs Spending Score")
        fig = px.scatter(
            df, x="Age", y="Spending Score (1-100)", color="Gender",
            size="Annual Income (k$)", hover_data=["CustomerID"],
        )
        st.plotly_chart(fig, width="stretch")

# ============================================================
# 4. PEMODELAN K-MEANS
# ============================================================
elif menu == "4. Pemodelan K-Means":
    st.title("Pemodelan Segmentasi Pelanggan (K-Means)")
    st.write("Pelanggan dikelompokkan berdasarkan fitur yang dipilih. Atur parameter pada panel di bawah.")

    fitur_semua = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
    col_a, col_b = st.columns(2)
    with col_a:
        fitur = st.multiselect("Pilih fitur untuk clustering", fitur_semua,
                               default=["Annual Income (k$)", "Spending Score (1-100)"])
    with col_b:
        k = st.slider("Jumlah cluster (K)", 2, 10, 5)

    if len(fitur) < 2:
        st.warning("Pilih minimal 2 fitur.")
        st.stop()

    if st.button("Jalankan Model"):
        X = StandardScaler().fit_transform(df[fitur])
        model = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
        hasil = df.copy()
        hasil["Cluster"] = model.labels_.astype(str)

        st.success(f"Model selesai. Silhouette Score: {silhouette_score(X, model.labels_):.3f}")

        st.subheader("Visualisasi Cluster")
        fig = px.scatter(hasil, x=fitur[0], y=fitur[1], color="Cluster",
                         hover_data=["CustomerID", "Gender", "Age"])
        st.plotly_chart(fig, width="stretch")

        st.subheader("Rata-rata Tiap Cluster")
        st.dataframe(hasil.groupby("Cluster")[fitur_semua].mean().round(1).join(
            hasil["Cluster"].value_counts().rename("Jumlah")))

        st.subheader("Data Hasil Clustering")
        st.dataframe(hasil, height=250)
        st.download_button("Unduh hasil (CSV)", hasil.to_csv(index=False), "hasil_cluster.csv", "text/csv")
    else:
        st.info("Klik tombol 'Jalankan Model' untuk memulai.")
