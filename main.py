import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="Analisis Mall Customers", page_icon="🛍️", layout="wide")

INCOME = "Annual Income (k$)"
SPENDING = "Spending Score (1-100)"
FITUR_SEMUA = ["Age", INCOME, SPENDING]
FITUR_MODEL = [INCOME, SPENDING]

# ============================================================
# TAMPILAN (CSS): judul kelompok menu dan kartu ringkasan
# ============================================================
st.markdown("""
<style>
[data-testid="stNavSectionHeader"], [data-testid="stNavSectionHeader"] * {
    font-size: 1.05rem !important; font-weight: 700 !important;
}
[data-testid="stSidebarNavLink"] span { font-size: 0.98rem; }
[data-testid="stMetric"] {
    border: 1px solid rgba(128,128,128,.25); border-left: 5px solid #7aa7e8;
    border-radius: 10px; padding: 14px 18px;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# DATA DAN MODEL
# ============================================================
@st.cache_data
def load_data():
    return pd.read_csv("Mall_Customers.csv")


def latih(data, fitur, k):
    scaler = StandardScaler().fit(data[fitur])
    X = scaler.transform(data[fitur])
    model = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    return scaler, model, X


@st.cache_resource
def model_utama():
    """Model untuk prediksi: Income + Spending Score, K = 5."""
    return latih(load_data(), FITUR_MODEL, 5)


@st.cache_data
def evaluasi_k(fitur):
    rows = []
    for k in range(2, 11):
        _, model, X = latih(load_data(), list(fitur), k)
        rows.append({"K": k, "Inertia": model.inertia_,
                     "Silhouette Score": silhouette_score(X, model.labels_)})
    return pd.DataFrame(rows)


def interpretasi(income, spending):
    if 40 <= income <= 70 and 40 <= spending <= 60:
        return "Pelanggan menengah (sedang-sedang)"
    a = "tinggi" if income > 60 else "rendah"
    b = "tinggi" if spending > 50 else "rendah"
    return f"Income {a}, belanja {b}"


def profil_cluster():
    _, model, _ = model_utama()
    hasil = df.assign(Cluster=model.labels_)
    tabel = hasil.groupby("Cluster")[FITUR_SEMUA].mean()
    tabel.insert(0, "Jumlah", hasil["Cluster"].value_counts())
    tabel["Interpretasi"] = [interpretasi(r[INCOME], r[SPENDING]) for _, r in tabel.iterrows()]
    return tabel.round(1)


def prediksi(data):
    scaler, model, _ = model_utama()
    return model.predict(scaler.transform(data[FITUR_MODEL]))


df = load_data()


# ============================================================
# PENGANTAR
# ============================================================
def beranda():
    st.title("🛍️ Aplikasi Analisis Mall Customers")
    st.write("Selamat datang di aplikasi analisis data Mall Customers.")
    st.write("Aplikasi ini mengelompokkan pelanggan mall menggunakan algoritma **K-Means**.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Jumlah Pelanggan", len(df))
    c2.metric("Jumlah Segmen", 5)
    c3.metric("Silhouette Score (K=5)", f"{evaluasi_k(tuple(FITUR_MODEL)).set_index('K').loc[5, 'Silhouette Score']:.3f}")

    st.subheader("Tujuan")
    st.markdown(
        "- Mempelajari framework **Streamlit** untuk membuat aplikasi web dengan Python.\n"
        "- Menampilkan data dan membuat grafik.\n"
        "- Membangun segmentasi pelanggan dengan **K-Means**."
    )

    st.subheader("Alur Aplikasi")
    st.markdown("Tentang Dataset → EDA → PCA → K-Means & Evaluasi → Prediksi → Kesimpulan")

    with st.expander("Demo Elemen Teks"):
        st.header("Header: Tentang Dataset")
        st.subheader("Subheader: Informasi Singkat")
        st.markdown(f"Dataset ini berisi **{len(df)} pelanggan** mall dengan kolom *Gender*, *Age*, *Annual Income*, dan *Spending Score*.")
        st.caption("Caption: data digunakan untuk keperluan praktikum Machine Learning.")
        st.text("Teks biasa: aplikasi dibuat menggunakan Streamlit.")
        st.code("df = pd.read_csv('Mall_Customers.csv')", language="python")
        st.latex(r"d(x,c)=\sqrt{\sum_{i=1}^{n}(x_i-c_i)^2}")


def tentang_dataset():
    st.title("📄 Tentang Dataset")
    st.caption("Mengenali struktur, kualitas data, dan statistik deskriptif.")
    tab1, tab2, tab3 = st.tabs(["📋 Ringkasan", "✅ Kualitas Data", "🔢 Statistik"])

    with tab1:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Jumlah Pelanggan", len(df))
        c2.metric("Rata-rata Usia", f"{df['Age'].mean():.1f}")
        c3.metric("Rata-rata Income (k$)", f"{df[INCOME].mean():.1f}")
        c4.metric("Rata-rata Spending Score", f"{df[SPENDING].mean():.1f}")
        st.subheader("Tabel Data")
        st.dataframe(df, width="stretch", height=300, hide_index=True)
        st.subheader("Arti Kolom")
        st.table(pd.DataFrame({
            "Kolom": df.columns,
            "Artinya": ["Nomor identitas pelanggan (tidak dipakai sebagai fitur)",
                        "Jenis kelamin", "Usia pelanggan",
                        "Pendapatan tahunan dalam ribuan dolar", "Skor belanja (1-100)"],
        }).set_index("Kolom"))

    with tab2:
        c1, c2 = st.columns(2)
        c1.metric("Data Kosong", int(df.isna().sum().sum()))
        c2.metric("Baris Duplikat", int(df.duplicated().sum()))
        st.dataframe(pd.DataFrame({"Tipe Data": df.dtypes.astype(str),
                                   "Data Kosong": df.isna().sum()}), width="stretch")

    with tab3:
        st.subheader("Statistik Deskriptif")
        st.table(df[FITUR_SEMUA].describe().round(2))


def eda():
    st.title("📊 Eksplorasi Data (EDA)")
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Garis & Area", "Batang", "Histogram", "Altair", "Plotly"])

    with tab1:
        st.subheader("Grafik Garis (st.line_chart)")
        st.line_chart(df.set_index("CustomerID")[FITUR_MODEL])
        st.subheader("Grafik Area (st.area_chart)")
        st.area_chart(df.set_index("CustomerID")[["Age"]])
    with tab2:
        st.subheader("Jumlah Pelanggan Berdasarkan Gender (st.bar_chart)")
        st.bar_chart(df["Gender"].value_counts())
        st.subheader("Rata-rata Spending Score per Gender")
        st.bar_chart(df.groupby("Gender")[SPENDING].mean())
    with tab3:
        st.subheader("Histogram Usia (Matplotlib)")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(df["Age"], bins=15, color="#4C78A8", edgecolor="white")
        ax.set_xlabel("Usia")
        ax.set_ylabel("Jumlah Pelanggan")
        st.pyplot(fig)
        plt.close(fig)
    with tab4:
        st.subheader("Scatter Plot Altair: Income vs Spending Score")
        chart = (alt.Chart(df).mark_circle(size=70)
                 .encode(x=INCOME, y=SPENDING, color="Gender",
                         tooltip=["CustomerID", "Gender", "Age", INCOME, SPENDING])
                 .interactive())
        st.altair_chart(chart, width="stretch")
    with tab5:
        st.subheader("Scatter Plot Plotly: Usia vs Spending Score")
        fig = px.scatter(df, x="Age", y=SPENDING, color="Gender", size=INCOME, hover_data=["CustomerID"])
        st.plotly_chart(fig, width="stretch")


# ============================================================
# PENGOLAHAN & MODEL
# ============================================================
def halaman_pca():
    st.title("📉 Reduksi Dimensi (PCA)")
    st.write("PCA meringkas 3 fitur (usia, income, spending score) menjadi 2 komponen agar mudah divisualisasikan.")
    st.info("PCA di halaman ini hanya untuk visualisasi. Model K-Means tidak dilatih dari hasil PCA.")

    X = StandardScaler().fit_transform(df[FITUR_SEMUA])
    pca = PCA(n_components=2)
    Z = pca.fit_transform(X)
    var = pca.explained_variance_ratio_

    c1, c2, c3 = st.columns(3)
    c1.metric("Variansi PC1", f"{var[0]:.1%}")
    c2.metric("Variansi PC2", f"{var[1]:.1%}")
    c3.metric("Total Variansi", f"{var.sum():.1%}")

    hasil = df.assign(PC1=Z[:, 0], PC2=Z[:, 1], Cluster=model_utama()[1].labels_.astype(str))
    st.subheader("Proyeksi 2 Dimensi")
    st.caption("Warna menunjukkan cluster dari model Income + Spending Score (K = 5).")
    st.plotly_chart(px.scatter(hasil, x="PC1", y="PC2", color="Cluster", hover_data=["CustomerID"],
                               category_orders={"Cluster": list("01234")}), width="stretch")
    st.subheader("Bobot Fitur (Loading)")
    st.dataframe(pd.DataFrame(pca.components_.T, index=FITUR_SEMUA, columns=["PC1", "PC2"]).round(3), width="stretch")


def pemodelan():
    st.title("⚙️ K-Means & Evaluasi")
    st.write("Pelanggan dikelompokkan berdasarkan fitur yang dipilih. Atur parameter di bawah.")
    col_a, col_b = st.columns(2)
    fitur = col_a.multiselect("Pilih fitur untuk clustering", FITUR_SEMUA, default=FITUR_MODEL)
    k = col_b.slider("Jumlah cluster (K)", 2, 10, 5)

    if len(fitur) < 2:
        st.warning("Pilih minimal 2 fitur.")
        st.stop()

    tab_model, tab_eval = st.tabs(["🎯 Hasil Model", "📈 Evaluasi K"])

    with tab_eval:
        ev = evaluasi_k(tuple(fitur))
        c1, c2 = st.columns(2)
        c1.plotly_chart(px.line(ev, x="K", y="Inertia", markers=True, title="Metode Elbow"), width="stretch")
        c2.plotly_chart(px.line(ev, x="K", y="Silhouette Score", markers=True, title="Silhouette Score"), width="stretch")
        st.dataframe(ev.round(3), width="stretch", hide_index=True)
        st.success(f"Silhouette tertinggi untuk fitur ini: K = {int(ev.loc[ev['Silhouette Score'].idxmax(), 'K'])}")

    with tab_model:
        if st.button("Jalankan Model", type="primary"):
            _, model, X = latih(df, fitur, k)
            hasil = df.assign(Cluster=model.labels_.astype(str))
            st.success(f"Model selesai. Silhouette Score: {silhouette_score(X, model.labels_):.3f}")

            st.subheader("Visualisasi Cluster")
            st.plotly_chart(px.scatter(hasil, x=fitur[0], y=fitur[1], color="Cluster",
                                       hover_data=["CustomerID", "Gender", "Age"],
                                       category_orders={"Cluster": [str(i) for i in range(k)]}), width="stretch")
            st.subheader("Rata-rata Tiap Cluster")
            st.dataframe(hasil.groupby("Cluster")[FITUR_SEMUA].mean().round(1).join(
                hasil["Cluster"].value_counts().rename("Jumlah")), width="stretch")
            st.subheader("Data Hasil Clustering")
            st.dataframe(hasil, height=250, width="stretch", hide_index=True)
            st.download_button("Unduh hasil (CSV)", hasil.to_csv(index=False), "hasil_cluster.csv", "text/csv")
        else:
            st.info("Klik tombol 'Jalankan Model' untuk memulai.")


# ============================================================
# PENERAPAN
# ============================================================
def prediksi_satu():
    st.title("👤 Prediksi Satu Pelanggan")
    st.write("Menentukan segmen pelanggan baru dengan model K-Means (Income + Spending Score, K = 5).")
    with st.form("form_prediksi"):
        c1, c2 = st.columns(2)
        income = c1.number_input("Annual Income (k$)", 0, 500, 60)
        spending = c2.number_input("Spending Score (1-100)", 1, 100, 50)
        kirim = st.form_submit_button("Prediksi Segmen", type="primary")

    if kirim:
        baru = pd.DataFrame({INCOME: [income], SPENDING: [spending]})
        cluster = int(prediksi(baru)[0])
        st.success(f"Pelanggan masuk **Cluster {cluster}**: {profil_cluster().loc[cluster, 'Interpretasi']}")
        if not df[INCOME].min() <= income <= df[INCOME].max():
            st.warning(f"Income di luar rentang data latih ({df[INCOME].min()}-{df[INCOME].max()} k$), hasil kurang dapat diandalkan.")
        gambar = df.assign(Kelompok=[f"Cluster {c}" for c in model_utama()[1].labels_])
        fig = px.scatter(gambar, x=INCOME, y=SPENDING, color="Kelompok",
                         category_orders={"Kelompok": [f"Cluster {i}" for i in range(5)]})
        fig.add_scatter(x=[income], y=[spending], mode="markers", name="Pelanggan baru",
                        marker=dict(size=18, symbol="star", color="gold", line=dict(width=1, color="black")))
        st.plotly_chart(fig, width="stretch")


def prediksi_massal():
    st.title("📁 Prediksi Massal (CSV)")
    st.write(f"Upload file CSV yang memiliki kolom **{INCOME}** dan **{SPENDING}**.")
    st.download_button("Unduh contoh format CSV", df[FITUR_MODEL].head(10).to_csv(index=False),
                       "contoh_input.csv", "text/csv")
    file = st.file_uploader("Pilih file CSV", type="csv")
    if file is None:
        return
    try:
        data = pd.read_csv(file)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError):
        st.error("File tidak bisa dibaca. Pastikan formatnya CSV dengan pemisah koma.")
        return
    kurang = [c for c in FITUR_MODEL if c not in data.columns]
    if kurang:
        st.error("Kolom belum ada: " + ", ".join(kurang))
        return
    data[FITUR_MODEL] = data[FITUR_MODEL].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    bersih = data.dropna(subset=FITUR_MODEL).copy()
    if len(bersih) < len(data):
        st.warning(f"{len(data) - len(bersih)} baris dilewati karena nilainya kosong atau bukan angka.")
    if bersih.empty:
        st.error("Tidak ada baris yang bisa diprediksi.")
        return
    bersih["Cluster"] = prediksi(bersih)
    bersih["Interpretasi"] = bersih["Cluster"].map(profil_cluster()["Interpretasi"])
    st.success(f"{len(bersih)} baris berhasil diprediksi.")
    st.dataframe(bersih, width="stretch", hide_index=True)
    st.download_button("Unduh hasil prediksi", bersih.to_csv(index=False), "hasil_prediksi.csv", "text/csv")


def kesimpulan():
    st.title("✅ Kesimpulan & Saran")
    skor = evaluasi_k(tuple(FITUR_MODEL)).set_index("K").loc[5, "Silhouette Score"]
    st.subheader("Karakteristik Tiap Cluster (K = 5)")
    st.dataframe(profil_cluster(), width="stretch")

    st.subheader("Kesimpulan")
    st.markdown(
        "Streamlit mempermudah pembuatan aplikasi web dengan Python: menampilkan data, membuat visualisasi, "
        "dan menjalankan model machine learning secara interaktif tanpa perlu menulis HTML atau JavaScript. "
        f"K-Means pada dataset Mall Customers menghasilkan **lima segmen pelanggan** dengan Silhouette Score **{skor:.3f}**."
    )
    st.markdown(
        "- Segmen **income tinggi, belanja tinggi** paling potensial bagi pihak mall.\n"
        "- Segmen **income tinggi, belanja rendah** berpeluang ditingkatkan lewat promosi yang tepat sasaran."
    )

    st.subheader("Keterbatasan")
    st.markdown(
        "1. File Mall_Customers.csv harus satu folder dengan main.py.\n"
        "2. Data hanya 200 pelanggan dengan sedikit fitur numerik, sehingga belum tentu mewakili kondisi sebenarnya.\n"
        "3. Hanya memakai K-Means, belum dibandingkan dengan algoritma lain.\n"
        "4. Belum ada penyimpanan permanen; hasil diunduh manual dalam bentuk CSV."
    )
    st.subheader("Saran")
    st.markdown(
        "- Bandingkan K-Means dengan DBSCAN atau Hierarchical Clustering.\n"
        "- Tambah jumlah data dan fitur pelanggan.\n"
        "- Tambahkan database untuk menyimpan hasil."
    )


# ============================================================
# NAVIGASI SIDEBAR
# ============================================================
halaman = st.navigation({
    "Pengantar": [
        st.Page(beranda, title="Beranda", icon="🏠", default=True),
        st.Page(tentang_dataset, title="Tentang Dataset", icon="📄"),
        st.Page(eda, title="Eksplorasi Data (EDA)", icon="📊"),
    ],
    "Pengolahan & Model": [
        st.Page(halaman_pca, title="Reduksi Dimensi (PCA)", icon="📉"),
        st.Page(pemodelan, title="K-Means & Evaluasi", icon="⚙️"),
    ],
    "Penerapan": [
        st.Page(prediksi_satu, title="Prediksi Satu Pelanggan", icon="👤"),
        st.Page(prediksi_massal, title="Prediksi Massal (CSV)", icon="📁"),
        st.Page(kesimpulan, title="Kesimpulan & Saran", icon="✅"),
    ],
}, expanded=True)

with st.sidebar:
    st.markdown(
        "**Tugas Machine Learning**  \n"
        "Streamlit Mall Customers  \n"
        "Kelas 05TPLP013 · Universitas Pamulang"
    )
    st.markdown(
        "**Disusun oleh Kelompok 8:**\n"
        "- Dika Rahmat Fadillah (241011400525)\n"
        "- Kurniawan Bagus Satria (241011400529)\n"
        "- Muhammad Arwani Mahfud (241011402040)"
    )
    st.caption("Dosen: Farizi Ilham, S.Kom., M.Kom.")

halaman.run()
