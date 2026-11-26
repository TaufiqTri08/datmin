import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

# ==============================================================================
# KONFIGURASI HALAMAN STREAMLIT
# ==============================================================================
st.set_page_config(
    page_title="Dashboard Data Mining — Klasifikasi Tingkat Kelarisan Produk Shopee",
    layout="wide",
)

FEATURE_COLS = [
    "Harga Awal",
    "Harga Setelah Diskon",
    "Persen_Diskon",
    "Selisih_Diskon",
    "Berat_Produk_Gram",
    "Total_Berat_Gram",
    "Punya_Variasi",
    "Judul_Kata_COD",
    "Jumlah",
    "Jumlah Produk di Pesan",
    "Ongkos Kirim Dibayar oleh Pembeli",
    "Estimasi Potongan Biaya Pengiriman",
    "Perkiraan Ongkos Kirim",
    "Is_COD",
    "Is_Hemat_Kargo",
    "Is_Jawa_Timur",
    "Jam_Pesan",
    "Durasi_Penyelesaian_Jam",
]


@st.cache_data
def load_and_prepare_data(csv_path: str = "FP-DM-A-01-DATASET.csv"):
    """Memuat dataset mentah Shopee Seller dan menjalankan Data Preparation."""
    df_raw = pd.read_csv(csv_path)
    df = df_raw.copy()

    # 1. Pembersihan teks satuan berat (' gr' -> float)
    df["Berat_Produk_Gram"] = (
        df["Berat Produk"]
        .astype(str)
        .str.replace(" gr", "", regex=False)
        .str.replace(".", "", regex=False)
        .astype(float)
    )
    df["Total_Berat_Gram"] = (
        df["Total Berat"]
        .astype(str)
        .str.replace(" gr", "", regex=False)
        .str.replace(".", "", regex=False)
        .astype(float)
    )

    # 2. Konversi waktu & ekstraksi fitur temporal
    df["Waktu_Buat"] = pd.to_datetime(df["Waktu Pesanan Dibuat"])
    df["Waktu_Selesai"] = pd.to_datetime(df["Waktu Pesanan Selesai"])
    df["Durasi_Penyelesaian_Jam"] = (
        df["Waktu_Selesai"] - df["Waktu_Buat"]
    ).dt.total_seconds() / 3600.0
    df["Jam_Pesan"] = df["Waktu_Buat"].dt.hour

    # 3. Feature engineering atribut harga, diskon, variasi, pembayaran, logistik
    df["Persen_Diskon"] = np.where(
        df["Harga Awal"] > 0,
        (df["Harga Awal"] - df["Harga Setelah Diskon"]) / df["Harga Awal"] * 100.0,
        0.0,
    )
    df["Selisih_Diskon"] = df["Harga Awal"] - df["Harga Setelah Diskon"]
    df["Punya_Variasi"] = df["Nama Variasi"].notna().astype(int)
    df["Judul_Kata_COD"] = (
        df["Nama Produk"].astype(str).str.upper().str.contains("COD").astype(int)
    )
    df["Is_COD"] = (df["Metode Pembayaran"] == "COD (Bayar di Tempat)").astype(int)
    df["Is_Hemat_Kargo"] = (
        df["Opsi Pengiriman"].astype(str).str.contains("Hemat").astype(int)
    )
    df["Is_Jawa_Timur"] = (df["Provinsi"] == "JAWA TIMUR").astype(int)

    # 4. Target klasifikasi biner: Fast-Moving (>= 30 unit/bulan) vs Slow-Moving (< 30 unit/bulan)
    prod_total_qty = df.groupby("Nama Produk")["Jumlah"].transform("sum")
    df["Target_Laris"] = (prod_total_qty >= 30).astype(int)
    df["Label_Kelarisan"] = np.where(
        df["Target_Laris"] == 1,
        "Laris / Fast-Moving (>= 30 unit)",
        "Kurang Laris / Slow-Moving (< 30 unit)",
    )
    return df_raw, df


@st.cache_resource
def train_and_evaluate_models(df: pd.DataFrame):
    """Melatih 3 model klasifikasi bebas Data Leakage (fit hanya pada X_train)."""
    X = df[FEATURE_COLS].copy()
    y = df["Target_Laris"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    pipelines = {
        "Logistic Regression (L2)": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
            ]
        ),
        "Decision Tree (CART)": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "clf",
                    DecisionTreeClassifier(
                        max_depth=5, min_samples_split=12, random_state=42
                    ),
                ),
            ]
        ),
        "Random Forest (45 Trees)": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=45,
                        max_depth=7,
                        min_samples_split=10,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }

    rows = []
    conf_matrices = {}
    for name, pipe in pipelines.items():
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]
        cm = confusion_matrix(y_test, y_pred)
        conf_matrices[name] = cm
        tn, fp, fn, tp = cm.ravel()
        rows.append(
            {
                "Algoritma": name,
                "Accuracy (%)": round(accuracy_score(y_test, y_pred) * 100, 2),
                "Precision (%)": round(precision_score(y_test, y_pred) * 100, 2),
                "Recall (%)": round(recall_score(y_test, y_pred) * 100, 2),
                "F1-Score (%)": round(f1_score(y_test, y_pred) * 100, 2),
                "ROC-AUC (%)": round(roc_auc_score(y_test, y_prob) * 100, 2),
                "TP": int(tp),
                "TN": int(tn),
                "FP": int(fp),
                "FN": int(fn),
            }
        )

    results_df = pd.DataFrame(rows)
    rf_pipe = pipelines["Random Forest (45 Trees)"]
    rf_imp = pd.DataFrame(
        {
            "Fitur": FEATURE_COLS,
            "Importance (%)": np.round(
                rf_pipe.named_steps["clf"].feature_importances_ * 100, 2
            ),
        }
    ).sort_values("Importance (%)", ascending=False)

    return results_df, conf_matrices, rf_pipe, rf_imp, len(X_train), len(X_test)


# ==============================================================================
# HEADER & SIDEBAR
# ==============================================================================
st.title("Dashboard Final Project Data Mining: Klasifikasi Tingkat Kelarisan Produk Shopee")
st.caption(
    "Kelompok: Anggoro Adit Haptono (23082010006) | Taufiq Tri Winardi (23082010008) | "
    "Riki Audy Bin Ghozali (23082010011) — Dataset: Transaksi Shopee Seller (Januari 2023)"
)

DATASET_FILE = "FP-DM-A-01-DATASET.csv"
if not os.path.exists(DATASET_FILE):
    st.error(
        f"File '{DATASET_FILE}' tidak ditemukan di direktori aplikasi. "
        "Pastikan file CSV mentah berada di folder yang sama dengan app.py."
    )
    st.stop()

df_raw, df = load_and_prepare_data(DATASET_FILE)
results_df, conf_matrices, rf_model, rf_imp, n_train, n_test = train_and_evaluate_models(df)

# Ringkasan KPI Utama
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Baris Transaksi", f"{len(df):,}")
k2.metric("Pesanan Unik / Produk Unik", f"{df['No. Pesanan'].nunique():,} / {df['Nama Produk'].nunique()}")
k3.metric("Proporsi Laris (>=30 unit)", f"{df['Target_Laris'].mean()*100:.2f}%")
best_acc = results_df.loc[results_df["Algoritma"] == "Random Forest (45 Trees)", "Accuracy (%)"].values[0]
k4.metric("Akurasi Terbaik (Random Forest)", f"{best_acc:.2f}%")

# ==============================================================================
# TAB VISUALISASI UTAMA (EDA, MODELING, FEATURE IMPORTANCE, SIMULATOR)
# ==============================================================================
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "1. Exploratory Data Analysis (EDA)",
        "2. Evaluasi Model Klasifikasi",
        "3. Feature Importance & Strategi Bisnis",
        "4. Simulator Prediksi Kelarisan Produk",
    ]
)

with tab1:
    st.subheader("Exploratory Data Analysis (Data Understanding)")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Enam Produk dengan Volume Penjualan Tertinggi (Unit)**")
        top6 = df.groupby("Nama Produk")["Jumlah"].sum().sort_values(ascending=True).tail(6)
        short_idx = [s[:36] + "..." if len(s) > 36 else s for s in top6.index]
        fig1, ax1 = plt.subplots(figsize=(6.2, 3.6))
        ax1.barh(short_idx, top6.values, color="#1e3a5f")
        ax1.set_xlabel("Total Unit Terjual (Januari 2023)")
        for i, v in enumerate(top6.values):
            ax1.text(v + 8, i, f"{int(v)} unit", va="center", fontsize=8.5, fontweight="bold")
        ax1.set_xlim(0, max(top6.values) * 1.22)
        st.pyplot(fig1)

    with c2:
        st.markdown("**Distribusi Kelas Target Kelarisan Produk (n = 2.211)**")
        cls_counts = df["Label_Kelarisan"].value_counts()
        fig2, ax2 = plt.subplots(figsize=(6.2, 3.6))
        bars = ax2.bar(cls_counts.index, cls_counts.values, color=["#1e3a5f", "#94a3b8"], width=0.5)
        ax2.set_ylabel("Jumlah Baris Transaksi")
        ax2.set_ylim(0, max(cls_counts.values) * 1.2)
        for b in bars:
            h = b.get_height()
            ax2.text(
                b.get_x() + b.get_width() / 2,
                h + 25,
                f"{int(h):,} ({h/len(df)*100:.2f}%)",
                ha="center",
                fontweight="bold",
                fontsize=9,
            )
        st.pyplot(fig2)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("**Distribusi Metode Pembayaran (Konteks Perilaku Pembeli)**")
        pay_df = df_raw["Metode Pembayaran"].fillna("Lainnya").value_counts().reset_index()
        pay_df.columns = ["Metode Pembayaran", "Jumlah Transaksi"]
        pay_df["Persentase (%)"] = (pay_df["Jumlah Transaksi"] / len(df_raw) * 100).round(2)
        st.dataframe(pay_df, use_container_width=True, hide_index=True)

    with c4:
        st.markdown("**Audit 12 Kolom dengan Missing Values pada Data Mentah**")
        null_s = df_raw.isna().sum()
        null_df = null_s[null_s > 0].sort_values(ascending=False).reset_index()
        null_df.columns = ["Nama Kolom", "Jumlah Kosong"]
        null_df["Persentase (%)"] = (null_df["Jumlah Kosong"] / len(df_raw) * 100).round(2)
        st.dataframe(null_df, use_container_width=True, hide_index=True)

with tab2:
    st.subheader(
        f"Hasil Modeling & Evaluation (Stratified Split: Train = {n_train:,} | Test = {n_test:,})"
    )
    st.markdown(
        "Seluruh proses imputasi median (`SimpleImputer`) dan standardisasi (`StandardScaler`) "
        "di-*fit* **hanya pada Data Latih** untuk menjamin proses bebas dari *data leakage*."
    )
    st.dataframe(results_df, use_container_width=True, hide_index=True)

    m1, m2 = st.columns([1.3, 1.0])
    with m1:
        st.markdown("**Perbandingan Skor Accuracy, F1-Score, dan ROC-AUC**")
        fig3, ax3 = plt.subplots(figsize=(6.5, 3.8))
        x = np.arange(len(results_df))
        w = 0.25
        b1 = ax3.bar(x - w, results_df["Accuracy (%)"], w, label="Accuracy", color="#64748b")
        b2 = ax3.bar(x, results_df["F1-Score (%)"], w, label="F1-Score", color="#1e3a5f")
        b3 = ax3.bar(x + w, results_df["ROC-AUC (%)"], w, label="ROC-AUC", color="#d97706")
        ax3.set_xticks(x)
        ax3.set_xticklabels(results_df["Algoritma"], fontsize=8.5)
        ax3.set_ylim(50, 110)
        ax3.set_ylabel("Skor Evaluasi (%)")
        ax3.legend(loc="upper left")
        for bars in [b1, b2, b3]:
            for bar in bars:
                h = bar.get_height()
                ax3.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + 1.0,
                    f"{h:.1f}",
                    ha="center",
                    fontsize=8,
                    fontweight="bold",
                )
        st.pyplot(fig3)

    with m2:
        st.markdown("**Confusion Matrix Model Terbaik (Random Forest)**")
        rf_cm = conf_matrices["Random Forest (45 Trees)"]
        fig4, ax4 = plt.subplots(figsize=(4.6, 3.8))
        sns.heatmap(
            rf_cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax4,
            xticklabels=["Pred: Kurang Laris", "Pred: Laris"],
            yticklabels=["Aktual: Kurang Laris", "Aktual: Laris"],
            annot_kws={"size": 12, "weight": "bold"},
        )
        st.pyplot(fig4)

with tab3:
    st.subheader("Kontribusi Fitur Utama (Random Forest Feature Importance)")
    top10_imp = rf_imp.head(10).sort_values("Importance (%)", ascending=True)
    fig5, ax5 = plt.subplots(figsize=(8.0, 4.4))
    ax5.barh(top10_imp["Fitur"], top10_imp["Importance (%)"], color="#1e3a5f")
    ax5.set_xlabel("Kontribusi Penurunan Gini Impurity (%)")
    ax5.set_xlim(0, max(top10_imp["Importance (%)"]) * 1.2)
    for i, v in enumerate(top10_imp["Importance (%)"]):
        ax5.text(v + 0.25, i, f"{v:.2f}%", va="center", fontweight="bold", fontsize=9)
    st.pyplot(fig5)

    st.info(
        "**Implikasi Bisnis (BAB IV):** Struktur harga dan diskon (Harga Setelah Diskon, Harga Awal, "
        "Selisih Diskon, Persentase Diskon) serta bobot paket (Berat Produk & Total Berat) merupakan "
        "faktor utama yang membedakan produk Fast-Moving dan Slow-Moving. Penjual disarankan "
        "mempertahankan strategi harga coret dan memprioritaskan stok pada produk ringan (<= 250 gram)."
    )

with tab4:
    st.subheader("Simulator Prediksi Tingkat Kelarisan Produk (Model Random Forest Aktif)")
    st.write(
        "Masukkan karakteristik harga, diskon, bobot, dan pengiriman produk untuk memprediksi "
        "apakah produk cenderung masuk kategori **Laris (Fast-Moving)** atau **Kurang Laris (Slow-Moving)**."
    )

    col_a, col_b, col_c = st.columns(3)
    harga_awal = col_a.number_input(
        "Harga Awal (ribuan Rp, contoh 15.0 = Rp15.000)", min_value=1.0, max_value=500.0, value=15.0, step=1.0
    )
    harga_diskon = col_b.number_input(
        "Harga Setelah Diskon (ribuan Rp)", min_value=0.5, max_value=500.0, value=7.5, step=0.5
    )
    berat_produk = col_c.number_input(
        "Berat Produk Satuan (gram)", min_value=10.0, max_value=20000.0, value=100.0, step=10.0
    )

    col_d, col_e, col_f = st.columns(3)
    jml_beli = col_d.slider("Jumlah Unit Dibeli per Transaksi", 1, 10, 1)
    jml_produk_order = col_e.slider("Jumlah Produk dalam 1 Pesanan", 1, 10, 1)
    punya_variasi = col_f.selectbox("Produk Memiliki Variasi?", options=[("Tidak", 0), ("Ya", 1)], format_func=lambda x: x[0])[1]

    col_g, col_h, col_i = st.columns(3)
    is_cod = col_g.selectbox("Metode Pembayaran COD?", options=[("Ya (COD)", 1), ("Tidak (Non-COD)", 0)], format_func=lambda x: x[0])[1]
    is_hemat = col_h.selectbox("Opsi Pengiriman Hemat Kargo?", options=[("Ya", 1), ("Tidak", 0)], format_func=lambda x: x[0])[1]
    is_jatim = col_i.selectbox("Tujuan Provinsi Jawa Timur?", options=[("Ya", 1), ("Luar Jawa Timur", 0)], format_func=lambda x: x[0])[1]

    persen_diskon = max(0.0, (harga_awal - harga_diskon) / harga_awal * 100.0) if harga_awal > 0 else 0.0
    selisih_diskon = max(0.0, harga_awal - harga_diskon)

    input_row = pd.DataFrame(
        [
            {
                "Harga Awal": harga_awal,
                "Harga Setelah Diskon": harga_diskon,
                "Persen_Diskon": persen_diskon,
                "Selisih_Diskon": selisih_diskon,
                "Berat_Produk_Gram": berat_produk,
                "Total_Berat_Gram": berat_produk * jml_beli,
                "Punya_Variasi": punya_variasi,
                "Judul_Kata_COD": 1 if is_cod == 1 else 0,
                "Jumlah": jml_beli,
                "Jumlah Produk di Pesan": jml_produk_order,
                "Ongkos Kirim Dibayar oleh Pembeli": 0.0,
                "Estimasi Potongan Biaya Pengiriman": 10.0,
                "Perkiraan Ongkos Kirim": 10.0,
                "Is_COD": is_cod,
                "Is_Hemat_Kargo": is_hemat,
                "Is_Jawa_Timur": is_jatim,
                "Jam_Pesan": 19,
                "Durasi_Penyelesaian_Jam": 72.0,
            }
        ]
    )

    prob_laris = float(rf_model.predict_proba(input_row)[0, 1]) * 100.0
    pred_cls = int(rf_model.predict(input_row)[0])

    if pred_cls == 1:
        st.success(
            f"**Hasil Prediksi: LARIS / FAST-MOVING (>= 30 unit/bulan)** — "
            f"Probabilitas Kelarisan: **{prob_laris:.1f}%** (Diskon efektif: {persen_diskon:.1f}%)."
        )
    else:
        st.warning(
            f"**Hasil Prediksi: KURANG LARIS / SLOW-MOVING (< 30 unit/bulan)** — "
            f"Probabilitas Kelarisan: **{prob_laris:.1f}%** (Diskon efektif: {persen_diskon:.1f}%)."
        )