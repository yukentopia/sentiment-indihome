import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- TAMBAHKAN IMPORT INI UNTUK LEVEL 2 ---
from streamlit_option_menu import option_menu
from wordcloud import WordCloud

# Import modul-modul buatanmu (sesuaikan dengan yang sudah kamu buat)
from modules.database import get_db_connection
from modules.text_processing import preprocess_text
from modules.machine_learning import get_tfidf_vectorizer, evaluate_model

import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score, matthews_corrcoef, confusion_matrix
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Konfigurasi Halaman (Harus ditaruh paling atas setelah import)
st.set_page_config(page_title="Sentimen Indihome", page_icon="📶", layout="wide")

# ==========================================
# 🔒 SISTEM AUTENTIKASI (LOGIN)
# ==========================================
# 1. Inisialisasi status login di memori aplikasi
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

# 2. Jika belum login, tampilkan halaman Login
if not st.session_state['logged_in']:
    st.markdown("<h1 style='text-align: center;'>🔒 Sentimen Analisis Login</h1>", unsafe_allow_html=True)
    st.write("<p style='text-align: center;'>Silakan masuk untuk mengakses Dashboard Administrator.</p>", unsafe_allow_html=True)
    
    # Bikin kolom agar form loginnya ada di tengah dan tidak terlalu lebar
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        with st.form("login_form"):
            st.write("### Masukkan Kredensial")
            username = st.text_input("Username", placeholder="Masukkan username...")
            password = st.text_input("Password", type="password", placeholder="Masukkan password...")
            
            submit_button = st.form_submit_button("Masuk / Login", use_container_width=True)
            
            if submit_button:
                # Kamu bisa mengubah username dan password ini sesuai keinginanmu
                if username == "admin" and password == "admin123":
                    st.session_state['logged_in'] = True
                    st.success("✅ Login berhasil! Memuat sistem...")
                    st.rerun() # Refresh halaman untuk masuk ke menu utama
                else:
                    st.error("❌ Username atau Password salah!")
                    
    # st.stop() akan memblokir semua kode menu di bawahnya agar tidak dirender
    st.stop()

# ==========================================
# --- LEVEL 3: INJEKSI CUSTOM CSS ---
# ==========================================
def load_custom_css():
    st.markdown("""
        <style>
        /* Sembunyikan menu bawaan streamlit di pojok kanan atas dan footer */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Modifikasi bentuk kartu/metrik agar ada bayangannya */
        div[data-testid="metric-container"] {
            background-color: #ffffff;
            border: 1px solid #e6e9ef;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.05);
        }
        
        /* Bikin tombol lebih membulat */
        .stButton>button {
            border-radius: 20px;
            transition: all 0.3s ease;
        }
        
        /* Efek hover pada tombol */
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0px 4px 10px rgba(230, 0, 0, 0.3);
        }
        </style>
    """, unsafe_allow_html=True)

# Panggil fungsi CSS-nya di sini
load_custom_css()
# ==========================================

# ==========================================
# --- LEVEL 2 NOMOR 3: MENU SIDEBAR ---
# ==========================================
# Hapus kode menu st.sidebar.selectbox lama kamu, dan ganti dengan ini:
with st.sidebar:
    # --- KODE SAKTI MENGHILANGKAN ROUNDED CORNER ---
    st.markdown(
        """
        <style>
            [data-testid="stSidebar"] img {
                border-radius: 0px !important;
            }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # --- KODE LOGO (YANG TADI SUDAH CENTER) ---
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("logo_indihome.png", use_container_width=True)
    
    st.write("---")
    
    menu = option_menu(
        menu_title="Main Menu",
        options=["Dashboard", "Dataset", "Labeling", "Preprocessing", "TF-IDF", "Modeling & Evaluation", "History & Comparison", "Word Cloud", "Prediksi Model"],
        icons=["house", "database", "tags", "gear", "calculator", "bar-chart-line", "clock-history", "cloud", "robot"], 
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "5!important", "background-color": "#FAFAFA"},
            "icon": {"color": "#E60000", "font-size": "18px"}, 
            "nav-link": {"font-size": "15px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
            "nav-link-selected": {"background-color": "#262730", "color": "white"},
        }
    )

    # ==========================================
    # TOMBOL LOGOUT (Paling Bawah Sidebar)
    # ==========================================
    st.markdown("<br><br>", unsafe_allow_html=True) # Memberi jarak kosong agar tombol turun ke bawah
    st.markdown("---") # Garis pembatas
    
    if st.button("🚪 Keluar / Logout", use_container_width=True, type="secondary"):
        # Jika diklik, ubah status login menjadi False dan bersihkan session state yang lain
        st.session_state['logged_in'] = False
        
        # Bersihkan juga session evaluasi agar tidak bocor ke user selanjutnya
        if 'eval_results' in st.session_state:
            del st.session_state['eval_results']
            
        st.rerun() # Refresh halaman untuk kembali ke gerbang login

# --- 1. FUNGSI UNTUK MENGUBAH TEKS JADI ANGKA (TF-IDF) ---
def get_tfidf_vectorizer(text_data):
    vectorizer = TfidfVectorizer()
    X_tfidf = vectorizer.fit_transform(text_data)
    return X_tfidf, vectorizer

# --- 2. FUNGSI UNTUK MELATIH DAN MENGEVALUASI MODEL (Sesuai Colab K-Fold) ---
def evaluate_model(model_name, X, y, k_fold, use_smote):
    # 1. Pilih Algoritma (Parameter disamakan 100% dengan Colab)
    if model_name == "Support Vector Machine (SVM)":
        clf = SVC(kernel='linear', random_state=42)
    elif model_name == "Naive Bayes":
        clf = MultinomialNB()
    elif model_name == "Decision Tree":
        clf = DecisionTreeClassifier(random_state=42)
    else:
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        
    # 2. Gunakan Pipeline agar SMOTE hanya diterapkan pada Data Latih di setiap Fold
    if use_smote:
        smote = SMOTE(random_state=42)
        pipeline = Pipeline([('smote', smote), ('classifier', clf)])
    else:
        pipeline = Pipeline([('classifier', clf)])
        
    # 3. Stratified K-Fold Cross Validation & Prediksi
    cv = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=42)
    y_pred = cross_val_predict(pipeline, X, y, cv=cv)
    
    # 4. Hitung Semua Metrik Evaluasi
    acc = accuracy_score(y, y_pred) * 100
    prec = precision_score(y, y_pred, average='weighted', zero_division=0) * 100
    rec = recall_score(y, y_pred, average='weighted', zero_division=0) * 100
    f1_weighted = f1_score(y, y_pred, average='weighted', zero_division=0) * 100
    
    # Metrik Khusus Imbalanced Data
    f1_macro = f1_score(y, y_pred, average='macro', zero_division=0)
    bal_acc = balanced_accuracy_score(y, y_pred)
    mcc = matthews_corrcoef(y, y_pred)
    
    cm = confusion_matrix(y, y_pred, labels=["negatif", "netral", "positif"])
    
    # Harus urut mengembalikan 8 nilai metrik
    return acc, prec, rec, f1_weighted, f1_macro, bal_acc, mcc, cm

# ----------------- 1. DASHBOARD -----------------
if menu == "Dashboard":  # <--- INI YANG DIPERBAIKI (pakai 'if', bukan 'elif')
    st.title("📊 Dashboard Analisis Sentimen Indihome")
    st.write("Sistem Klasifikasi Sentimen Layanan Indihome di Platform X.")
    st.markdown("---")
    
    conn = get_db_connection()
    try:
        df = pd.read_sql("SELECT * FROM dataset_tweets", conn)
    except:
        df = pd.DataFrame()
    conn.close()
    
    if df.empty:
        st.info("Dataset masih kosong. Silakan upload data di menu 'Dataset'.")
    else:
        # Cek apakah kolom label sudah ada di database
        if 'label' in df.columns and not df['label'].isnull().all():
            
            # Bersihkan teks label (ubah jadi huruf kecil semua untuk akurasi hitungan)
            df['label_bersih'] = df['label'].astype(str).str.strip().str.lower()
            
            total_data = len(df)
            total_positif = len(df[df['label_bersih'] == 'positif'])
            total_negatif = len(df[df['label_bersih'] == 'negatif'])
            total_netral = len(df[df['label_bersih'] == 'netral'])
            
            # --- BAGIAN 1: METRIK ANGKA (DIBUAT 4 KOLOM) ---
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Dataset", total_data)
            col2.metric("🟢 Positif", total_positif)
            col3.metric("🔴 Negatif", total_negatif)
            col4.metric("🟡 Netral", total_netral)
            
            st.markdown("---")
            
            # --- BAGIAN 2: VISUALISASI GRAFIK (AGAR TIDAK BASIC) ---
            st.write("### 📈 Visualisasi Distribusi Sentimen")
            
            import matplotlib.pyplot as plt
            
            # Membagi layar jadi 2 (Kiri grafik, Kanan penjelasan)
            col_chart, col_desc = st.columns([1.5, 1])
            
            with col_chart:
                fig, ax = plt.subplots(figsize=(6, 4))
                labels = ['Positif', 'Negatif', 'Netral']
                sizes = [total_positif, total_negatif, total_netral]
                colors = ['#28a745', '#dc3545', '#ffc107'] # Kode warna: Hijau, Merah, Kuning
                
                # Pastikan ada data yang bisa diplot
                if sum(sizes) > 0:
                    # Membuat Donut Chart (Pie chart berlubang tengah) agar lebih modern
                    wedges, texts, autotexts = ax.pie(
                        sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors,
                        wedgeprops=dict(width=0.4, edgecolor='w'),
                        pctdistance=0.80 # <--- INI TAMBAHANNYA
                    )
                    # Mempertebal font persen di dalam grafik
                    plt.setp(autotexts, size=10, weight="bold", color="white")
                    
                    ax.axis('equal') # Pastikan bulat sempurna
                    st.pyplot(fig)
                else:
                    st.warning("Belum ada label valid yang bisa ditampilkan dalam grafik.")
                    
            with col_desc:
                st.write("**Keterangan Distribusi Data:**")
                st.write("- **🟢 Positif:** Tanggapan baik, kepuasan pelanggan, atau kondisi jaringan lancar.")
                st.write("- **🔴 Negatif:** Keluhan, masalah jaringan (lemot/RTO), atau kekecewaan pelanggan.")
                st.write("- **🟡 Netral:** Pertanyaan umum, informasi layanan, atau cuitan yang tidak mengandung sentimen spesifik.")
                
                # Cek jika data tidak seimbang (Imbalanced) untuk memberi hint/informasi tambahan
                if total_netral > (total_positif + total_negatif):
                    st.info("💡 **Catatan:** Data didominasi oleh sentimen Netral (Imbalanced Data). Disarankan menggunakan metode **SMOTE** saat melakukan *Training* model.")
                
        else:
            # Jika dataset sudah diupload tapi belum di-labeling
            st.warning("Data belum dilabeli. Jumlah Positif, Negatif, dan Netral belum bisa ditampilkan.")
            col1, col2 = st.columns(2)
            col1.metric("Total Dataset", len(df))
            col2.metric("Status Data", "Belum Dilabeli ⏳")
            st.info("Silakan buka menu **Labeling** untuk mulai memberi label pada dataset ini.")

# ----------------- 2. DATASET -----------------
elif menu == "Dataset":
    st.title("📂 Manajemen Dataset (CRUD)")
    
    # --- PERUBAHAN: PANEL AKSI (CRUD) SEKARANG DI TENGAH ---
    tab_tabel, tab_aksi, tab_upload = st.tabs([
        "📋 Tabel Data", 
        "⚙️ Panel Aksi (CRUD)",
        "📤 Upload CSV Baru"
    ])
    
    # ========================================================
    # TAB 1: TAMPILKAN DATABASE, PENCARIAN & PAGINATION
    # ========================================================
    with tab_tabel:
        conn = get_db_connection()
        try:
            df_db = pd.read_sql("SELECT id, original_tweet, clean_tweet, label FROM dataset_tweets ORDER BY id DESC", conn)
        except:
            df_db = pd.DataFrame()
            
        if not df_db.empty:
            # --- 1. MENYANDINGKAN PENCARIAN DAN LIMIT BARIS ---
            col_search, col_limit = st.columns([3, 1])
            
            with col_search:
                search_query = st.text_input("🔍 Cari tweet atau label...", "")
            
            with col_limit:
                per_page = st.selectbox("Tampilkan baris:", [10, 20, 50, 100], index=0)
                
            # Filter pencarian
            if search_query:
                df_db = df_db[
                    df_db['original_tweet'].str.contains(search_query, case=False, na=False) |
                    df_db['label'].str.contains(search_query, case=False, na=False)
                ]
            
            # --- 2. FITUR PAGINATION ---
            total_data = len(df_db)
            total_pages = max(1, (total_data - 1) // per_page + 1)
            
            if 'current_page' not in st.session_state:
                st.session_state['current_page'] = 1
                
            if st.session_state['current_page'] > total_pages:
                st.session_state['current_page'] = 1
                
            start_idx = (st.session_state['current_page'] - 1) * per_page
            end_idx = start_idx + per_page
            df_display = df_db.iloc[start_idx:end_idx]
            
            # --- 3. TAMPILAN TABEL ---
            st.dataframe(df_display, use_container_width=True, hide_index=True)
            
            # Tombol Navigasi Pagination
            c_prev, c_info, c_next = st.columns([1, 2, 1])
            with c_prev:
                if st.button("⬅️ Sebelumnya", use_container_width=True) and st.session_state['current_page'] > 1:
                    st.session_state['current_page'] -= 1
                    st.rerun()
            with c_info:
                st.markdown(f"<div style='text-align: center; padding-top: 5px;'><b>Halaman {st.session_state['current_page']} dari {total_pages}</b> (Total Data: {total_data})</div>", unsafe_allow_html=True)
            with c_next:
                if st.button("Selanjutnya ➡️", use_container_width=True) and st.session_state['current_page'] < total_pages:
                    st.session_state['current_page'] += 1
                    st.rerun()
        else:
            st.info("Database MySQL saat ini masih kosong. Silakan upload CSV di tab sebelah.")
        conn.close()

    # ========================================================
    # TAB 2: PANEL AKSI CRUD (POSISI DI TENGAH)
    # ========================================================
    with tab_aksi:
        st.write("### 🛠️ Panel Aksi (Create, Update, Delete)")
        conn = get_db_connection()
        aksi = st.radio("Pilih Tindakan:", ["➕ Tambah Data", "✏️ Edit Data", "🗑️ Hapus Data"], horizontal=True)
        
        if aksi == "➕ Tambah Data":
            with st.form("form_tambah"):
                st.write("**Tambah Tweet Baru**")
                new_tweet = st.text_area("Teks Tweet:", placeholder="Masukkan teks ulasan di sini...")
                new_label = st.selectbox("Label (Opsional):", ["Belum Dilabeli", "Positif", "Negatif", "Netral"])
                
                if st.form_submit_button("Simpan Data", type="primary"):
                    if new_tweet.strip():
                        cursor = conn.cursor()
                        lbl = None if new_label == "Belum Dilabeli" else new_label.lower()
                        cursor.execute("INSERT INTO dataset_tweets (original_tweet, label) VALUES (%s, %s)", (new_tweet, lbl))
                        conn.commit()
                        cursor.close()
                        st.success("✅ Data berhasil ditambahkan!")
                        st.rerun()
                    else:
                        st.warning("Teks tweet tidak boleh kosong!")

        elif aksi == "✏️ Edit Data":
            st.write("**Edit Data Berdasarkan ID**")
            col_edit1, col_edit2 = st.columns([1, 3])
            with col_edit1:
                edit_id = st.number_input("Masukkan ID Data:", min_value=1, step=1)
            
            with col_edit2:
                # Cek langsung dari database spesifik menggunakan ID
                try:
                    df_target = pd.read_sql(f"SELECT * FROM dataset_tweets WHERE id={edit_id}", conn)
                except:
                    df_target = pd.DataFrame()
                    
                if not df_target.empty:
                    with st.form("form_edit"):
                        tweet_lama = df_target.iloc[0]['original_tweet']
                        label_lama = str(df_target.iloc[0]['label']).capitalize() if pd.notna(df_target.iloc[0]['label']) else "Belum Dilabeli"
                        
                        edit_tweet = st.text_area("Teks Tweet:", value=tweet_lama)
                        options = ["Belum Dilabeli", "Positif", "Negatif", "Netral"]
                        edit_label = st.selectbox("Label:", options, index=options.index(label_lama) if label_lama in options else 0)
                        
                        if st.form_submit_button("Update Data", type="primary"):
                            cursor = conn.cursor()
                            lbl = None if edit_label == "Belum Dilabeli" else edit_label.lower()
                            cursor.execute("UPDATE dataset_tweets SET original_tweet=%s, label=%s WHERE id=%s", (edit_tweet, lbl, edit_id))
                            conn.commit()
                            cursor.close()
                            st.success(f"✅ Data ID {edit_id} berhasil diperbarui!")
                            st.rerun()
                else:
                    st.info("Silakan masukkan ID yang valid untuk memunculkan form edit.")

        elif aksi == "🗑️ Hapus Data":
            st.write("**Hapus Data Berdasarkan ID**")
            col_del1, col_del2 = st.columns([1, 3])
            
            with col_del1:
                del_id = st.number_input("Masukkan ID Data:", min_value=1, step=1, key="del_input_final")
                
            with col_del2:
                # Cek ke database apakah ID tersebut ada
                try:
                    df_del_target = pd.read_sql(f"SELECT * FROM dataset_tweets WHERE id={del_id}", conn)
                except:
                    df_del_target = pd.DataFrame()
                    
                if not df_del_target.empty:
                    st.warning("⚠️ Pratinjau Data yang akan dihapus:")
                    
                    # Ambil teks dan labelnya
                    tweet_to_delete = df_del_target.iloc[0]['original_tweet']
                    label_to_delete = str(df_del_target.iloc[0]['label']).capitalize() if pd.notna(df_del_target.iloc[0]['label']) else "Belum Dilabeli"
                    
                    # Tampilkan dalam kotak info agar rapi
                    st.info(f"**Tweet:** {tweet_to_delete}\n\n**Label:** {label_to_delete}")
                    
                    # Tombol hapus hanya muncul JIKA data ditemukan
                    if st.button("🗑️ Konfirmasi Hapus Data", type="primary"):
                        cursor = conn.cursor()
                        cursor.execute("DELETE FROM dataset_tweets WHERE id=%s", (del_id,))
                        conn.commit()
                        cursor.close()
                        st.success(f"✅ Data ID {del_id} berhasil dihapus secara permanen!")
                        st.rerun()
                else:
                    # Spacing agar tulisan rata dengan input box
                    st.write("") 
                    st.write("")
                    st.info(f"Masukkan ID yang valid. Data dengan ID {del_id} tidak ditemukan.")
                    
        # Tombol Truncate (Hapus Semua)
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("⚠️ Hapus Seluruh Database"):
            st.warning("Tindakan ini akan mengosongkan seluruh tabel dataset. Tidak bisa dibatalkan.")
            if st.button("Kosongkan Database MySQL"):
                cursor = conn.cursor()
                cursor.execute("TRUNCATE TABLE dataset_tweets")
                conn.commit()
                cursor.close()
                st.success("Database berhasil dikosongkan!")
                st.rerun()
        conn.close()

    # ========================================================
    # TAB 3: UPLOAD & OLAH CSV BARU
    # ========================================================
    with tab_upload:
        st.write("### 📤 Upload CSV Baru")
        
        if 'current_dataset' in st.session_state and st.session_state['current_dataset'] is not None:
            st.success("Terdapat file CSV yang sedang diolah saat ini.")
            df_csv = st.session_state['current_dataset']
            
            st.write("#### 1. Pratinjau CSV")
            st.dataframe(df_csv.head(10))
            st.write(f"Total baris: **{len(df_csv)}** | Total kolom: **{len(df_csv.columns)}**")
            
            st.markdown("---")
            
            st.write("#### 2. Bersihkan Data (Hapus Duplikat)")
            current_columns = df_csv.columns.tolist()
            col_dup = st.selectbox("Pilih kolom acuan (teks tweet):", current_columns, key="select_dup_final")
            
            if st.button("Hapus Duplikat", key="btn_hapus_dup_final"):
                initial_len = len(df_csv)
                df_clean = df_csv.drop_duplicates(subset=[col_dup])
                removed_count = initial_len - len(df_clean)
                st.session_state['current_dataset'] = df_clean
                
                if removed_count > 0:
                    st.success(f"Berhasil menghapus {removed_count} baris duplikat!")
                else:
                    st.warning("Tidak ditemukan data duplikat.")
                st.rerun()
                
            st.markdown("---")
            
            st.write("#### 3. Simpan ke Database MySQL")
            st.info("⚠️ Tindakan ini akan MENIMPA (menghapus) data lama di database dengan data CSV ini.")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                col_tweet = st.selectbox("Kolom Teks Asli:", current_columns, key="select_tweet_final")
            with col2:
                label_options = ["-- Tidak Ada Label --"] + current_columns
                col_label = st.selectbox("Kolom Label (Opsional):", label_options, key="select_label_final")
            with col3:
                clean_options = ["-- Tidak Ada Teks Bersih --"] + current_columns
                col_clean = st.selectbox("Kolom Teks Bersih (Opsional):", clean_options, key="select_clean_final")
            
            if st.button("Simpan Dataset ke MySQL", key="btn_simpan_db_final"):
                conn = get_db_connection()
                cursor = conn.cursor()
                
                with st.spinner("Menyimpan data ke database..."):
                    cursor.execute("TRUNCATE TABLE dataset_tweets")
                    
                    try: cursor.execute("ALTER TABLE dataset_tweets ADD COLUMN clean_tweet TEXT")
                    except: pass
                    try: cursor.execute("ALTER TABLE dataset_tweets ADD COLUMN label VARCHAR(20)")
                    except: pass
                    
                    sql = "INSERT INTO dataset_tweets (original_tweet, label, clean_tweet) VALUES (%s, %s, %s)"
                    val = []
                    for index, row in df_csv.iterrows():
                        tweet_val = row[col_tweet]
                        label_val = None if col_label == "-- Tidak Ada Label --" else row[col_label]
                        clean_val = None if col_clean == "-- Tidak Ada Teks Bersih --" else row[col_clean]
                        val.append((tweet_val, label_val, clean_val))
                    
                    batch_size = 1000
                    for i in range(0, len(val), batch_size):
                        batch = val[i:i + batch_size]
                        cursor.executemany(sql, batch)
                    conn.commit()
                
                cursor.execute("SELECT COUNT(*) FROM dataset_tweets")
                total_saved = cursor.fetchone()[0]
                cursor.close()
                conn.close()
                
                st.success(f"Berhasil menyimpan {total_saved} baris ke database!")
                st.session_state['current_dataset'] = None
                st.rerun()
                
            st.markdown("---")
            if st.button("Batal & Hapus CSV dari Memori", type="primary", key="btn_batal_csv_final"):
                st.session_state['current_dataset'] = None
                st.rerun()
                
        else:
            st.info("Belum ada file CSV yang diunggah.")
            # Kunci (key) di sini sudah saya ubah agar error duplikatnya hilang
            uploaded_file = st.file_uploader("Upload file CSV", type=["csv"], key="uploader_dataset_final")
            
            if uploaded_file is not None:
                df_csv = pd.read_csv(uploaded_file)
                st.session_state['current_dataset'] = df_csv
                st.rerun()

# ----------------- 3. LABELING -----------------
elif menu == "Labeling":
    st.title("🏷️ Labeling Sentimen")
    st.write("Beri label sentimen secara otomatis menggunakan pendekatan Leksikon (Kamus) dengan tambahan Kamus Khusus Jaringan Internet.")
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 1. Buat tabel kamus custom otomatis jika belum ada di database
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS custom_lexicon (
            id INT AUTO_INCREMENT PRIMARY KEY,
            word VARCHAR(255) UNIQUE,
            score INT
        )
    """)
    conn.commit()
    
    # Gunakan Tabs agar tampilan rapi
    tab1, tab2 = st.tabs(["⚙️ Proses Labeling Otomatis", "📖 Manajemen Kamus Jaringan"])
    
    # ==========================================
    # TAB 2: MANAJEMEN KAMUS KHUSUS
    # ==========================================
    with tab2:
        st.write("### Tambah Kata Khusus Jaringan (Indihome)")
        st.info("Contoh: 'lemot' (Skor: -2), 'los' (Skor: -3), 'lancar' (Skor: 2), 'rto' (Skor: -1)")
        
        col_form1, col_form2, col_form3 = st.columns([2, 1, 1])
        with col_form1:
            input_word = st.text_input("Kata / Istilah Jaringan (Gunakan huruf kecil):")
        with col_form2:
            input_score = st.number_input("Skor (-5 s/d 5):", min_value=-5, max_value=5, value=0)
        with col_form3:
            st.write("") # Spacing agar tombol sejajar
            st.write("")
            if st.button("➕ Tambah Kata", type="primary"):
                if input_word.strip() != "":
                    try:
                        cursor.execute("INSERT INTO custom_lexicon (word, score) VALUES (%s, %s)", 
                                      (input_word.strip().lower(), input_score))
                        conn.commit()
                        st.success(f"Kata '{input_word}' berhasil ditambahkan!")
                        st.rerun()
                    except Exception as e:
                        st.error("Gagal menambahkan. Kata mungkin sudah ada di dalam kamus.")
                else:
                    st.warning("Kata tidak boleh kosong!")
                    
        st.markdown("---")
        st.write("### 📚 Daftar Kamus Jaringan Saat Ini")
        df_lexicon = pd.read_sql("SELECT * FROM custom_lexicon", conn)
        
        if not df_lexicon.empty:
            st.dataframe(df_lexicon, use_container_width=True, hide_index=True)
            
            # Fitur Hapus Kata
            del_word = st.selectbox("Pilih kata yang ingin dihapus dari kamus khusus:", df_lexicon['word'].tolist())
            if st.button("🗑️ Hapus Kata"):
                cursor.execute("DELETE FROM custom_lexicon WHERE word = %s", (del_word,))
                conn.commit()
                st.success(f"Kata '{del_word}' berhasil dihapus!")
                st.rerun()
        else:
            st.info("Kamus khusus jaringan masih kosong. Silakan tambahkan kata di atas.")

    # ==========================================
    # TAB 1: PROSES LABELING OTOMATIS
    # ==========================================
    with tab1:
        try:
            df = pd.read_sql("SELECT id, clean_tweet FROM dataset_tweets", conn)
        except:
            df = pd.DataFrame()
            
        if df.empty or 'clean_tweet' not in df.columns or df['clean_tweet'].isnull().all():
            st.warning("Data bersih (clean_tweet) belum tersedia. Lakukan Preprocessing terlebih dahulu di menu sebelumnya!")
        else:
            st.write(f"Total data siap dilabeli: **{len(df)} baris**")
            
            # --- 1. MEMBACA KAMUS UMUM DARI FOLDER (FORMAT EXCEL) ---
            kata_positif = []
            kata_negatif = []
            
            try:
                # Membaca file Excel kamus umum
                df_pos = pd.read_excel('kamus_positive.xlsx')
                df_neg = pd.read_excel('kamus_negative.xlsx')
                
                # Ambil data dari kolom pertama (index 0), buang yang kosong, jadikan string, ubah ke huruf kecil
                kata_positif = df_pos.iloc[:, 0].dropna().astype(str).str.lower().tolist()
                kata_negatif = df_neg.iloc[:, 0].dropna().astype(str).str.lower().tolist()
                
            except FileNotFoundError:
                st.error("File 'kamus_positive.xlsx' atau 'kamus_negative.xlsx' tidak ditemukan di folder proyek!")
            except ImportError:
                st.error("Library 'openpyxl' belum terpasang. Matikan server, ketik 'pip install openpyxl' di terminal, lalu jalankan ulang.")
            except Exception as e:
                st.error(f"Terjadi kesalahan saat membaca kamus Excel: {e}")
            
            # --- 2. KAMUS KHUSUS JARINGAN (DARI DATABASE) ---
            custom_dict = {}
            if not df_lexicon.empty:
                custom_dict = dict(zip(df_lexicon['word'], df_lexicon['score']))
            
            if st.button("Mulai Labeling Otomatis", type="primary"):
                progress_text = "Menghitung skor sentimen berdasarkan Kamus Excel & Jaringan..."
                my_bar = st.progress(0, text=progress_text)
                
                labeled_data = []
                total_data = len(df)
                
                for i, row in df.iterrows():
                    # Ambil tweet yang sudah di-cleansing dan pisahkan per kata
                    tweet = str(row['clean_tweet']).split()
                    skor = 0
                    
                    for kata in tweet:
                        # 1. Cek di kamus khusus jaringan (prioritas utama)
                        if kata in custom_dict:
                            skor += custom_dict[kata]
                        # 2. Jika tidak ada, cek di kamus umum (Excel)
                        elif kata in kata_positif:
                            skor += 1
                        elif kata in kata_negatif:
                            skor -= 1
                            
                    # Tentukan label berdasarkan total skor sentimen
                    if skor > 0:
                        label = 'Positif'
                    elif skor < 0:
                        label = 'Negatif'
                    else:
                        label = 'Netral'
                        
                    labeled_data.append((label, skor, row['id']))
                    
                    # Update progress bar
                    my_bar.progress(int(((i + 1) / total_data) * 100))
                
                # Simpan ke database
                with st.spinner("Menyimpan hasil label ke database..."):
                    # Tambahkan kolom label secara terpisah (abaikan jika sudah ada)
                    try:
                        cursor.execute("ALTER TABLE dataset_tweets ADD COLUMN label VARCHAR(20)")
                    except:
                        pass
                        
                    # Tambahkan kolom sentiment_score secara terpisah (abaikan jika sudah ada)
                    try:
                        cursor.execute("ALTER TABLE dataset_tweets ADD COLUMN sentiment_score INT")
                    except:
                        pass
                    
                    # Update data menggunakan batch (executemany) agar lebih cepat
                    sql_update = "UPDATE dataset_tweets SET label = %s, sentiment_score = %s WHERE id = %s"
                    cursor.executemany(sql_update, labeled_data)
                    conn.commit()
                
                my_bar.empty() # Hilangkan progress bar
                st.success("🎉 Labeling Otomatis Selesai dan Berhasil Disimpan!")
                st.rerun()

            # Tampilkan Hasil Pratinjau setelah berhasil di-labeling
            try:
                df_labeled = pd.read_sql("SELECT id, clean_tweet, sentiment_score, label FROM dataset_tweets WHERE label IS NOT NULL LIMIT 10", conn)
                if not df_labeled.empty:
                    st.write("### 🔍 Hasil Labeling Terakhir (Pratinjau 10 Baris)")
                    
                    # --- KODE YANG DIPERBAIKI (KEBAL HURUF KECIL/BESAR) ---
                    def color_label(val):
                        val_bersih = str(val).strip().lower()
                        if val_bersih == 'positif': return 'background-color: #d4edda; color: green;'
                        elif val_bersih == 'negatif': return 'background-color: #f8d7da; color: red;'
                        return 'background-color: #fff3cd; color: #856404;' # Warna Netral
                    
                    st.dataframe(df_labeled.style.map(color_label, subset=['label']), use_container_width=True)
            except:
                pass

    cursor.close()
    conn.close()
    
# ----------------- 4. PREPROCESSING -----------------
elif menu == "Preprocessing":
    st.title("⚙️ Preprocessing Teks")
    st.write("Pilih tahapan preprocessing yang ingin digunakan untuk membersihkan dataset.")
    
    # --- UI CHECKBOX / CEKLIS TAHAPAN ---
    st.markdown("### 🛠️ Pengaturan Tahapan Preprocessing")
    col_cek1, col_cek2 = st.columns(2)
    
    with col_cek1:
        # Menambahkan checkbox emoji di urutan pertama
        chk_emoji = st.checkbox("1. Emoji to Text (Ubah emoji jadi teks)", value=False) 
        chk_cleansing = st.checkbox("2. Cleansing (Hapus URL, Mention, Simbol)", value=True)
        chk_casefolding = st.checkbox("3. Case Folding (Ubah ke huruf kecil)", value=True)
        chk_tokenization = st.checkbox("4. Tokenization (Pemecahan kalimat jadi kata)", value=True)
        
    with col_cek2:
        chk_normalization = st.checkbox("5. Normalization (Perbaikan kata tidak baku)", value=True)
        chk_stopword = st.checkbox("6. Stopword Removal (Hapus kata hubung)", value=True)
        chk_stemming = st.checkbox("7. Stemming (Ubah ke kata dasar - Sastrawi)", value=True)
        
    st.markdown("---")
    
    conn = get_db_connection()
    try:
        df = pd.read_sql("SELECT id, original_tweet FROM dataset_tweets", conn)
    except:
        df = pd.DataFrame()
        
    if df.empty:
        st.warning("Data belum tersedia. Silakan upload dataset terlebih dahulu.")
    else:
        st.write(f"Total data yang siap diproses: **{len(df)} baris**")
        
        # Pratinjau data mentah
        st.write("### Data Mentah (Original Tweet)")
        st.dataframe(df.head(5), use_container_width=True)
        
        st.markdown("---")
        
        if st.button("Mulai Preprocessing Data", type="primary", key="btn_prep"):
            # Peringatan waktu
            if chk_stemming:
                progress_text = "Memproses teks... Mohon tunggu (Proses Stemming membutuhkan waktu yang lumayan lama)."
            else:
                progress_text = "Memproses teks dengan cepat (Tanpa Stemming)..."
                
            my_bar = st.progress(0, text=progress_text)
            
            clean_tweets = []
            total_data = len(df)
            
            # Looping per baris agar bisa update progress bar
            for i, row in df.iterrows():
                # Memanggil fungsi dengan parameter tambahan use_emoji
                hasil_bersih = preprocess_text(
                    row['original_tweet'], 
                    use_emoji=chk_emoji,          # <--- INI PARAMETER BARUNYA
                    use_cleansing=chk_cleansing, 
                    use_casefolding=chk_casefolding, 
                    use_tokenization=chk_tokenization, 
                    use_normalization=chk_normalization, 
                    use_stopword=chk_stopword, 
                    use_stemming=chk_stemming
                )
                
                clean_tweets.append((hasil_bersih, row['id']))
                
                # Update progress bar
                persentase = int(((i + 1) / total_data) * 100)
                my_bar.progress(persentase, text=f"Memproses baris {i+1} dari {total_data} ({persentase}%)")
            
            # Selesai processing, simpan ke database
            with st.spinner("Menyimpan hasil bersih ke database..."):
                cursor = conn.cursor()
                try:
                    cursor.execute("ALTER TABLE dataset_tweets ADD COLUMN clean_tweet TEXT")
                except:
                    pass # Abaikan jika kolom sudah ada
                
                sql_update = "UPDATE dataset_tweets SET clean_tweet = %s WHERE id = %s"
                
                batch_size = 1000
                for i in range(0, len(clean_tweets), batch_size):
                    batch = clean_tweets[i:i + batch_size]
                    cursor.executemany(sql_update, batch)
                
                conn.commit()
                cursor.close()
                
            my_bar.empty()
            st.success("🎉 Preprocessing Selesai dan Berhasil Disimpan!")
            st.rerun()
            
        # Tampilkan Hasil yang sudah ada di database
        try:
            df_clean = pd.read_sql("SELECT id, original_tweet, clean_tweet FROM dataset_tweets WHERE clean_tweet IS NOT NULL LIMIT 10", conn)
            if not df_clean.empty:
                st.write("### 🔍 Hasil Preprocessing Terakhir (Pratinjau 10 Baris)")
                st.dataframe(df_clean, use_container_width=True)
        except:
            pass

    conn.close()

# ----------------- 5. TF-IDF -----------------
elif menu == "TF-IDF":
    st.title("Ekstraksi Fitur (TF-IDF)")
    conn = get_db_connection()
    df = pd.read_sql("SELECT clean_tweet FROM dataset_tweets WHERE clean_tweet IS NOT NULL", conn)
    conn.close()
    
    if df.empty or df['clean_tweet'].isna().all():
        st.warning("Lakukan preprocessing terlebih dahulu.")
    else:
        if st.button("Generate TF-IDF"):
            X_tfidf, vectorizer = get_tfidf_vectorizer(df['clean_tweet'])
            
            # Ubah ke dataframe untuk visualisasi (ambil sebagian fitur)
            df_tfidf = pd.DataFrame(X_tfidf.toarray(), columns=vectorizer.get_feature_names_out())
            st.success(f"Berhasil! Ukuran matriks TF-IDF: {df_tfidf.shape}")
            
            # --- TABEL 1: MATRIKS ASLI (BAWAAN SEBELUMNYA) ---
            st.write("Sampel Matriks TF-IDF:")
            st.dataframe(df_tfidf.iloc[:10, :20]) # Tampilkan 10 baris, 20 kolom pertama
            
            # --- TABEL 2: RANGKUMAN KATA TERTINGGI (DISAMAKAN DENGAN COLAB) ---
            st.markdown("---")
            st.write("### 🏆 20 Kata dengan Skor TF-IDF Tertinggi")
            st.info("Tabel ini menunjukkan kata-kata dengan nilai RATA-RATA skor TF-IDF tertinggi di seluruh dataset.")
            
            import numpy as np
            
            # 1. Hitung nilai rata-rata (mean) per kolom (per kata) dari seluruh dataset
            mean_tfidf = np.asarray(X_tfidf.mean(axis=0)).ravel()
            
            # 2. Dapatkan daftar nama kata dan pasangkan dengan skor rata-ratanya
            feature_names = vectorizer.get_feature_names_out()
            tfidf_scores = pd.DataFrame({'Kata': feature_names, 'Skor TF-IDF': mean_tfidf})
            
            # 3. Urutkan dari skor tertinggi ke terendah
            df_top_words = tfidf_scores.sort_values(by='Skor TF-IDF', ascending=False).reset_index(drop=True)
            
            # 4. Tampilkan tabelnya
            st.dataframe(df_top_words.head(20), use_container_width=True)

# ----------------- 6. TRAINING (K-FOLD) -----------------
elif menu == "Training (K-Fold CV)":
    st.title("Pengaturan K-Fold Cross Validation")
    k_value = st.slider("Pilih nilai K (Lipatan):", min_value=2, max_value=10, value=5)
    st.session_state['k_fold'] = k_value
    st.info(f"Model akan dievaluasi menggunakan {k_value}-Fold Cross Validation.")

# ----------------- 7. MODELING & EVALUATION -----------------
elif menu == "Modeling & Evaluation":
    st.title("⚙️ Modeling dan Evaluasi (K-Fold CV)")
    
    # --- FITUR BARU: TOMBOL KHUSUS UNTUK SIDEBAR ---
    st.write("### 🧠 Persiapan Uji Coba Real-Time")
    st.info("Klik tombol ini SEKALI SAJA agar 4 algoritma tersimpan dan fitur prediksi di sidebar kiri bisa digunakan.")
    
    if st.button("🚀 Latih & Simpan 8 Model (Dengan & Tanpa SMOTE)", type="primary"):
        with st.spinner("Melatih 8 model sekaligus dan menyimpan 'otak' Machine Learning..."):
            try:
                conn = get_db_connection()
                df_train = pd.read_sql("SELECT clean_tweet, label FROM dataset_tweets WHERE label IS NOT NULL AND clean_tweet != ''", conn)
                conn.close()
                
                if df_train.empty:
                    st.error("Data bersih belum ada. Lakukan labeling terlebih dahulu!")
                else:
                    # 1. TF-IDF
                    vectorizer_side = TfidfVectorizer()
                    X_side = vectorizer_side.fit_transform(df_train['clean_tweet'])
                    y_side = df_train['label']
                    
                    # 2. Siapkan data dengan SMOTE
                    smote = SMOTE(random_state=42)
                    X_side_smote, y_side_smote = smote.fit_resample(X_side, y_side)
                    
                    # 3. Dictionary Model (Parameter sesuai Colab)
                    models_side = {
                        "svm": SVC(kernel='linear', random_state=42),
                        "naive_bayes": MultinomialNB(),
                        "decision_tree": DecisionTreeClassifier(random_state=42),
                        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42)
                    }
                    
                    # 4. Latih dan Simpan (TANPA SMOTE)
                    for file_name, model in models_side.items():
                        model.fit(X_side, y_side)
                        with open(f'model_{file_name}.pkl', 'wb') as f:
                            pickle.dump(model, f)
                            
                    # 5. Latih dan Simpan (DENGAN SMOTE)
                    for file_name, model in models_side.items():
                        model.fit(X_side_smote, y_side_smote)
                        with open(f'model_{file_name}_smote.pkl', 'wb') as f:
                            pickle.dump(model, f)
                            
                    # Simpan Vectorizer
                    with open('tfidf_vectorizer.pkl', 'wb') as f:
                        pickle.dump(vectorizer_side, f)
                        
                    st.success("✅ 8 Model (4 Tanpa SMOTE + 4 Dengan SMOTE) berhasil disimpan!")
            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")
                
    st.markdown("---")
    
    # --- EVALUASI EKSPERIMEN (SMOTE & K-FOLD) ---
    st.write("### 📊 Evaluasi Eksperimen (Tersimpan ke History)")
    conn = get_db_connection()
    
    # Tarik semua yang teks bersihnya tidak kosong
    df = pd.read_sql("SELECT clean_tweet, label FROM dataset_tweets WHERE clean_tweet IS NOT NULL AND clean_tweet != ''", conn)
    
    # Bikin 2 kolom agar tampilannya sejajar dan rapi
    col_model, col_kfold = st.columns(2)
    
    with col_model:
        model_choice = st.selectbox("Pilih Model:", ["Support Vector Machine (SVM)", "Naive Bayes", "Decision Tree", "Random Forest"])
        
    with col_kfold:
        # Default K-Fold diset ke 5 sesuai Colab
        k_fold = st.slider("Pilih jumlah K-Fold CV:", min_value=2, max_value=10, value=5)
        st.session_state['k_fold'] = k_fold
    
    st.write("#### Penanganan Imbalanced Data")
    use_smote = st.toggle("Gunakan SMOTE (Synthetic Minority Over-sampling Technique)", value=False)
    if use_smote:
        st.info("SMOTE Aktif: Data latih pada setiap lipatan (fold) akan diseimbangkan secara sintetis.")
    else:
        st.warning("SMOTE Nonaktif: Model akan dilatih dengan distribusi data asli.")
        
    st.markdown("---")
    
    if st.button("Mulai Klasifikasi", key="btn_klasifikasi"):
        if df.empty:
            st.error("❌ Data bersih (clean_tweet) masih kosong. Pastikan Preprocessing sudah dijalankan!")
        elif df['label'].isnull().all() or (df['label'].astype(str).str.strip() == '').all() or (df['label'].astype(str) == 'None').all():
            st.error("❌ Data sudah bersih, TAPI Label Sentimen masih kosong! Sistem tidak bisa belajar tanpa label. Silakan ke menu **Labeling** terlebih dahulu.")
        else:
            # Ambil hanya baris yang benar-benar punya label
            df_valid = df.dropna(subset=['label']).copy()
            df_valid['label'] = df_valid['label'].astype(str).str.strip().str.lower()
            
            status_text = "dengan SMOTE" if use_smote else "Tanpa SMOTE"
            with st.spinner(f"Melatih {model_choice} dengan {k_fold}-Fold CV ({status_text})..."):
                
                # Transformasi teks
                X_tfidf, vectorizer = get_tfidf_vectorizer(df_valid['clean_tweet'])
                y = df_valid['label']
                
                # Menangkap 8 output metrik
                acc, prec, rec, f1_weighted, f1_macro, bal_acc, mcc, cm = evaluate_model(model_choice, X_tfidf, y, k_fold, use_smote)
                
                # Update skema database (abaikan error jika kolom sudah ada)
                cursor = conn.cursor()
                try:
                    cursor.execute("ALTER TABLE model_evaluation ADD COLUMN f1_macro FLOAT, ADD COLUMN balanced_acc FLOAT, ADD COLUMN mcc FLOAT")
                    conn.commit()
                except:
                    pass 
                
                # Simpan metrik ke DB
                try:
                    cursor.execute(
                        "INSERT INTO model_evaluation (model_name, k_fold, accuracy, precision_score, recall_score, f1_score, use_smote, f1_macro, balanced_acc, mcc) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                        (model_choice, k_fold, float(acc), float(prec), float(rec), float(f1_weighted), use_smote, float(f1_macro), float(bal_acc), float(mcc))
                    )
                except Exception as db_err:
                    st.error(f"Gagal menyimpan ke database: {db_err}")
                    
                conn.commit()
                cursor.close()
                
                # SIMPAN KE SESSION STATE
                st.session_state['eval_results'] = {
                    'model': model_choice,
                    'smote': use_smote,
                    'acc': acc, 'f1_weight': f1_weighted, 
                    'f1_macro': f1_macro, 'bal_acc': bal_acc, 'mcc': mcc,
                    'cm': cm, 'labels': ["negatif", "netral", "positif"] # Label disesuaikan dengan standar visualisasi Colab
                }
                
                st.rerun()

    # --- TAMPILKAN HASIL DARI SESSION STATE ---
    if 'eval_results' in st.session_state:
        res = st.session_state['eval_results']
        st.success(f"Evaluasi {res['model']} ({'Dengan SMOTE' if res['smote'] else 'Tanpa SMOTE'}) Selesai!")
        
        st.subheader("Hasil Metrik Evaluasi")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        col1.metric("Accuracy", f"{res['acc']:.2f}%")
        col2.metric("Weighted F1", f"{res['f1_weight']:.2f}%")
        col3.metric("Macro F1", f"{res['f1_macro']:.3f}")
        col4.metric("Balanced Acc", f"{res['bal_acc']:.3f}")
        col5.metric("MCC", f"{res['mcc']:.3f}")
        
        st.subheader("Confusion Matrix")
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(res['cm'], annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=res['labels'], yticklabels=res['labels'])
        plt.ylabel('Aktual (Data Asli)')
        plt.xlabel('Prediksi Model')
        st.pyplot(fig)
        
    conn.close()

# ----------------- 8. HISTORY & COMPARISON -----------------
elif menu == "History & Comparison":
    st.title("Riwayat & Perbandingan Model")
    
    conn = get_db_connection()
    
    try:
        df_eval = pd.read_sql("SELECT * FROM model_evaluation ORDER BY id DESC", conn)
    except Exception as e:
        df_eval = pd.DataFrame() 
        
    if df_eval.empty:
        st.info("Belum ada riwayat model yang dilatih. Silakan lakukan eksperimen di menu 'Modeling & Evaluation'.")
    else:
        # --- 1. RAPIKAN & URUTKAN TABEL ---
        df_display = df_eval.copy()
        
        # Konversi boolean SMOTE
        if 'use_smote' in df_display.columns:
            df_display['use_smote'] = df_display['use_smote'].apply(lambda x: "Ya" if x else "Tidak")
        else:
            df_display['use_smote'] = "Tidak"

        # Rename kolom agar mudah dibaca
        rename_dict = {
            'id': 'ID', 
            'model_name': 'Algoritma', 
            'k_fold': 'K-Fold', 
            'use_smote': 'Pakai SMOTE', 
            'accuracy': 'Akurasi', 
            'precision_score': 'Presisi', 
            'recall_score': 'Recall', 
            'f1_score': 'F1-Score',
            'f1_macro': 'Macro F1',
            'balanced_acc': 'Balanced Acc',
            'mcc': 'MCC'
        }
        df_display = df_display.rename(columns=rename_dict)
        
        # Susun urutan kolom yang logis dan Rapi
        kolom_urutan = [c for c in [
            'ID', 'Algoritma', 'K-Fold', 'Pakai SMOTE', 
            'Akurasi', 'Presisi', 'Recall', 'F1-Score', 
            'Macro F1', 'Balanced Acc', 'MCC'
        ] if c in df_display.columns]
        
        df_display = df_display[kolom_urutan]

        st.write("### 📜 Tabel Riwayat Eksperimen")
        st.dataframe(df_display, use_container_width=True, hide_index=True)
        st.markdown("---")
        
        # --- 2. GRAFIK PERBANDINGAN DINAMIS ---
        st.write("### 📊 Grafik Perbandingan Model")
        
        # Pilihan metrik untuk grafik
        metric_map = {
            "Akurasi": "accuracy",
            "Presisi": "precision_score",
            "Recall": "recall_score",
            "F1-Score (Weighted)": "f1_score",
            "Macro F1": "f1_macro",
            "Balanced Accuracy": "balanced_acc",
            "MCC (Matthews Corr Coef)": "mcc"
        }
        
        col_metric, col_space = st.columns([2, 2])
        with col_metric:
            selected_metric_label = st.selectbox("Pilih Metrik Evaluasi untuk Grafik:", list(metric_map.keys()))
            selected_metric_col = metric_map[selected_metric_label]

        # Buat label sumbu X yang informatif
        if 'use_smote' in df_eval.columns:
            df_eval['Model_Label'] = df_eval.apply(
                lambda x: f"{x['model_name']} (SMOTE) [ID:{x['id']}]" if x['use_smote'] else f"{x['model_name']} [ID:{x['id']}]", 
                axis=1
            )
        else:
            df_eval['Model_Label'] = df_eval.apply(lambda x: f"{x['model_name']} [ID:{x['id']}]", axis=1)

        # Plotting Grafik
        if selected_metric_col in df_eval.columns and not df_eval[selected_metric_col].isnull().all():
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(data=df_eval, x='Model_Label', y=selected_metric_col, palette='Reds_r', ax=ax)
            
            plt.xticks(rotation=45, ha='right')
            plt.ylabel(selected_metric_label)
            plt.xlabel("Eksperimen Model")
            
            # Pengaturan batas nilai Y berdasarkan jenis metrik
            if selected_metric_col == 'mcc':
                plt.ylim(-1.0, 1.1)  # MCC bernilai rentang -1 s/d 1
            elif df_eval[selected_metric_col].max() > 1.0:
                plt.ylim(0, 105)     # Jika metrik menggunakan persentase (0-100)
            else:
                plt.ylim(0, 1.1)     # Jika metrik menggunakan rasio desimal (0-1)
            
            # Berikan label nilai di atas bar
            import numpy as np
            for p in ax.patches:
                val = p.get_height()
                if not np.isnan(val):
                    fmt = '.2f' if val > 1.0 else '.3f'
                    ax.annotate(format(val, fmt), 
                                (p.get_x() + p.get_width() / 2., val), 
                                ha='center', va='center', 
                                xytext=(0, 9), 
                                textcoords='offset points')
            
            st.pyplot(fig)
        else:
            st.warning(f"Data untuk metrik '{selected_metric_label}' belum tersedia pada riwayat lama.")

        st.markdown("---")
        
        # --- 3. TOMBOL HAPUS RIWAYAT ---
        st.write("### 🗑️ Bersihkan Riwayat")
        st.warning("Hati-hati, aksi ini akan mengosongkan seluruh tabel riwayat eksperimen.")
        if st.button("Hapus Semua Riwayat", type="primary"):
            cursor = conn.cursor()
            cursor.execute("TRUNCATE TABLE model_evaluation")
            conn.commit()
            cursor.close()
            st.success("Semua riwayat berhasil dihapus!")
            st.rerun()

    conn.close()

    # ----------------- 9. WORD CLOUD -----------------
elif menu == "Word Cloud":
    st.title("☁️ Word Cloud Sentimen")
    st.write("Visualisasi kata-kata yang paling sering muncul dalam dataset Indihome.")
    
    conn = get_db_connection()
    
    try:
        df = pd.read_sql("SELECT * FROM dataset_tweets", conn)
    except:
        df = pd.DataFrame()
        
    if df.empty or 'clean_tweet' not in df.columns or df['clean_tweet'].isnull().all():
        st.warning("Data bersih (clean_tweet) belum tersedia. Pastikan kamu sudah melakukan tahap Preprocessing!")
    else:
        # Layout filter
        col1, col2 = st.columns(2)
        
        with col1:
            # Pilihan label sentimen
            if 'label' in df.columns and not df['label'].isnull().all():
                labels = df['label'].dropna().unique().tolist()
                pilihan_label = st.selectbox("Pilih Filter Sentimen:", ["Semua Sentimen"] + labels)
            else:
                st.info("Dataset belum dilabeli.")
                pilihan_label = "Semua Sentimen"
                
        with col2:
            # Pilihan kolom (biasanya WordCloud menggunakan data yang sudah di-cleansing)
            kolom_teks = st.selectbox("Gunakan Teks Dari Tahap:", ["clean_tweet", "original_tweet"])
            
        st.markdown("---")
        
        if st.button("Generate Word Cloud", type="primary", key="btn_wordcloud"):
            with st.spinner("Merangkai kata-kata..."):
                # Filter dataframe berdasarkan pilihan sentimen
                if pilihan_label != "Semua Sentimen":
                    df_filtered = df[df['label'] == pilihan_label]
                else:
                    df_filtered = df
                
                # Ambil teks dan gabungkan menjadi satu string panjang
                text_data = df_filtered[kolom_teks].dropna().astype(str).tolist()
                all_words = " ".join(text_data)
                
                if len(all_words.strip()) == 0:
                    st.error(f"Tidak ada teks yang ditemukan untuk kategori {pilihan_label}.")
                else:
                    # Penentuan warna Word Cloud agar estetik
                    # Negatif -> Merah (Reds), Positif -> Hijau (Greens), Netral/Semua -> Biru (Blues)
                    if pilihan_label.lower() == 'negatif':
                        warna_tema = 'Reds'
                    elif pilihan_label.lower() == 'positif':
                        warna_tema = 'Greens'
                    else:
                        warna_tema = 'Blues'
                    
                    # Generate Word Cloud
                    wordcloud = WordCloud(
                        width=800, 
                        height=400, 
                        background_color='white', 
                        colormap=warna_tema,
                        max_words=100, # Menampilkan 100 kata teratas
                        contour_width=3, 
                        contour_color='steelblue'
                    ).generate(all_words)
                    
                    # Tampilkan menggunakan Matplotlib
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off') # Hilangkan garis sumbu x dan y
                    
                    st.success(f"Berhasil menampilkan Word Cloud untuk: **{pilihan_label}**")
                    st.pyplot(fig)
                    
    conn.close()

    # ----------------- 9. PREDIKSI MODEL (TESTING) -----------------
elif menu == "Prediksi Model":
    st.title("🧪 Uji Prediksi Model Real-Time")
    st.write("Ketik kalimat bebas untuk melihat perbandingan tebakan dari 8 model (Dengan dan Tanpa SMOTE).")
    st.markdown("---")

    test_text = st.text_area("Masukkan teks ulasan/keluhan:", placeholder="contoh: indihome lemot banget parah, teknisinya lama rto terus!", height=150)

    if st.button("🚀 Prediksi Sentimen", type="primary"):
        if test_text.strip() == "":
            st.warning("Teks tidak boleh kosong!")
        else:
            try:
                # 1. Load TF-IDF Vectorizer
                with open('tfidf_vectorizer.pkl', 'rb') as f:
                    vectorizer = pickle.load(f)
                    
                # 2. Ubah teks input menjadi angka
                text_tfidf = vectorizer.transform([test_text.lower()])
                
                model_names = {
                    "svm": "SVM", 
                    "naive_bayes": "Naive Bayes", 
                    "decision_tree": "Decision Tree", 
                    "random_forest": "Random Forest"
                }
                
                # Dictionary untuk menampung hasil prediksi sementara sebelum masuk database
                hasil_rekaman = {}
                
                # --- BARIS 1: TANPA SMOTE ---
                st.markdown("### 📊 Hasil Prediksi (Data Asli / Tanpa SMOTE):")
                c1, c2, c3, c4 = st.columns(4)
                cols_no_smote = [c1, c2, c3, c4]
                
                for i, (file_name, display_name) in enumerate(model_names.items()):
                    with cols_no_smote[i]:
                        st.markdown(f"**{display_name}**")
                        try:
                            with open(f'model_{file_name}.pkl', 'rb') as f:
                                model = pickle.load(f)
                                pred = model.predict(text_tfidf)[0]
                                
                                pred_bersih = str(pred).strip().lower()
                                hasil_rekaman[file_name] = pred_bersih # Simpan untuk history
                                
                                st.caption(f"Label Asli Model: '{pred}'")
                                
                                if pred_bersih == 'positif':
                                    st.success("Positif 😃")
                                elif pred_bersih == 'negatif':
                                    st.error("Negatif 😡")
                                else:
                                    st.warning("Netral 😐")
                        except FileNotFoundError:
                            st.error("❌ Belum dilatih")
                            hasil_rekaman[file_name] = "Error/Belum ada"
                
                st.markdown("---")
                
                # --- BARIS 2: DENGAN SMOTE ---
                st.markdown("### 📊 Hasil Prediksi (Data Seimbang / Dengan SMOTE):")
                cs1, cs2, cs3, cs4 = st.columns(4)
                cols_smote = [cs1, cs2, cs3, cs4]
                
                for i, (file_name, display_name) in enumerate(model_names.items()):
                    with cols_smote[i]:
                        st.markdown(f"**{display_name} (SMOTE)**")
                        try:
                            with open(f'model_{file_name}_smote.pkl', 'rb') as f:
                                model = pickle.load(f)
                                pred = model.predict(text_tfidf)[0]
                                
                                pred_bersih = str(pred).strip().lower()
                                hasil_rekaman[file_name + "_smote"] = pred_bersih # Simpan untuk history
                                
                                st.caption(f"Label Asli Model: '{pred}'")
                                
                                if pred_bersih == 'positif':
                                    st.success("Positif 😃")
                                elif pred_bersih == 'negatif':
                                    st.error("Negatif 😡")
                                else:
                                    st.warning("Netral 😐")
                        except FileNotFoundError:
                            st.error("❌ Belum dilatih")
                            hasil_rekaman[file_name + "_smote"] = "Error/Belum ada"
                            
                # --- SIMPAN KE DATABASE ---
                conn = get_db_connection()
                cursor = conn.cursor()
                
                # Bikin tabel otomatis kalau belum pernah dibuat sebelumnya
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS history_prediksi (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        waktu TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        teks TEXT,
                        svm VARCHAR(20), naive_bayes VARCHAR(20), decision_tree VARCHAR(20), random_forest VARCHAR(20),
                        svm_smote VARCHAR(20), naive_bayes_smote VARCHAR(20), decision_tree_smote VARCHAR(20), random_forest_smote VARCHAR(20)
                    )
                """)
                
                # Masukkan ke tabel
                cursor.execute("""
                    INSERT INTO history_prediksi 
                    (teks, svm, naive_bayes, decision_tree, random_forest, svm_smote, naive_bayes_smote, decision_tree_smote, random_forest_smote) 
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    test_text, 
                    hasil_rekaman.get('svm'), hasil_rekaman.get('naive_bayes'), hasil_rekaman.get('decision_tree'), hasil_rekaman.get('random_forest'),
                    hasil_rekaman.get('svm_smote'), hasil_rekaman.get('naive_bayes_smote'), hasil_rekaman.get('decision_tree_smote'), hasil_rekaman.get('random_forest_smote')
                ))
                
                conn.commit()
                cursor.close()
                conn.close()
                
                st.toast("✅ Prediksi berhasil disimpan ke riwayat!")
                            
            except FileNotFoundError:
                st.error("⚠️ Model belum ada! Silakan ke menu 'Modeling & Evaluation' dan klik tombol Latih terlebih dahulu.")

    # ==========================================
    # TAMPILKAN TABEL HISTORY DI BAWAH
    # ==========================================
    st.markdown("---")
    st.write("### 🕒 Riwayat Prediksi Pengguna")
    
    conn = get_db_connection()
    try:
        # Tarik data dari yang terbaru
        df_history = pd.read_sql("SELECT waktu, teks, svm, naive_bayes, decision_tree, random_forest, svm_smote, naive_bayes_smote, decision_tree_smote, random_forest_smote FROM history_prediksi ORDER BY waktu DESC", conn)
        
        if not df_history.empty:
            # Rapikan nama kolom agar cantik saat ditampilkan di tabel Streamlit
            df_history.rename(columns={
                'waktu': 'Waktu', 'teks': 'Teks Input',
                'svm': 'SVM', 'naive_bayes': 'NB', 'decision_tree': 'DT', 'random_forest': 'RF',
                'svm_smote': 'SVM (SMOTE)', 'naive_bayes_smote': 'NB (SMOTE)', 'decision_tree_smote': 'DT (SMOTE)', 'random_forest_smote': 'RF (SMOTE)'
            }, inplace=True)
            
            st.dataframe(df_history, use_container_width=True)
            
            # Tombol hapus riwayat
            if st.button("🗑️ Hapus Semua Riwayat", type="secondary"):
                cursor = conn.cursor()
                cursor.execute("TRUNCATE TABLE history_prediksi")
                conn.commit()
                cursor.close()
                st.rerun()
        else:
            st.info("Belum ada riwayat prediksi yang dilakukan. Silakan coba prediksi di atas!")
            
    except Exception as e:
        st.info("Tabel riwayat belum tersedia. Lakukan prediksi pertama untuk membuatnya otomatis.")
        
    finally:
        conn.close()