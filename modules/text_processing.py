import re
import emoji
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# --- INISIALISASI SASTRAWI (Diletakkan di luar fungsi agar lebih cepat dipanggil) ---
factory_stemmer = StemmerFactory()
stemmer = factory_stemmer.create_stemmer()

factory_stopword = StopWordRemoverFactory()
stopword_remover = factory_stopword.create_stop_word_remover()

# --- KAMUS NORMALISASI SEDERHANA ---
# Kamu bisa menambahkan kata-kata gaul/typo Indihome lainnya di sini nanti
kamus_slang = {
    "yg": "yang", "dgn": "dengan", "tdk": "tidak", "gak": "tidak", 
    "bgt": "banget", "kalo": "kalau", "sampe": "sampai", "indihom": "indihome",
    "lemot": "lambat", "ngelag": "lambat", "ilang": "hilang", "tp": "tapi"
}

# --- FUNGSI TAMBAHAN: EMOJI TO TEXT ---
def convert_emoji_to_text(text):
    # Tambahkan parameter language='id' agar diterjemahkan ke Bahasa Indonesia
    # Contoh: 😂 -> :wajah_gembira_dengan_air_mata:
    text = emoji.demojize(text, language='id')
    
    # Menghapus tanda titik dua (:) dan mengubah underscore (_) menjadi spasi
    text = text.replace(":", " ").replace("_", " ")
    
    # Menghapus spasi berlebih
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- 1. CLEANSING ---
def cleansing(text):
    text = re.sub(r'http\S+', '', text) # Hapus URL/Link
    text = re.sub(r'@[A-Za-z0-9_]+', '', text) # Hapus Mention
    text = re.sub(r'#[A-Za-z0-9_]+', '', text) # Hapus Hashtag
    text = re.sub(r'[^a-zA-Z\s]', ' ', text) # Hapus Angka dan Tanda Baca (Sisakan huruf dan spasi)
    return text.strip()

# --- 2. CASE FOLDING ---
def case_folding(text):
    return text.lower()

# --- 3. TOKENIZATION ---
def tokenization(text):
    # Memecah kalimat menjadi list kata
    return text.split()

# --- 4. NORMALIZATION ---
def normalization(tokens):
    # Mengganti kata tidak baku menjadi baku berdasarkan kamus_slang
    return [kamus_slang.get(t, t) for t in tokens]

# --- 5. STOPWORD REMOVAL ---
def stopword_removal(text):
    return stopword_remover.remove(text)

# --- 6. STEMMING ---
def stemming(text):
    return stemmer.stem(text)

# --- FUNGSI UTAMA YANG MENGGABUNGKAN KE-7 TAHAPAN (DENGAN CEKLIS) ---
def preprocess_text(text, use_emoji=False, use_cleansing=True, use_casefolding=True, use_tokenization=True, use_normalization=True, use_stopword=True, use_stemming=True):
    if not isinstance(text, str):
        return ""
    
    # Tahap Tambahan: Konversi Emoji
    # Dilakukan PALING PERTAMA agar teks terjemahan emoji tidak ikut terhapus oleh fungsi Cleansing
    if use_emoji:
        text = convert_emoji_to_text(text)
        
    # Tahap 1: Cleansing
    if use_cleansing:
        text = cleansing(text)
        
    # Tahap 2: Case Folding
    if use_casefolding:
        text = case_folding(text)
    
    # Tahap 3 & 4: Tokenization & Normalization
    # (Normalization butuh tokenization terlebih dahulu agar bisa memproses per kata)
    if use_tokenization or use_normalization:
        tokens = tokenization(text)
        
        if use_normalization:
            tokens = normalization(tokens)
            
        # Gabungkan kembali jadi kalimat (string) untuk proses selanjutnya
        text = " ".join(tokens) 
    
    # Tahap 5: Stopword Removal
    if use_stopword:
        text = stopword_removal(text)
        
    # Tahap 6: Stemming
    if use_stemming:
        text = stemming(text)
    
    return text