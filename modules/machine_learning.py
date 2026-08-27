from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
# --- TAMBAHKAN IMPORT SMOTE ---
from imblearn.over_sampling import SMOTE

def get_tfidf_vectorizer(corpus):
    vectorizer = TfidfVectorizer()
    X_tfidf = vectorizer.fit_transform(corpus)
    return X_tfidf, vectorizer

def get_model(model_name):
    if model_name == "Support Vector Machine (SVM)":
        return SVC(kernel='linear')
    elif model_name == "Naive Bayes":
        return MultinomialNB()
    elif model_name == "Decision Tree":
        return DecisionTreeClassifier(random_state=42)
    elif model_name == "Random Forest":
        return RandomForestClassifier(random_state=42)
    return None

# --- UPDATE FUNGSI EVALUATE_MODEL DENGAN PARAMETER USE_SMOTE ---
def evaluate_model(model_name, X_tfidf, y, k_folds, use_smote=False):
    model = get_model(model_name)
    
    cv = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    # Menyiapkan variabel untuk menampung prediksi
    import numpy as np
    y_pred_all = np.empty_like(y)
    
    for train_index, test_index in cv.split(X_tfidf, y):
        X_train, X_test = X_tfidf[train_index], X_tfidf[test_index]
        # Pastikan menggunakan .iloc jika y adalah Pandas Series agar indeksnya benar
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # --- LOGIKA SMOTE DITERAPKAN HANYA PADA DATA TRAINING ---
        if use_smote:
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            
        # Latih model
        model.fit(X_train, y_train)
        
        # Prediksi data testing
        y_pred = model.predict(X_test)
        
        # Simpan hasil prediksi
        y_pred_all[test_index] = y_pred
        
    # Hitung metrik evaluasi dari keseluruhan prediksi cross-validation
    acc = accuracy_score(y, y_pred_all)
    prec = precision_score(y, y_pred_all, average='weighted', zero_division=0)
    rec = recall_score(y, y_pred_all, average='weighted', zero_division=0)
    f1 = f1_score(y, y_pred_all, average='weighted', zero_division=0)
    cm = confusion_matrix(y, y_pred_all)
    
    return acc, prec, rec, f1, cm