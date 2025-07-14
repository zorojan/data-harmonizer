import streamlit as st
import pandas as pd
from rapidfuzz import fuzz
import nltk
import spacy
from sentence_transformers import SentenceTransformer, util
import os

st.title("Обработка параметров финального результата")

# Всегда загружаем только из grouped_categories.csv
csv_path = os.path.join(os.path.dirname(__file__), '..', 'grouped_categories.csv')
try:
    df_param = pd.read_csv(csv_path, encoding='utf-8-sig')
    st.info(f"Загружены данные из grouped_categories.csv")
except Exception as e:
    df_param = pd.DataFrame()
    st.warning(f"Не удалось загрузить grouped_categories.csv: {e}")

# --- Загрузка моделей spaCy и sentence-transformers ---
@st.cache_resource(show_spinner=False)
def get_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
        return spacy.load("en_core_web_sm")

@st.cache_resource(show_spinner=False)
def get_st_model():
    return SentenceTransformer('distiluse-base-multilingual-cased', device='cpu')

if not df_param.empty:
    st.dataframe(df_param.head(20))
    all_columns = list(df_param.columns)

    # --- Универсальное определение колонок для исключения ---
    default_excluded_cols = [
        col for col in all_columns
        if 'name' in col.lower() or 'id' in col.lower() or 'source_file' in col.lower()
    ]

    st.markdown("#### Исключить колонки из автоматического сравнения")
    excluded_cols = st.multiselect(
        "Выберите колонки, которые не нужно включать в автоматический поиск пар:",
        options=all_columns,
        default=default_excluded_cols
    )

    columns = [c for c in all_columns if c not in excluded_cols]

    st.markdown("#### 1. Выберите метод интеллектуального сравнения колонок")
    method = st.selectbox(
        "Метод сравнения:",
        ["Best Practice Combined", "NLTK Jaccard", "spaCy Similarity", "Sentence-Transformers", "RapidFuzz"],
        index=4
    )
    
    # --- Individual Threshold Sliders ---
    if method == "Best Practice Combined":
        threshold_col = st.slider("Порог схожести для Best Practice (%)", 60, 100, 85, 1, key="bp_thresh")
    elif method == "NLTK Jaccard":
        threshold_col = st.slider("Порог схожести для NLTK Jaccard (%)", 60, 100, 85, 1, key="nltk_thresh")
    elif method == "spaCy Similarity":
        threshold_col = st.slider("Порог схожести для spaCy Similarity (%)", 60, 100, 85, 1, key="spacy_thresh")
    elif method == "Sentence-Transformers":
        threshold_col = st.slider("Порог схожести для Sentence-Transformers (%)", 60, 100, 85, 1, key="st_thresh")
    elif method == "RapidFuzz":
        threshold_col = st.slider("Порог схожести для RapidFuzz (%)", 60, 100, 85, 1, key="rf_thresh")

    group_diff_sources = st.checkbox(
        "Группировать только если параметры из разных источников (source_file)",
        value=True,
        help="Если включено, автоматическое объединение будет только для пар, где значения из разных source_file."
    )
    similar_pairs = []
    nlp = get_spacy_model()
    def lemmatize(text):
        doc = nlp(text.lower())
        return " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])
    from nltk.corpus import wordnet as wn
    nltk.download('wordnet', quiet=True)
    def are_synonyms(word1, word2):
        syns1 = set([l.name() for s in wn.synsets(word1) for l in s.lemmas()])
        syns2 = set([l.name() for s in wn.synsets(word2) for l in s.lemmas()])
        return len(syns1 & syns2) > 0
    import re
    def extract_unit(text):
        match = re.findall(r"\\b([a-zA-Z]{1,4})\\b", text)
        return set(match)
    # --- Автоматическое определение похожих пар ---
    if method == "Best Practice Combined":
        st_model = get_st_model()
        for i in range(len(columns)):
            for j in range(i+1, len(columns)):
                col1, col2 = columns[i], columns[j]
                # Если включен фильтр по источникам, проверяем что есть хотя бы одна строка с разными source_file
                if group_diff_sources:
                    if 'source_file' in df_param.columns:
                        sources1 = set(df_param[df_param[col1].notna()]['source_file'])
                        sources2 = set(df_param[df_param[col2].notna()]['source_file'])
                        if not (sources1 and sources2 and len(sources1 & sources2) < max(len(sources1), len(sources2))):
                            continue
                lem1, lem2 = lemmatize(col1), lemmatize(col2)
                emb1 = st_model.encode([lem1], convert_to_tensor=True)
                emb2 = st_model.encode([lem2], convert_to_tensor=True)
                st_score = util.cos_sim(emb1, emb2)[0][0].item()
                if st_score > 0.7:
                    similar_pairs.append({
                        "Column 1": col1,
                        "Column 2": col2,
                        "Score": int(st_score*100),
                        "Method": "SentenceTr"
                    })
                    continue
                rf_score = fuzz.token_sort_ratio(lem1, lem2)
                if rf_score > 80:
                    similar_pairs.append({
                        "Column 1": col1,
                        "Column 2": col2,
                        "Score": rf_score,
                        "Method": "RapidFuzz"
                    })
                    continue
                if len(lem1.split()) == 1 and len(lem2.split()) == 1 and are_synonyms(lem1, lem2):
                    similar_pairs.append({
                        "Column 1": col1,
                        "Column 2": col2,
                        "Score": 100,
                        "Method": "WordNet Synonym"
                    })
                    continue
                units1, units2 = extract_unit(col1), extract_unit(col2)
                if units1 and units2 and units1 & units2:
                    similar_pairs.append({
                        "Column 1": col1,
                        "Column 2": col2,
                        "Score": 90,
                        "Method": "Unit Match"
                    })
        sim_col_df = pd.DataFrame([p for p in similar_pairs if p["Score"] >= threshold_col]) if similar_pairs else pd.DataFrame()
    elif method == "NLTK Jaccard":
        from nltk.corpus import stopwords
        nltk.download('stopwords', quiet=True)
        stop_words = set(stopwords.words('english'))
        def preprocess(text):
            return ' '.join([w for w in text.lower().split() if w not in stop_words])
        for i in range(len(columns)):
            for j in range(i+1, len(columns)):
                col1, col2 = columns[i], columns[j]
                set1 = set(preprocess(col1).split())
                set2 = set(preprocess(col2).split())
                if set1 or set2:
                    jaccard = int(100 * len(set1 & set2) / max(1, len(set1 | set2)))
                else:
                    jaccard = 0
                if jaccard >= threshold_col:
                    similar_pairs.append({
                        "Column 1": col1,
                        "Column 2": col2,
                        "NLTK Jaccard": jaccard
                    })
        sim_col_df = pd.DataFrame(similar_pairs)
    elif method == "spaCy Similarity":
        nlp = get_spacy_model()
        for i in range(len(columns)):
            for j in range(i+1, len(columns)):
                col1, col2 = columns[i], columns[j]
                doc1, doc2 = nlp(col1), nlp(col2)
                spacy_score = int(doc1.similarity(doc2) * 100)
                if spacy_score >= threshold_col:
                    similar_pairs.append({
                        "Column 1": col1,
                        "Column 2": col2,
                        "spaCy": spacy_score
                    })
        sim_col_df = pd.DataFrame(similar_pairs)
    elif method == "Sentence-Transformers":
        st_model = get_st_model()
        for i in range(len(columns)):
            for j in range(i+1, len(columns)):
                col1, col2 = columns[i], columns[j]
                emb1 = st_model.encode([col1], convert_to_tensor=True)
                emb2 = st_model.encode([col2], convert_to_tensor=True)
                st_score = int(util.cos_sim(emb1, emb2)[0][0].item() * 100)
                if st_score >= threshold_col:
                    similar_pairs.append({
                        "Column 1": col1,
                        "Column 2": col2,
                        "SentenceTr": st_score
                    })
        sim_col_df = pd.DataFrame(similar_pairs)
    elif method == "RapidFuzz":
        for i in range(len(columns)):
            for j in range(i+1, len(columns)):
                col1, col2 = columns[i], columns[j]
                rf_score = fuzz.token_sort_ratio(col1, col2)
                if rf_score >= threshold_col:
                    similar_pairs.append({
                        "Column 1": col1,
                        "Column 2": col2,
                        "RapidFuzz": rf_score
                    })
        sim_col_df = pd.DataFrame(similar_pairs)
    st.markdown("#### 2. Автоматически найденные пары для объединения")
    if not sim_col_df.empty:
        st.dataframe(sim_col_df.sort_values(sim_col_df.columns[-1], ascending=False).reset_index(drop=True))
        auto_merge_cols = st.button("Зафиксировать все найденные пары (авто-объединение)")
        if auto_merge_cols:
            for _, row in sim_col_df.iterrows():
                col_a, col_b = row["Column 1"], row["Column 2"]
                new_col_name = f"{col_a}_{col_b}_merged"
                if new_col_name not in df_param.columns:
                    df_param[new_col_name] = df_param[col_a].combine_first(df_param[col_b])
            st.success("Автоматическое объединение выполнено. Проверьте новые колонки.")
            st.dataframe(df_param.head(10))
    else:
        st.info("Похожих колонок не найдено по текущему порогу.")

    st.markdown("#### 3. Ручное подтверждение и объединение колонок")
    col1, col2 = st.columns(2)
    with col1:
        col_a = st.selectbox("Колонка 1", all_columns, key="merge_col_a_param")
    with col2:
        col_b = st.selectbox("Колонка 2", [c for c in all_columns if c != col_a], key="merge_col_b_param")
    new_col_name = st.text_input("Новое имя для объединённой колонки", value=f"{col_a}_{col_b}_merged", key="merge_col_name_param")
    if st.button("Объединить выбранные колонки и зафиксировать", key="merge_btn_param"):
        if col_a and col_b and new_col_name:
            df_param[new_col_name] = df_param[col_a].combine_first(df_param[col_b])
            st.success(f"Колонки '{col_a}' и '{col_b}' объединены в '{new_col_name}' и зафиксированы.")
            st.dataframe(df_param[[col_a, col_b, new_col_name]].head(10))
        else:
            st.warning("Выберите обе колонки и введите новое имя.")
else:
    st.warning("Нет данных для обработки. Передайте DataFrame через сессию или сохраните grouped_categories.csv.")
