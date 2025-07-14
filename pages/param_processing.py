import streamlit as st
import pandas as pd
from rapidfuzz import fuzz
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

# --- Загрузка модели sentence-transformers ---
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

    st.markdown("#### 1. Выберите методы интеллектуального сравнения колонок")
    
    # --- Checkboxes for method selection ---
    col1, col2 = st.columns(2)
    with col1:
        use_rapidfuzz = st.checkbox("RapidFuzz", value=True, key="use_rapidfuzz")
    with col2:
        use_sentence_transformers = st.checkbox("Sentence-Transformers", value=False, key="use_st")
    
    # --- Individual Threshold Sliders ---
    rf_threshold = None
    st_threshold = None
    
    if use_rapidfuzz:
        rf_threshold = st.slider("Порог схожести для RapidFuzz (%)", 60, 100, 85, 1, key="rf_thresh")
    
    if use_sentence_transformers:
        st_threshold = st.slider("Порог схожести для Sentence-Transformers (%)", 60, 100, 85, 1, key="st_thresh")

    group_diff_sources = st.checkbox(
        "Группировать только если параметры из разных источников (source_file)",
        value=True,
        help="Если включено, автоматическое объединение будет только для пар, где значения из разных source_file."
    )
    
    # --- Автоматическое определение похожих пар ---
    similar_pairs = []
    
    # RapidFuzz method
    if use_rapidfuzz and rf_threshold is not None:
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
                rf_score = fuzz.token_sort_ratio(col1, col2)
                if rf_score >= rf_threshold:
                    similar_pairs.append({
                        "Column 1": col1,
                        "Column 2": col2,
                        "Score": rf_score,
                        "Method": "RapidFuzz"
                    })
    
    # Sentence-Transformers method
    if use_sentence_transformers and st_threshold is not None:
        if len(columns) > 30:
            st.warning(f"Внимание: выбрано {len(columns)} колонок. Сравнение с Sentence-Transformers может занять значительное время!")
        st_model = get_st_model()
        with st.spinner("Вычисление схожести с помощью Sentence-Transformers, пожалуйста, подождите..."):
            # Precompute all embeddings ONCE
            embeddings = st_model.encode(columns, convert_to_tensor=True)
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
                    st_score = int(util.cos_sim(embeddings[i], embeddings[j]).item() * 100)
                    if st_score >= st_threshold:
                        similar_pairs.append({
                            "Column 1": col1,
                            "Column 2": col2,
                            "Score": st_score,
                            "Method": "SentenceTransformers"
                        })
    
    # Remove duplicates and create DataFrame
    seen_pairs = set()
    unique_pairs = []
    for pair in similar_pairs:
        pair_key = tuple(sorted([pair["Column 1"], pair["Column 2"]]))
        if pair_key not in seen_pairs:
            seen_pairs.add(pair_key)
            unique_pairs.append(pair)
    
    sim_col_df = pd.DataFrame(unique_pairs) if unique_pairs else pd.DataFrame()
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
