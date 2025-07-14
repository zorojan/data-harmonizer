import streamlit as st
import pandas as pd
from rapidfuzz import fuzz

st.title("Обработка параметров финального результата")

# Получаем финальный DataFrame из сессии (или создаём пустой)
df_param = st.session_state.get('final_df_for_param_page', pd.DataFrame())

if not df_param.empty:
    st.dataframe(df_param.head(20))
    columns = list(df_param.columns)
    st.markdown("#### 1. Автоматический поиск похожих колонок (RapidFuzz)")
    similar_pairs = []
    threshold_col = st.slider("Порог схожести для колонок (%)", 60, 100, 85, 1, key="col_sim_threshold_param")
    for i in range(len(columns)):
        for j in range(i+1, len(columns)):
            score = fuzz.token_sort_ratio(columns[i], columns[j])
            if score >= threshold_col:
                similar_pairs.append({
                    "Column 1": columns[i],
                    "Column 2": columns[j],
                    "Score": score
                })
    sim_col_df = pd.DataFrame(similar_pairs)
    if not sim_col_df.empty:
        st.dataframe(sim_col_df.sort_values("Score", ascending=False).reset_index(drop=True))
    else:
        st.info("Похожих колонок не найдено по текущему порогу.")

    st.markdown("#### 2. Ручное подтверждение и объединение колонок")
    if 'column_merge_history_param' not in st.session_state:
        st.session_state['column_merge_history_param'] = []
    merge_history = st.session_state['column_merge_history_param']

    col1, col2 = st.columns(2)
    with col1:
        col_a = st.selectbox("Колонка 1", columns, key="merge_col_a_param")
    with col2:
        col_b = st.selectbox("Колонка 2", [c for c in columns if c != col_a], key="merge_col_b_param")
    new_col_name = st.text_input("Новое имя для объединённой колонки", value=f"{col_a}_{col_b}_merged", key="merge_col_name_param")
    if st.button("Объединить выбранные колонки и зафиксировать", key="merge_btn_param"):
        if col_a and col_b and new_col_name:
            df_param[new_col_name] = df_param[col_a].combine_first(df_param[col_b])
            merge_history.append({
                "col_a": col_a,
                "col_b": col_b,
                "new_col": new_col_name
            })
            st.session_state['column_merge_history_param'] = merge_history
            st.session_state['final_df_for_param_page'] = df_param
            st.success(f"Колонки '{col_a}' и '{col_b}' объединены в '{new_col_name}' и зафиксированы.")
            st.dataframe(df_param[[col_a, col_b, new_col_name]].head(10))
        else:
            st.warning("Выберите обе колонки и введите новое имя.")

    if merge_history:
        st.markdown("#### История объединений колонок:")
        for i, rec in enumerate(merge_history, 1):
            st.write(f"{i}. {rec['col_a']} + {rec['col_b']} → {rec['new_col']}")
    else:
        st.info("Пока не было объединений колонок.")
else:
    st.warning("Нет данных для обработки. Передайте DataFrame через сессию.")
