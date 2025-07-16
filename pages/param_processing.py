import streamlit as st
import pandas as pd
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer, util
import os
import numpy as np

st.title("Обработка параметров финального результата")

# Всегда загружаем только из grouped_categories.csv
csv_path = os.path.join(os.path.dirname(__file__), '..', 'grouped_categories.csv')
try:
    # Используем session state для сохранения изменений
    if 'df_param' not in st.session_state:
        st.session_state['df_param'] = pd.read_csv(csv_path, encoding='utf-8-sig')
    df_param = st.session_state['df_param']
    st.info(f"Загружены данные из grouped_categories.csv ({len(df_param)} строк, {len(df_param.columns)} колонок)")
except Exception as e:
    df_param = pd.DataFrame()
    st.session_state['df_param'] = df_param
    st.warning(f"Не удалось загрузить grouped_categories.csv: {e}")

# --- Загрузка модели sentence-transformers ---
@st.cache_resource(show_spinner=False)
def get_st_model():
    return SentenceTransformer('distiluse-base-multilingual-cased', device='cpu')

# --- Record Linkage схожесть колонок ---
def record_linkage_similarity(col1, col2):
    """Простая реализация Record Linkage для названий колонок"""
    try:
        import recordlinkage as rl
        # Создаем простой индекс для сравнения
        indexer = rl.Index()
        indexer.full()
        
        # Создаем DataFrame с названиями колонок
        df_cols = pd.DataFrame({'column': [col1, col2]})
        candidate_links = indexer.index(df_cols)
        
        # Создаем объект сравнения
        compare_cl = rl.Compare()
        compare_cl.string('column', 'column', method='jarowinkler', threshold=0.0)
        
        features = compare_cl.compute(candidate_links, df_cols)
        if len(features) > 0:
            return float(features.iloc[0, 0]) * 100
        return 0
    except ImportError:
        # Fallback если recordlinkage не установлен
        # Используем простую Jaro-Winkler схожесть через rapidfuzz
        from rapidfuzz.distance import JaroWinkler
        return int(JaroWinkler.similarity(col1, col2) * 100)
    except Exception:
        return 0

# --- Проверка совместимости колонок для умного объединения ---
def check_column_compatibility(df, col1, col2, smart_merge=True):
    """
    Проверяет совместимость двух колонок для объединения
    
    Args:
        df: DataFrame с данными
        col1, col2: названия колонок для проверки
        smart_merge: использовать ли умную логику объединения
    
    Returns:
        dict с результатами проверки
    """
    result = {
        'compatible': True,
        'reason': '',
        'sources_different': False,
        'non_conflicting': True,
        'conflicts_count': 0,
        'both_filled_count': 0,
        'total_rows': len(df)
    }
    
    if not smart_merge:
        return result
    
    # Проверка 1: Разные источники
    if 'source_file' in df.columns:
        sources1 = set(df[df[col1].notna()]['source_file'])
        sources2 = set(df[df[col2].notna()]['source_file'])
        if sources1 and sources2:
            result['sources_different'] = len(sources1 & sources2) < max(len(sources1), len(sources2))
    
    # Проверка 2: Непересекающиеся/совместимые значения
    both_filled = df[(df[col1].notna()) & (df[col2].notna())]
    result['both_filled_count'] = len(both_filled)
    
    if len(both_filled) > 0:
        # Проверяем конфликтующие значения (разные непустые значения)
        conflicts = both_filled[both_filled[col1].astype(str) != both_filled[col2].astype(str)]
        result['conflicts_count'] = len(conflicts)
        
        # Если больше 10% конфликтов, считаем несовместимыми
        conflict_ratio = len(conflicts) / len(both_filled)
        if conflict_ratio > 0.1:
            result['non_conflicting'] = False
            result['reason'] = f"Много конфликтов: {len(conflicts)}/{len(both_filled)} ({conflict_ratio:.1%})"
    
    # Итоговая совместимость
    result['compatible'] = result['sources_different'] or result['non_conflicting']
    
    if not result['compatible']:
        if not result['sources_different'] and not result['non_conflicting']:
            result['reason'] = "Одинаковые источники И есть конфликты значений"
    
    return result

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
    col1, col2, col3 = st.columns(3)
    with col1:
        use_rapidfuzz = st.checkbox("RapidFuzz", value=True, key="use_rapidfuzz")
    with col2:
        use_sentence_transformers = st.checkbox("Sentence-Transformers", value=False, key="use_st")
    with col3:
        use_record_linkage = st.checkbox("Record Linkage", value=False, key="use_rl", 
                                       help="Для использования установите: pip install recordlinkage")
    
    # --- Individual Threshold Sliders ---
    rf_threshold = None
    st_threshold = None
    rl_threshold = None
    
    if use_rapidfuzz:
        rf_threshold = st.slider("Порог схожести для RapidFuzz (%)", 60, 100, 85, 1, key="rf_thresh")
    
    if use_sentence_transformers:
        st_threshold = st.slider("Порог схожести для Sentence-Transformers (%)", 60, 100, 75, 1, key="st_thresh")
    
    if use_record_linkage:
        rl_threshold = st.slider("Порог схожести для Record Linkage (%)", 60, 100, 80, 1, key="rl_thresh")
    
    # Информация о выбранных методах
    selected_methods = []
    if use_rapidfuzz: selected_methods.append("RapidFuzz")
    if use_sentence_transformers: selected_methods.append("Sentence-Transformers") 
    if use_record_linkage: selected_methods.append("Record Linkage")
    
    if selected_methods:
        st.info(f"Выбранные методы: {', '.join(selected_methods)}")
    else:
        st.warning("Не выбран ни один метод сравнения!")

    group_diff_sources = st.checkbox(
        "Умное объединение: только непересекающиеся значения",
        value=True,
        help="""Если включено, объединение происходит только когда:
        1. Параметры из разных источников (source_file), И/ИЛИ
        2. У конкретного товара заполнен только один из параметров (А или Б, но не оба)
        Это предотвращает потерю данных и конфликты значений."""
    )
    
    # --- Автоматический поиск и найденные пары ---
    with st.expander("#### 2. 🔍 Автоматический поиск похожих колонок", expanded=True):
        st.markdown("**Поиск и результаты**")
        
        # Инициализируем состояние для контроля процесса
        if 'start_search' not in st.session_state:
            st.session_state['start_search'] = False
        if 'confirmed_long' not in st.session_state:
            st.session_state['confirmed_long'] = False
        
        # Кнопка запуска поиска
        if st.button("🔍 Найти похожие колонки", key="find_similar"):
            if len(columns) == 0:
                st.warning("Нет колонок для анализа после исключения.")
            elif not any([use_rapidfuzz, use_sentence_transformers, use_record_linkage]):
                st.warning("Выберите хотя бы один метод сравнения!")
            elif len(columns) > 100:
                st.error(f"❌ Слишком много колонок: {len(columns)}. Максимум 100 колонок для предотвращения зависания!")
            elif len(columns) > 30:
                st.warning(f"⚠️ Внимание: выбрано {len(columns)} колонок. Это займет {len(columns)*(len(columns)-1)//2} сравнений и может занять время!")
                st.session_state['start_search'] = True
                st.session_state['confirmed_long'] = False
            else:
                st.session_state['start_search'] = True
                st.session_state['confirmed_long'] = True
        
        # Если много колонок, показываем подтверждение
        if st.session_state['start_search'] and not st.session_state['confirmed_long']:
            if st.checkbox("Я понимаю, что это может занять время", key="confirm_long"):
                st.session_state['confirmed_long'] = True
                st.rerun()
        
        # Выполняем поиск если подтверждено
        if st.session_state['start_search'] and st.session_state['confirmed_long']:
            similar_pairs = []
            
            # Подсчитываем количество активных методов
            active_methods = sum([
                use_rapidfuzz and rf_threshold is not None,
                use_sentence_transformers and st_threshold is not None, 
                use_record_linkage and rl_threshold is not None
            ])
            
            comparisons_per_method = len(columns) * (len(columns) - 1) // 2
            total_comparisons = comparisons_per_method * active_methods
            
            st.info(f"Выполняется {total_comparisons} сравнений с помощью {active_methods} методов...")
            
            # Создаем progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            comparison_count = 0
            
            # RapidFuzz method
            if use_rapidfuzz and rf_threshold is not None:
                status_text.text("Выполняется поиск с помощью RapidFuzz...")
                with st.spinner("Анализ схожести с помощью RapidFuzz..."):
                    for i in range(len(columns)):
                        for j in range(i+1, len(columns)):
                            col1, col2 = columns[i], columns[j]
                            comparison_count += 1
                            
                            # Обновляем progress bar каждые 10 сравнений
                            if comparison_count % 10 == 0:
                                progress_bar.progress(comparison_count / total_comparisons)
                            
                            # Проверяем совместимость колонок
                            compatibility = check_column_compatibility(df_param, col1, col2, group_diff_sources)
                            if not compatibility['compatible']:
                                continue
                            
                            rf_score = fuzz.token_sort_ratio(col1, col2)
                            if rf_score >= rf_threshold:
                                # Подсчитываем количество непустых значений
                                count1 = df_param[col1].notna().sum()
                                count2 = df_param[col2].notna().sum()
                                overlap = df_param[(df_param[col1].notna()) & (df_param[col2].notna())].shape[0]
                                
                                similar_pairs.append({
                                    "Колонка 1": col1,
                                    "Колонка 2": col2,
                                    "Схожесть %": rf_score,
                                    "Метод": "RapidFuzz",
                                    "Значений в 1": count1,
                                    "Значений в 2": count2,
                                    "Пересечений": overlap,
                                    "Конфликтов": compatibility['conflicts_count'],
                                    "Разные источники": "✅" if compatibility['sources_different'] else "❌",
                                    "Потенциал": count1 + count2 - overlap
                                })
            
            # Sentence-Transformers method
            if use_sentence_transformers and st_threshold is not None:
                status_text.text("Выполняется поиск с помощью Sentence-Transformers...")
                with st.spinner("Анализ схожести с помощью Sentence-Transformers..."):
                    st_model = get_st_model()
                    # Предварительно вычисляем все embeddings ОДИН РАЗ
                    embeddings = st_model.encode(columns, convert_to_tensor=True)
                    
                    for i in range(len(columns)):
                        for j in range(i+1, len(columns)):
                            col1, col2 = columns[i], columns[j]
                            comparison_count += 1
                            
                            # Обновляем progress bar каждые 10 сравнений
                            if comparison_count % 10 == 0:
                                progress_bar.progress(comparison_count / total_comparisons)
                            
                            # Проверяем совместимость колонок
                            compatibility = check_column_compatibility(df_param, col1, col2, group_diff_sources)
                            if not compatibility['compatible']:
                                continue
                            
                            # Вычисляем схожесть (используем уже готовые embeddings)
                            st_score = int(util.cos_sim(embeddings[i], embeddings[j]).item() * 100)
                            if st_score >= st_threshold:
                                # Подсчитываем количество непустых значений
                                count1 = df_param[col1].notna().sum()
                                count2 = df_param[col2].notna().sum()
                                overlap = df_param[(df_param[col1].notna()) & (df_param[col2].notna())].shape[0]
                                
                                similar_pairs.append({
                                    "Колонка 1": col1,
                                    "Колонка 2": col2,
                                    "Схожесть %": st_score,
                                    "Метод": "SentenceTransformers",
                                    "Значений в 1": count1,
                                    "Значений в 2": count2,
                                    "Пересечений": overlap,
                                    "Конфликтов": compatibility['conflicts_count'],
                                    "Разные источники": "✅" if compatibility['sources_different'] else "❌",
                                    "Потенциал": count1 + count2 - overlap
                                })
            
            # Record Linkage method
            if use_record_linkage and rl_threshold is not None:
                status_text.text("Выполняется поиск с помощью Record Linkage...")
                with st.spinner("Анализ схожести с помощью Record Linkage..."):
                    for i in range(len(columns)):
                        for j in range(i+1, len(columns)):
                            col1, col2 = columns[i], columns[j]
                            comparison_count += 1
                            
                            # Обновляем progress bar каждые 10 сравнений
                            if comparison_count % 10 == 0:
                                progress_bar.progress(comparison_count / total_comparisons)
                            
                            # Проверяем совместимость колонок
                            compatibility = check_column_compatibility(df_param, col1, col2, group_diff_sources)
                            if not compatibility['compatible']:
                                continue
                            
                            rl_score = record_linkage_similarity(col1, col2)
                            if rl_score >= rl_threshold:
                                # Подсчитываем количество непустых значений
                                count1 = df_param[col1].notna().sum()
                                count2 = df_param[col2].notna().sum()
                                overlap = df_param[(df_param[col1].notna()) & (df_param[col2].notna())].shape[0]
                                
                                similar_pairs.append({
                                    "Колонка 1": col1,
                                    "Колонка 2": col2,
                                    "Схожесть %": int(rl_score),
                                    "Метод": "RecordLinkage",
                                    "Значений в 1": count1,
                                    "Значений в 2": count2,
                                    "Пересечений": overlap,
                                    "Конфликтов": compatibility['conflicts_count'],
                                    "Разные источники": "✅" if compatibility['sources_different'] else "❌",
                                    "Потенциал": count1 + count2 - overlap
                                })
            
            # Завершаем progress bar
            status_text.text("Завершено!")
            progress_bar.progress(1.0)
            progress_bar.empty()
            status_text.empty()
            
            # Remove duplicates and create DataFrame
            seen_pairs = set()
            unique_pairs = []
            for pair in similar_pairs:
                pair_key = tuple(sorted([pair["Колонка 1"], pair["Колонка 2"]]))
                if pair_key not in seen_pairs:
                    seen_pairs.add(pair_key)
                    unique_pairs.append(pair)
            
            # Сохраняем результаты в session state
            if unique_pairs:
                sim_col_df = pd.DataFrame(unique_pairs).sort_values("Схожесть %", ascending=False)
                st.session_state['similar_pairs_df'] = sim_col_df
                st.success(f"Найдено {len(unique_pairs)} похожих пар!")
            else:
                st.info("Похожих колонок не найдено по текущим порогам.")
                st.session_state['similar_pairs_df'] = pd.DataFrame()
            
            # Сбрасываем состояние поиска
            st.session_state['start_search'] = False
            st.session_state['confirmed_long'] = False

        # --- Отображение найденных пар ---
        if 'similar_pairs_df' in st.session_state and not st.session_state['similar_pairs_df'].empty:
            sim_col_df = st.session_state['similar_pairs_df']
            
            st.markdown("**📋 Найденные похожие пары**")
            st.dataframe(sim_col_df, use_container_width=True)
            
            # Массовое объединение всех найденных пар
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Объединить ВСЕ найденные пары", key="merge_all_auto"):
                    merged_count = 0
                    for _, row in sim_col_df.iterrows():
                        col_a, col_b = row["Колонка 1"], row["Колонка 2"]
                        new_col_name = f"{col_a}_{col_b}_merged"
                        if new_col_name not in df_param.columns:
                            df_param[new_col_name] = df_param[col_a].combine_first(df_param[col_b])
                            merged_count += 1
                    
                    if merged_count > 0:
                        # Обновляем session state
                        st.session_state['df_param'] = df_param
                        st.success(f"Объединено {merged_count} пар колонок! Новые колонки добавлены с суффиксом '_merged'.")
                        # Показываем пример результата
                        merged_cols = [col for col in df_param.columns if '_merged' in col]
                        if merged_cols:
                            st.markdown("**Примеры объединенных колонок:**")
                            st.dataframe(df_param[merged_cols].head(5))
                    else:
                        st.info("Все пары уже были объединены ранее.")
            
            with col2:
                if st.button("🗑️ Очистить результаты поиска", key="clear_results"):
                    st.session_state['similar_pairs_df'] = pd.DataFrame()
                    st.rerun()

    # --- Выборочное объединение ---
    with st.expander("#### 3. 🎯 Выборочное объединение пар", expanded=False):
        if 'similar_pairs_df' in st.session_state and not st.session_state['similar_pairs_df'].empty:
            sim_col_df = st.session_state['similar_pairs_df']
            
            selected_idx = st.selectbox(
                "Выберите пару для объединения:",
                options=range(len(sim_col_df)),
                format_func=lambda x: f"{sim_col_df.iloc[x]['Колонка 1']} ↔ {sim_col_df.iloc[x]['Колонка 2']} ({sim_col_df.iloc[x]['Схожесть %']}% - {sim_col_df.iloc[x]['Метод']})",
                key="selected_pair"
            )
            
            if selected_idx is not None:
                selected_row = sim_col_df.iloc[selected_idx]
                col_a, col_b = selected_row["Колонка 1"], selected_row["Колонка 2"]
                
                # Предварительный просмотр объединения
                st.markdown("**Предварительный просмотр объединения:**")
                preview_df = df_param[[col_a, col_b]].copy()
                preview_df[f"Объединенный_результат"] = preview_df[col_a].combine_first(preview_df[col_b])
                
                # Показываем потенциальные конфликты
                conflicts = preview_df[(preview_df[col_a].notna()) & (preview_df[col_b].notna()) & 
                                      (preview_df[col_a].astype(str) != preview_df[col_b].astype(str))]
                if len(conflicts) > 0:
                    st.warning(f"⚠️ Найдено {len(conflicts)} потенциальных конфликтов значений:")
                    st.dataframe(conflicts.head(5))
                    st.info("💡 При объединении будет использовано значение из первой колонки")
                
                # Показываем полезную статистику
                col_stat1, col_stat2, col_stat3 = st.columns(3)
                with col_stat1:
                    st.metric("Заполнено в колонке 1", preview_df[col_a].notna().sum())
                with col_stat2:
                    st.metric("Заполнено в колонке 2", preview_df[col_b].notna().sum())
                with col_stat3:
                    st.metric("Будет заполнено после объединения", preview_df[f"Объединенный_результат"].notna().sum())
                
                st.dataframe(preview_df.head(10))
                
                # Настройки объединения
                new_col_name = st.text_input(
                    "Имя для новой колонки:", 
                    value=f"{col_a}_{col_b}_merged",
                    key="custom_merge_name"
                )
                
                if st.button("✅ Объединить выбранную пару", key="merge_selected"):
                    if new_col_name and new_col_name not in df_param.columns:
                        df_param[new_col_name] = df_param[col_a].combine_first(df_param[col_b])
                        # Обновляем session state
                        st.session_state['df_param'] = df_param
                        st.success(f"Пара '{col_a}' и '{col_b}' объединена в колонку '{new_col_name}'!")
                        # Удаляем обработанную пару из результатов
                        st.session_state['similar_pairs_df'] = sim_col_df.drop(selected_idx).reset_index(drop=True)
                        st.rerun()
                    elif new_col_name in df_param.columns:
                        st.error(f"Колонка '{new_col_name}' уже существует!")
                    else:
                        st.error("Введите корректное имя для новой колонки!")
        else:
            st.info("Сначала найдите похожие пары в разделе выше")

    # --- Ручное объединение ---
    with st.expander("#### 4. ⚙️ Ручное объединение колонок", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            col_a = st.selectbox("Колонка 1", all_columns, key="merge_col_a_param")
        with col2:
            col_b = st.selectbox("Колонка 2", [c for c in all_columns if c != col_a], key="merge_col_b_param")
        
        # Показываем анализ совместимости для ручного объединения
        if col_a and col_b:
            compatibility = check_column_compatibility(df_param, col_a, col_b, True)
            
            if compatibility['compatible']:
                st.success(f"✅ Колонки совместимы для объединения")
            else:
                st.warning(f"⚠️ Потенциальные проблемы: {compatibility['reason']}")
            
            # Статистика
            stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
            with stat_col1:
                st.metric("Пересечений", compatibility['both_filled_count'])
            with stat_col2:
                st.metric("Конфликтов", compatibility['conflicts_count'])
            with stat_col3:
                st.metric("Разные источники", "Да" if compatibility['sources_different'] else "Нет")
            with stat_col4:
                conflict_ratio = compatibility['conflicts_count'] / max(compatibility['both_filled_count'], 1) * 100
                st.metric("% конфликтов", f"{conflict_ratio:.1f}%")
        
        new_col_name = st.text_input("Новое имя для объединённой колонки", value=f"{col_a}_{col_b}_merged", key="merge_col_name_param")
        if st.button("Объединить выбранные колонки и зафиксировать", key="merge_btn_param"):
            if col_a and col_b and new_col_name:
                df_param[new_col_name] = df_param[col_a].combine_first(df_param[col_b])
                # Обновляем session state
                st.session_state['df_param'] = df_param
                st.success(f"Колонки '{col_a}' и '{col_b}' объединены в '{new_col_name}' и зафиксированы.")
                st.dataframe(df_param[[col_a, col_b, new_col_name]].head(10))
            else:
                st.warning("Выберите обе колонки и введите новое имя.")

    # --- Сохранение результатов ---
    with st.expander("#### 5. 💾 Сохранение результатов", expanded=False):
        
        # Показываем текущее состояние
        if 'df_param' in st.session_state:
            current_df = st.session_state['df_param']
            st.info(f"📊 Текущее состояние: {len(current_df)} строк, {len(current_df.columns)} колонок")
            merged_count = len([col for col in current_df.columns if '_merged' in col])
            if merged_count > 0:
                st.success(f"✅ Объединенных колонок: {merged_count}")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("💾 Сохранить в grouped_categories.csv", key="save_main"):
                try:
                    output_path = os.path.join(os.path.dirname(__file__), '..', 'grouped_categories.csv')
                    # Используем данные из session state
                    current_df = st.session_state.get('df_param', df_param)
                    current_df.to_csv(output_path, index=False, encoding='utf-8-sig')
                    st.success(f"✅ Данные сохранены в {output_path}")
                    st.success(f"📊 Сохранено: {len(current_df)} строк, {len(current_df.columns)} колонок")
                except Exception as e:
                    st.error(f"❌ Ошибка при сохранении: {e}")
        
        with col2:
            if st.button("📥 Скачать как CSV", key="download_csv"):
                # Используем данные из session state
                current_df = st.session_state.get('df_param', df_param)
                csv_data = current_df.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    label="⬇️ Скачать файл",
                    data=csv_data,
                    file_name="processed_parameters.csv",
                    mime="text/csv",
                    key="download_processed"
                )
        
        with col3:
            if st.button("🔄 Обновить исходные данные", key="reload_data"):
                # Очищаем кэш и session state
                st.cache_data.clear()
                st.cache_resource.clear()
                if 'df_param' in st.session_state:
                    del st.session_state['df_param']
                if 'similar_pairs_df' in st.session_state:
                    del st.session_state['similar_pairs_df']
                st.success("Данные обновлены! Страница будет перезагружена.")
                st.rerun()

    # --- Статистика по данным ---
    with st.expander("#### 6. 📊 Статистика по данным", expanded=False):
        
        # Используем актуальные данные из session state
        current_df = st.session_state.get('df_param', df_param)
        total_cols = len(current_df.columns)
        merged_cols = len([col for col in current_df.columns if '_merged' in col])
        total_rows = len(current_df)
        
        stat_col1, stat_col2, stat_col3 = st.columns(3)
        with stat_col1:
            st.metric("Всего колонок", total_cols)
        with stat_col2:
            st.metric("Объединенных колонок", merged_cols)
        with stat_col3:
            st.metric("Всего строк", total_rows)
        
        # Показываем текущее состояние данных
        if st.checkbox("Показать все данные", key="show_all_data"):
            st.dataframe(current_df, use_container_width=True)
        else:
            st.markdown("**Первые 20 строк:**")
            st.dataframe(current_df.head(20), use_container_width=True)

else:
    st.warning("Нет данных для обработки. Сначала выполните основную обработку в главном приложении.")
