import streamlit as st
import pandas as pd
import re
import os
import numpy as np
from collections import Counter

st.title("Стандартизация и нормализация параметров")

# Всегда загружаем только из grouped_categories.csv
csv_path = os.path.join(os.path.dirname(__file__), '..', 'grouped_categories.csv')
try:
    # Используем session state для сохранения изменений
    if 'df_standardization' not in st.session_state:
        st.session_state['df_standardization'] = pd.read_csv(csv_path, encoding='utf-8-sig')
    df_param = st.session_state['df_standardization']
    st.info(f"Загружены данные из grouped_categories.csv ({len(df_param)} строк, {len(df_param.columns)} колонок)")
except Exception as e:
    df_param = pd.DataFrame()
    st.session_state['df_standardization'] = df_param
    st.warning(f"Не удалось загрузить grouped_categories.csv: {e}")

# --- Словари единиц измерения ---
UNIT_PATTERNS = {
    # Мощность
    'power': {
        'patterns': [r'(\d+(?:\.\d+)?)\s*(kw|кв|квт|watt|w|ватт|вт)', 
                    r'(\d+(?:\.\d+)?)\s*(hp|л\.с|лс)', 
                    r'(\d+(?:\.\d+)?)\s*(mw|мвт)'],
        'units': ['kw', 'w', 'hp', 'mw'],
        'default_unit': 'w'
    },
    # Размеры
    'size': {
        'patterns': [r'(\d+(?:\.\d+)?)\s*(m|м|meter|метр)', 
                    r'(\d+(?:\.\d+)?)\s*(cm|см|сантиметр)', 
                    r'(\d+(?:\.\d+)?)\s*(mm|мм|миллиметр)',
                    r'(\d+(?:\.\d+)?)\s*(km|км|километр)',
                    r'(\d+(?:\.\d+)?)\s*(inch|дюйм|"|\')',
                    r'(\d+(?:\.\d+)?)\s*(ft|фут)'],
        'units': ['m', 'cm', 'mm', 'km', 'inch', 'ft'],
        'default_unit': 'cm'
    },
    # Вес
    'weight': {
        'patterns': [r'(\d+(?:\.\d+)?)\s*(kg|кг|килограмм)', 
                    r'(\d+(?:\.\d+)?)\s*(g|г|грамм)', 
                    r'(\d+(?:\.\d+)?)\s*(t|т|тонн)',
                    r'(\d+(?:\.\d+)?)\s*(lb|фунт)',
                    r'(\d+(?:\.\d+)?)\s*(oz|унция)'],
        'units': ['kg', 'g', 't', 'lb', 'oz'],
        'default_unit': 'kg'
    },
    # Объем
    'volume': {
        'patterns': [r'(\d+(?:\.\d+)?)\s*(l|л|литр)', 
                    r'(\d+(?:\.\d+)?)\s*(ml|мл|миллилитр)', 
                    r'(\d+(?:\.\d+)?)\s*(m3|м3|куб)',
                    r'(\d+(?:\.\d+)?)\s*(gallon|галлон)'],
        'units': ['l', 'ml', 'm3', 'gallon'],
        'default_unit': 'l'
    },
    # Температура
    'temperature': {
        'patterns': [r'(\d+(?:\.\d+)?)\s*(°c|c|цельси)', 
                    r'(\d+(?:\.\d+)?)\s*(°f|f|фаренгейт)', 
                    r'(\d+(?:\.\d+)?)\s*(k|кельвин)'],
        'units': ['°c', '°f', 'k'],
        'default_unit': '°c'
    },
    # Время
    'time': {
        'patterns': [r'(\d+(?:\.\d+)?)\s*(sec|с|секунд)', 
                    r'(\d+(?:\.\d+)?)\s*(min|м|минут)', 
                    r'(\d+(?:\.\d+)?)\s*(h|ч|час)',
                    r'(\d+(?:\.\d+)?)\s*(day|д|день|дней)',
                    r'(\d+(?:\.\d+)?)\s*(year|г|год|лет)'],
        'units': ['sec', 'min', 'h', 'day', 'year'],
        'default_unit': 'h'
    }
}

# --- Функции анализа и стандартизации ---

def extract_numeric_values(text):
    """Извлекает числовые значения из текста"""
    if pd.isna(text):
        return []
    
    # Ищем числа с единицами измерения
    numeric_pattern = r'(\d+(?:\.\d+)?(?:,\d+)?)\s*([a-zA-Zа-яё°"\'\s]+)?'
    matches = re.findall(numeric_pattern, str(text), re.IGNORECASE)
    
    results = []
    for match in matches:
        try:
            # Конвертируем число (заменяем запятую на точку)
            number = float(match[0].replace(',', '.'))
            unit = match[1].strip() if match[1] else ''
            results.append((number, unit))
        except ValueError:
            continue
    
    return results

def detect_measurement_type(column_name, values_sample):
    """Определяет тип измерения на основе названия колонки и примеров значений"""
    column_lower = column_name.lower()
    
    # Анализируем название колонки
    type_keywords = {
        'power': ['power', 'мощность', 'watt', 'ватт', 'kw', 'квт'],
        'size': ['size', 'length', 'width', 'height', 'dimension', 'размер', 'длина', 'ширина', 'высота', 'диаметр'],
        'weight': ['weight', 'mass', 'вес', 'масса', 'kg', 'кг'],
        'volume': ['volume', 'capacity', 'объем', 'емкость', 'литр', 'liter'],
        'temperature': ['temp', 'temperature', 'температура', 'градус'],
        'time': ['time', 'duration', 'время', 'продолжительность', 'час', 'минут']
    }
    
    # Проверяем ключевые слова в названии
    for measurement_type, keywords in type_keywords.items():
        for keyword in keywords:
            if keyword in column_lower:
                return measurement_type
    
    # Анализируем значения
    unit_counts = Counter()
    for value in values_sample:
        if pd.notna(value):
            for measurement_type, config in UNIT_PATTERNS.items():
                for pattern in config['patterns']:
                    if re.search(pattern, str(value), re.IGNORECASE):
                        unit_counts[measurement_type] += 1
    
    # Возвращаем наиболее частый тип
    if unit_counts:
        return unit_counts.most_common(1)[0][0]
    
    return 'unknown'

def extract_unit_from_value(value, measurement_type):
    """Извлекает единицу измерения из значения"""
    if pd.isna(value) or measurement_type == 'unknown':
        return None
    
    config = UNIT_PATTERNS.get(measurement_type, {})
    patterns = config.get('patterns', [])
    
    for pattern in patterns:
        match = re.search(pattern, str(value), re.IGNORECASE)
        if match:
            return match.group(2).lower()
    
    return None

def standardize_value(value, measurement_type, target_unit=None):
    """Стандартизирует значение с единицей измерения"""
    if pd.isna(value):
        return value
    
    numeric_values = extract_numeric_values(value)
    if not numeric_values:
        return value
    
    # Берем первое числовое значение
    number, unit = numeric_values[0]
    
    if measurement_type == 'unknown':
        return value
    
    config = UNIT_PATTERNS.get(measurement_type, {})
    default_unit = target_unit or config.get('default_unit', '')
    
    # Если единица не определена, добавляем единицу по умолчанию
    if not unit:
        detected_unit = extract_unit_from_value(value, measurement_type)
        if detected_unit:
            unit = detected_unit
        else:
            unit = default_unit
    
    # Форматируем результат
    if unit:
        return f"{number} {unit}"
    else:
        return str(number)

def analyze_column_statistics(df, column):
    """Анализирует статистику колонки"""
    values = df[column].dropna()
    
    # Общая статистика
    total_values = len(values)
    numeric_values = []
    units_found = Counter()
    
    for value in values:
        nums = extract_numeric_values(value)
        for num, unit in nums:
            numeric_values.append(num)
            if unit:
                units_found[unit.lower()] += 1
    
    stats = {
        'total_values': total_values,
        'numeric_count': len(numeric_values),
        'units_found': dict(units_found.most_common(10)),
        'numeric_stats': {}
    }
    
    if numeric_values:
        stats['numeric_stats'] = {
            'min': min(numeric_values),
            'max': max(numeric_values),
            'mean': sum(numeric_values) / len(numeric_values),
            'count': len(numeric_values)
        }
    
    return stats

# --- Основной интерфейс ---

if not df_param.empty:
    st.dataframe(df_param.head(20))
    all_columns = list(df_param.columns)

    # Исключаем системные колонки
    system_columns = [
        col for col in all_columns
        if any(keyword in col.lower() for keyword in ['name', 'id', 'source_file', 'category'])
    ]

    st.markdown("#### Исключить колонки из анализа")
    excluded_cols = st.multiselect(
        "Выберите колонки, которые не нужно анализировать:",
        options=all_columns,
        default=system_columns
    )

    analysis_columns = [c for c in all_columns if c not in excluded_cols]
    
    # --- Этап 1: Анализ числовых значений ---
    with st.expander("#### 1. 🔍 Анализ числовых значений в параметрах", expanded=True):
        st.markdown("**Определение колонок с числовыми значениями**")
        
        if st.button("🔍 Анализировать числовые значения", key="analyze_numeric"):
            if not analysis_columns:
                st.warning("Нет колонок для анализа после исключения.")
            else:
                # Анализируем каждую колонку
                numeric_analysis = {}
                
                progress_bar = st.progress(0)
                for i, column in enumerate(analysis_columns):
                    progress_bar.progress((i + 1) / len(analysis_columns))
                    
                    stats = analyze_column_statistics(df_param, column)
                    if stats['numeric_count'] > 0:
                        numeric_analysis[column] = stats
                
                progress_bar.empty()
                
                # Сохраняем результаты
                st.session_state['numeric_analysis'] = numeric_analysis
                
                if numeric_analysis:
                    st.success(f"Найдено {len(numeric_analysis)} колонок с числовыми значениями!")
                else:
                    st.info("Колонки с числовыми значениями не найдены.")
        
        # Отображение результатов анализа
        if 'numeric_analysis' in st.session_state and st.session_state['numeric_analysis']:
            st.markdown("**📊 Результаты анализа числовых значений**")
            
            analysis_data = []
            for column, stats in st.session_state['numeric_analysis'].items():
                row = {
                    'Колонка': column,
                    'Всего значений': stats['total_values'],
                    'Числовых значений': stats['numeric_count'],
                    '% числовых': f"{(stats['numeric_count'] / stats['total_values'] * 100):.1f}%",
                    'Мин': stats['numeric_stats'].get('min', 0),
                    'Макс': stats['numeric_stats'].get('max', 0),
                    'Среднее': f"{stats['numeric_stats'].get('mean', 0):.2f}",
                    'Единицы': ', '.join(list(stats['units_found'].keys())[:3])
                }
                analysis_data.append(row)
            
            analysis_df = pd.DataFrame(analysis_data)
            st.dataframe(analysis_df, use_container_width=True)

    # --- Этап 2: Определение единиц измерения ---
    with st.expander("#### 2. 📏 Определение единиц измерения", expanded=False):
        st.markdown("**Интеллектуальное определение типов измерений**")
        
        if 'numeric_analysis' in st.session_state and st.session_state['numeric_analysis']:
            
            if st.button("🔍 Определить единицы измерения", key="detect_units"):
                measurement_analysis = {}
                
                for column in st.session_state['numeric_analysis'].keys():
                    # Берем образец значений для анализа
                    sample_values = df_param[column].dropna().head(50).tolist()
                    measurement_type = detect_measurement_type(column, sample_values)
                    
                    # Анализируем единицы в значениях
                    detected_units = Counter()
                    for value in sample_values:
                        unit = extract_unit_from_value(value, measurement_type)
                        if unit:
                            detected_units[unit] += 1
                    
                    measurement_analysis[column] = {
                        'type': measurement_type,
                        'detected_units': dict(detected_units.most_common(5)),
                        'sample_values': sample_values[:10]
                    }
                
                st.session_state['measurement_analysis'] = measurement_analysis
                st.success("Анализ единиц измерения завершен!")
            
            # Отображение результатов определения единиц
            if 'measurement_analysis' in st.session_state:
                st.markdown("**📏 Результаты определения единиц измерения**")
                
                measurement_data = []
                for column, analysis in st.session_state['measurement_analysis'].items():
                    row = {
                        'Колонка': column,
                        'Тип измерения': analysis['type'],
                        'Найденные единицы': ', '.join(analysis['detected_units'].keys()),
                        'Примеры значений': ', '.join([str(v) for v in analysis['sample_values'][:3]])
                    }
                    measurement_data.append(row)
                
                measurement_df = pd.DataFrame(measurement_data)
                st.dataframe(measurement_df, use_container_width=True)
        else:
            st.info("Сначала выполните анализ числовых значений в разделе выше.")

    # --- Этап 3: Стандартизация значений ---
    with st.expander("#### 3. ⚙️ Стандартизация значений", expanded=False):
        st.markdown("**Применение стандартных единиц измерения**")
        
        if 'measurement_analysis' in st.session_state:
            
            # Выбор колонок для стандартизации
            standardizable_columns = list(st.session_state['measurement_analysis'].keys())
            selected_columns = st.multiselect(
                "Выберите колонки для стандартизации:",
                options=standardizable_columns,
                default=standardizable_columns
            )
            
            if selected_columns:
                # Настройки стандартизации для каждой колонки
                st.markdown("**Настройки стандартизации:**")
                standardization_config = {}
                
                for column in selected_columns:
                    analysis = st.session_state['measurement_analysis'][column]
                    measurement_type = analysis['type']
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**{column}** (тип: {measurement_type})")
                    
                    with col2:
                        if measurement_type in UNIT_PATTERNS:
                            available_units = UNIT_PATTERNS[measurement_type]['units']
                            default_unit = UNIT_PATTERNS[measurement_type]['default_unit']
                            
                            target_unit = st.selectbox(
                                f"Целевая единица для {column}:",
                                options=available_units,
                                index=available_units.index(default_unit) if default_unit in available_units else 0,
                                key=f"unit_{column}"
                            )
                            
                            standardization_config[column] = {
                                'type': measurement_type,
                                'target_unit': target_unit
                            }
                
                # Кнопка стандартизации
                if st.button("✅ Применить стандартизацию", key="apply_standardization"):
                    standardized_df = df_param.copy()
                    
                    for column, config in standardization_config.items():
                        measurement_type = config['type']
                        target_unit = config['target_unit']
                        
                        # Создаем новую колонку со стандартизированными значениями
                        new_column_name = f"{column}_standardized"
                        standardized_df[new_column_name] = standardized_df[column].apply(
                            lambda x: standardize_value(x, measurement_type, target_unit)
                        )
                    
                    # Сохраняем результат
                    st.session_state['df_standardization'] = standardized_df
                    st.success("Стандартизация применена!")
                    
                    # Показываем примеры
                    standardized_cols = [col for col in standardized_df.columns if '_standardized' in col]
                    if standardized_cols:
                        st.markdown("**Примеры стандартизированных значений:**")
                        comparison_cols = []
                        for col in standardized_cols:
                            original_col = col.replace('_standardized', '')
                            if original_col in standardized_df.columns:
                                comparison_cols.extend([original_col, col])
                        
                        if comparison_cols:
                            st.dataframe(standardized_df[comparison_cols].head(10))
        else:
            st.info("Сначала выполните определение единиц измерения в разделе выше.")

    # --- Сохранение результатов ---
    with st.expander("#### 4. 💾 Сохранение результатов", expanded=False):
        
        # Показываем текущее состояние
        if 'df_standardization' in st.session_state:
            current_df = st.session_state['df_standardization']
            st.info(f"📊 Текущее состояние: {len(current_df)} строк, {len(current_df.columns)} колонок")
            standardized_count = len([col for col in current_df.columns if '_standardized' in col])
            if standardized_count > 0:
                st.success(f"✅ Стандартизированных колонок: {standardized_count}")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("💾 Сохранить в grouped_categories.csv", key="save_main_std"):
                try:
                    output_path = os.path.join(os.path.dirname(__file__), '..', 'grouped_categories.csv')
                    current_df = st.session_state.get('df_standardization', df_param)
                    current_df.to_csv(output_path, index=False, encoding='utf-8-sig')
                    st.success(f"✅ Данные сохранены в {output_path}")
                    st.success(f"📊 Сохранено: {len(current_df)} строк, {len(current_df.columns)} колонок")
                except Exception as e:
                    st.error(f"❌ Ошибка при сохранении: {e}")
        
        with col2:
            if st.button("📥 Скачать как CSV", key="download_csv_std"):
                current_df = st.session_state.get('df_standardization', df_param)
                csv_data = current_df.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    label="⬇️ Скачать файл",
                    data=csv_data,
                    file_name="standardized_parameters.csv",
                    mime="text/csv",
                    key="download_standardized"
                )
        
        with col3:
            if st.button("🔄 Обновить исходные данные", key="reload_data_std"):
                # Очищаем кэш и session state
                for key in ['df_standardization', 'numeric_analysis', 'measurement_analysis']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.success("Данные обновлены! Страница будет перезагружена.")
                st.rerun()

    # --- Статистика ---
    with st.expander("#### 5. 📊 Статистика по данным", expanded=False):
        current_df = st.session_state.get('df_standardization', df_param)
        total_cols = len(current_df.columns)
        standardized_cols = len([col for col in current_df.columns if '_standardized' in col])
        total_rows = len(current_df)
        
        stat_col1, stat_col2, stat_col3 = st.columns(3)
        with stat_col1:
            st.metric("Всего колонок", total_cols)
        with stat_col2:
            st.metric("Стандартизированных колонок", standardized_cols)
        with stat_col3:
            st.metric("Всего строк", total_rows)
        
        # Показываем текущее состояние данных
        if st.checkbox("Показать все данные", key="show_all_data_std"):
            st.dataframe(current_df, use_container_width=True)
        else:
            st.markdown("**Первые 20 строк:**")
            st.dataframe(current_df.head(20), use_container_width=True)

else:
    st.warning("Нет данных для обработки. Сначала выполните основную обработку в главном приложении.")
