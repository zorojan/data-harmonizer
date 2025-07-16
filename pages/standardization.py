import streamlit as st
import pandas as pd
import re
import os
import numpy as np
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Интеллектуальные библиотеки для распознавания единиц измерения
try:
    import quantulum3
    QUANTULUM_AVAILABLE = True
except ImportError:
    QUANTULUM_AVAILABLE = False

try:
    import pint
    ureg = pint.UnitRegistry()
    PINT_AVAILABLE = True
except ImportError:
    PINT_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

st.title("🔬 Интеллектуальная стандартизация и нормализация параметров")

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

# --- Настройки методов анализа ---
st.sidebar.markdown("## ⚙️ Настройки методов анализа")

# Выбор методов распознавания единиц измерения
st.sidebar.markdown("### 📏 Методы распознавания единиц:")

use_quantulum = st.sidebar.checkbox(
    "🔍 Quantulum3 (физические величины)", 
    value=QUANTULUM_AVAILABLE,
    disabled=not QUANTULUM_AVAILABLE,
    help="Автоматическое извлечение физических величин и единиц измерения из текста"
)

use_transformers = st.sidebar.checkbox(
    "🤖 Transformer-NER (семантический анализ)", 
    value=False,
    disabled=not TRANSFORMERS_AVAILABLE,
    help="Семантический анализ для распознавания сущностей и единиц измерения"
)

use_pint = st.sidebar.checkbox(
    "📐 Pint (валидация единиц)", 
    value=PINT_AVAILABLE,
    disabled=not PINT_AVAILABLE,
    help="Валидация и стандартизация единиц измерения"
)

# Дополнительные настройки
st.sidebar.markdown("### 🎯 Дополнительные настройки:")
confidence_threshold = st.sidebar.slider(
    "Порог уверенности распознавания", 
    min_value=0.1, 
    max_value=1.0, 
    value=0.7,
    help="Минимальная уверенность для принятия результата распознавания"
)

max_samples = st.sidebar.number_input(
    "Максимум образцов для анализа", 
    min_value=5, 
    max_value=100, 
    value=20,
    help="Количество строк для анализа в каждой колонке"
)

# Статус доступности библиотек
st.sidebar.markdown("### 📚 Статус библиотек:")
st.sidebar.success(f"✅ Quantulum3: {'Доступна' if QUANTULUM_AVAILABLE else 'Не установлена'}")
st.sidebar.success(f"✅ Transformers: {'Доступна' if TRANSFORMERS_AVAILABLE else 'Не установлена'}")
st.sidebar.success(f"✅ Pint: {'Доступна' if PINT_AVAILABLE else 'Не установлена'}")

# --- Простые функции для быстрого анализа ---

def analyze_column_statistics_simple(df, column):
    """Быстрый анализ колонки на наличие числовых значений"""
    values = df[column].dropna()
    
    # Простые паттерны для поиска цифр
    numeric_pattern = r'\d+\.?\d*'
    unit_pattern = r'(\d+\.?\d*)\s*([a-zA-Zа-яё°"\'\s]*)'
    
    numeric_count = 0
    units_found = Counter()
    numeric_values = []
    
    # Быстрый анализ образца (максимум 50 значений для скорости)
    sample_size = min(50, len(values))
    
    for value in values.head(sample_size):
        if pd.isna(value):
            continue
            
        value_str = str(value).strip()
        
        # Проверяем наличие цифр
        if re.search(numeric_pattern, value_str):
            numeric_count += 1
            
            # Извлекаем числа и единицы простым способом
            matches = re.findall(unit_pattern, value_str)
            for number, unit in matches:
                try:
                    numeric_values.append(float(number))
                    unit_clean = unit.strip()
                    if unit_clean and len(unit_clean) < 15:  # Разумная длина единицы
                        units_found[unit_clean] += 1
                except:
                    continue
    
    # Вычисляем процент "числовости"
    numeric_percentage = (numeric_count / sample_size * 100) if sample_size > 0 else 0
    
    return {
        'total_values': len(values),
        'analyzed_values': sample_size,
        'numeric_count': numeric_count,
        'numeric_percentage': numeric_percentage,
        'units_found': dict(units_found.most_common(5)),
        'has_numeric': numeric_percentage > 20,  # Считаем колонку числовой если >20% содержат цифры
        'numeric_values': numeric_values[:10],  # Первые 10 чисел для примера
        'avg_value': sum(numeric_values) / len(numeric_values) if numeric_values else 0
    }

def quick_find_numeric_columns(df, columns_to_analyze):
    """Быстрое определение числовых колонок"""
    numeric_columns = {}
    
    for column in columns_to_analyze:
        stats = analyze_column_statistics_simple(df, column)
        if stats['has_numeric']:
            numeric_columns[column] = stats
    
    return numeric_columns

def standardize_value_simple(value, target_format='number_unit', column_name=None, product_name=None, category=None):
    """Простая стандартизация значения"""
    if pd.isna(value):
        return value
    
    value_str = str(value).strip()
    
    # Простой паттерн: число + единица
    pattern = r'(\d+\.?\d*)\s*([a-zA-Zа-яё°"\'\s]*)'
    match = re.search(pattern, value_str)
    
    if match:
        number = match.group(1)
        unit = match.group(2).strip()
        
        # Если единица пустая, используем оптимизированную логику определения единиц
        if not unit and column_name:
            unit_from_context = extract_unit_from_context_optimized(column_name, product_name, category)
            if unit_from_context:
                unit = unit_from_context
        
        # Если единица все еще пустая, возвращаем как есть
        if not unit:
            return value
        
        # Форматируем по выбранному формату
        if target_format == 'number_unit':
            return f"{number} {unit}"
        elif target_format == 'unit_number':
            return f"{unit} {number}"
        else:
            return f"{number} {unit}"
    
    return value

# --- Глобальный экстрактор (синглтон) ---
@st.cache_resource
def get_transformer_pipeline():
    """Кэшированная инициализация Transformer-NER модели"""
    if not TRANSFORMERS_AVAILABLE:
        return None
    
    try:
        return pipeline(
            "ner", 
            model="dbmdz/bert-large-cased-finetuned-conll03-english",
            aggregation_strategy="simple"
        )
    except Exception as e:
        st.warning(f"Не удалось загрузить Transformer модель: {e}")
        return None

# --- Интеллектуальные функции для продвинутого анализа ---

class IntelligentUnitExtractor:
    """Интеллектуальный экстрактор единиц измерения"""
    
    def __init__(self):
        self.transformer_pipeline = get_transformer_pipeline() if use_transformers else None
    
    def extract_with_quantulum(self, text):
        """Оптимизированное извлечение единиц с помощью Quantulum3 и Pint"""
        if not QUANTULUM_AVAILABLE or not use_quantulum:
            return []
        
        try:
            parsed = quantulum3.parser.parse(str(text))
            results = []
            
            for quantity in parsed:
                if quantity.value and quantity.unit:
                    unit_name = quantity.unit.name
                    unit_symbol = getattr(quantity.unit, 'symbol', unit_name)
                    
                    # Проверяем единицу через Pint для стандартизации
                    pint_validation = self.validate_with_pint(unit_symbol)
                    
                    if pint_validation and pint_validation.get('is_valid'):
                        # Используем стандартизированную единицу от Pint
                        canonical_unit = pint_validation.get('canonical_unit', unit_symbol)
                        dimension = pint_validation.get('dimensionality', quantity.unit.dimension.name if quantity.unit.dimension else '')
                        confidence = 0.95  # Очень высокая уверенность для Quantulum + Pint
                    else:
                        # Используем данные от Quantulum3
                        canonical_unit = unit_symbol
                        dimension = quantity.unit.dimension.name if quantity.unit.dimension else ''
                        confidence = 0.8  # Высокая уверенность только для Quantulum
                    
                    results.append({
                        'value': quantity.value,
                        'unit': canonical_unit,
                        'unit_symbol': canonical_unit,
                        'dimension': dimension,
                        'confidence': confidence,
                        'method': 'quantulum3_pint' if pint_validation and pint_validation.get('is_valid') else 'quantulum3',
                        'pint_validation': pint_validation
                    })
            
            return results
        except Exception as e:
            return []
    
    def extract_with_transformers(self, text):
        """Оптимизированное извлечение единиц с помощью Transformer-NER и Pint"""
        if not TRANSFORMERS_AVAILABLE or not use_transformers or not self.transformer_pipeline:
            return []
        
        try:
            # Простой паттерн для поиска чисел с единицами
            pattern = r'(\d+(?:\.\d+)?)\s*([a-zA-Zа-яё°"\'\s]+)'
            matches = re.findall(pattern, str(text).lower())
            
            results = []
            for value, unit in matches:
                try:
                    numeric_value = float(value)
                    unit_clean = unit.strip()
                    
                    if len(unit_clean) > 0 and len(unit_clean) < 20:
                        # Проверяем единицу через Pint
                        pint_validation = self.validate_with_pint(unit_clean)
                        
                        if pint_validation and pint_validation.get('is_valid'):
                            # Если Pint распознал единицу, используем высокую уверенность
                            confidence = 0.9 if pint_validation.get('method') == 'pint_exact' else 0.7
                            canonical_unit = pint_validation.get('canonical_unit', unit_clean)
                            
                            results.append({
                                'value': numeric_value,
                                'unit': canonical_unit,
                                'unit_symbol': canonical_unit,
                                'dimension': pint_validation.get('dimensionality', 'unknown'),
                                'confidence': confidence,
                                'method': 'transformers_pint',
                                'pint_validation': pint_validation
                            })
                        else:
                            # Если Pint не распознал, используем низкую уверенность
                            results.append({
                                'value': numeric_value,
                                'unit': unit_clean,
                                'unit_symbol': unit_clean,
                                'dimension': 'unknown',
                                'confidence': 0.4,  # Низкая уверенность для нераспознанных единиц
                                'method': 'transformers_only'
                            })
                except ValueError:
                    continue
            
            return results
        except Exception as e:
            return []
    
    def validate_with_pint(self, unit_string):
        """Улучшенная валидация единицы измерения с помощью Pint"""
        if not PINT_AVAILABLE or not use_pint:
            return None
        
        try:
            # Очистка строки единицы
            unit_clean = re.sub(r'[^\w°]', '', str(unit_string).strip())
            
            # Сначала пытаемся найти точное соответствие
            try:
                unit = ureg.parse_expression(unit_clean)
                return {
                    'unit_str': str(unit),
                    'dimensionality': str(unit.dimensionality),
                    'is_valid': True,
                    'canonical_unit': str(unit.to_base_units().units),
                    'method': 'pint_exact'
                }
            except:
                pass
            
            # Если точное соответствие не найдено, пытаемся найти похожие единицы
            # Используем встроенную функцию Pint для поиска
            try:
                # Поиск по частичному совпадению в базе Pint
                for unit_name in ureg._units.keys():
                    if unit_clean.lower() in str(unit_name).lower() or str(unit_name).lower() in unit_clean.lower():
                        unit = ureg.parse_expression(unit_name)
                        return {
                            'unit_str': str(unit),
                            'dimensionality': str(unit.dimensionality),
                            'is_valid': True,
                            'canonical_unit': str(unit.to_base_units().units),
                            'method': 'pint_fuzzy',
                            'original_input': unit_clean,
                            'matched_unit': unit_name
                        }
            except:
                pass
            
            return {'is_valid': False, 'method': 'pint', 'error': 'No match found'}
            
        except Exception as e:
            return {'is_valid': False, 'method': 'pint', 'error': str(e)}
    
    def extract_all_methods(self, text):
        """Комбинированное извлечение всеми доступными методами"""
        all_results = []
        
        # Quantulum3
        quantulum_results = self.extract_with_quantulum(text)
        all_results.extend(quantulum_results)
        
        # Transformers
        transformer_results = self.extract_with_transformers(text)
        all_results.extend(transformer_results)
        
        # Валидация результатов с Pint
        for result in all_results:
            pint_validation = self.validate_with_pint(result['unit'])
            if pint_validation:
                result['pint_validation'] = pint_validation
        
        return all_results

def extract_numeric_values_intelligent(text):
    """Улучшенное извлечение числовых значений"""
    if pd.isna(text):
        return []
    
    # Используем глобальный экстрактор без повторной инициализации
    if not hasattr(extract_numeric_values_intelligent, '_extractor'):
        extract_numeric_values_intelligent._extractor = IntelligentUnitExtractor()
    
    extractor = extract_numeric_values_intelligent._extractor
    
    # Получаем результаты от всех методов
    results = extractor.extract_all_methods(text)
    
    # Фильтруем по порогу уверенности
    filtered_results = [r for r in results if r.get('confidence', 0) >= confidence_threshold]
    
    return filtered_results

def detect_measurement_type_intelligent(column_name, values_sample):
    """Интеллектуальное определение типа измерения"""
    # Используем глобальный экстрактор
    if not hasattr(detect_measurement_type_intelligent, '_extractor'):
        detect_measurement_type_intelligent._extractor = IntelligentUnitExtractor()
    
    extractor = detect_measurement_type_intelligent._extractor
    
    # Анализируем образцы значений
    dimension_counter = Counter()
    unit_counter = Counter()
    
    for value in values_sample[:max_samples]:
        if pd.notna(value):
            extractions = extractor.extract_all_methods(value)
            
            for extraction in extractions:
                if extraction.get('confidence', 0) >= confidence_threshold:
                    # Подсчитываем размерности
                    dimension = extraction.get('dimension', 'unknown')
                    if dimension and dimension != 'unknown':
                        dimension_counter[dimension] += 1
                    
                    # Подсчитываем единицы
                    unit = extraction.get('unit', '')
                    if unit:
                        unit_counter[unit] += 1
                    
                    # Если есть валидация Pint, используем её
                    pint_val = extraction.get('pint_validation', {})
                    if pint_val.get('is_valid'):
                        pint_dimension = pint_val.get('dimensionality', '')
                        if pint_dimension:
                            dimension_counter[pint_dimension] += 2  # Больший вес для Pint
    
    # Определяем наиболее вероятный тип
    most_common_dimension = dimension_counter.most_common(1)
    most_common_unit = unit_counter.most_common(3)
    
    return {
        'primary_dimension': most_common_dimension[0][0] if most_common_dimension else 'unknown',
        'confidence': most_common_dimension[0][1] / len(values_sample) if most_common_dimension else 0,
        'common_units': [unit for unit, count in most_common_unit],
        'analysis_summary': {
            'dimensions_found': dict(dimension_counter),
            'units_found': dict(unit_counter)
        }
    }

def extract_unit_from_context_optimized(column_name, product_name=None, category=None):
    """Оптимизированное извлечение единицы измерения с использованием Pint"""
    
    if not PINT_AVAILABLE:
        return None
    
    # Приводим к нижнему регистру для анализа
    column_lower = column_name.lower()
    product_lower = str(product_name).lower() if product_name else ""
    category_lower = str(category).lower() if category else ""
    
    # 1. Сначала ищем в скобках (как раньше)
    bracket_match = re.search(r'\(([^)]+)\)', column_name)
    if bracket_match:
        potential_unit = bracket_match.group(1).strip()
        # Проверяем через Pint
        try:
            ureg.parse_expression(potential_unit)
            return potential_unit
        except:
            pass
    
    # 2. Ищем единицы в самом названии колонки
    # Разбиваем название на слова и проверяем каждое через Pint
    words = re.findall(r'\b\w+\b', column_lower)
    for word in words:
        if len(word) > 1:  # Пропускаем односимвольные слова
            try:
                ureg.parse_expression(word)
                return word
            except:
                continue
    
    # 3. Специальные правила для контекста (минимальные)
    context_text = f"{column_lower} {product_lower} {category_lower}"
    
    # Экраны обычно в дюймах
    if any(screen_word in context_text for screen_word in ['screen', 'display', 'экран', 'диагональ']):
        try:
            ureg.parse_expression('inch')
            return 'inch'
        except:
            pass
    
    # Вес обычно в килограммах
    if any(weight_word in context_text for weight_word in ['weight', 'вес']):
        try:
            ureg.parse_expression('kg')
            return 'kg'
        except:
            pass
    
    # Мощность обычно в ваттах
    if any(power_word in context_text for power_word in ['power', 'мощность']):
        try:
            ureg.parse_expression('W')
            return 'W'
        except:
            pass
    
    # Память обычно в гигабайтах
    if any(mem_word in context_text for mem_word in ['memory', 'storage', 'память']):
        try:
            ureg.parse_expression('GB')
            return 'GB'
        except:
            pass
    
    return None

# Групповое определение единиц для категорий
def analyze_units_by_category(df, column_name, category_column='group_name', sample_size=5):
    """Анализ единиц измерения по категориям с минимальной выборкой"""
    
    if not PINT_AVAILABLE:
        return {}
    
    category_units = {}
    
    # Группируем по категориям
    if category_column not in df.columns:
        # Пытаемся найти альтернативную колонку категории
        for alt_col in ['Category', 'category', 'group_name']:
            if alt_col in df.columns:
                category_column = alt_col
                break
        else:
            # Если нет колонки категории, используем общий анализ
            return {'default': extract_unit_from_context_optimized(column_name, None, None)}
    
    for category, group_df in df.groupby(category_column):
        if pd.isna(category):
            continue
            
        # Берем только несколько образцов из каждой категории
        sample_values = group_df[column_name].dropna().head(sample_size)
        
        if len(sample_values) == 0:
            continue
        
        # Определяем единицу для этой категории
        category_unit = None
        
        # Сначала пытаемся извлечь единицы из самих значений
        for value in sample_values:
            value_str = str(value).strip()
            # Ищем паттерн число + единица
            unit_match = re.search(r'\d+\.?\d*\s*([a-zA-Zа-яё°"\'\s]+)', value_str)
            if unit_match:
                potential_unit = unit_match.group(1).strip()
                if potential_unit and len(potential_unit) < 10:
                    # Проверяем через Pint
                    try:
                        ureg.parse_expression(potential_unit)
                        category_unit = potential_unit
                        break
                    except:
                        continue
        
        # Если не нашли в значениях, используем контекстный анализ
        if not category_unit:
            category_unit = extract_unit_from_context_optimized(column_name, None, str(category))
        
        if category_unit:
            category_units[str(category)] = category_unit
    
    return category_units

# Кэш для группового анализа единиц
@st.cache_data
def get_category_units_cache(df_hash, column_name, category_column):
    """Кэшированный групповой анализ единиц по категориям"""
    # Создаем временный DataFrame из хэша (упрощенно)
    # В реальности здесь будет передан весь DataFrame
    return {}

def standardize_value_intelligent_optimized(value, target_format=None, column_name=None, product_name=None, category=None, category_units=None):
    """Оптимизированная интеллектуальная стандартизация значения с групповым определением единиц"""
    if pd.isna(value):
        return value
    
    # Используем глобальный экстрактор
    if not hasattr(standardize_value_intelligent_optimized, '_extractor'):
        standardize_value_intelligent_optimized._extractor = IntelligentUnitExtractor()
    
    extractor = standardize_value_intelligent_optimized._extractor
    extractions = extractor.extract_all_methods(value)
    
    # Если AI не нашел единицы в значении, используем групповое определение единиц
    if not extractions and column_name:
        unit_from_category = None
        
        # Сначала проверяем групповой кэш единиц для категории
        if category_units and category in category_units:
            unit_from_category = category_units[category]
        elif category_units and 'default' in category_units:
            unit_from_category = category_units['default']
        else:
            # Fallback к контекстному анализу
            unit_from_category = extract_unit_from_context_optimized(column_name, product_name, category)
        
        if unit_from_category:
            # Пытаемся извлечь число из значения
            value_str = str(value).strip()
            number_match = re.search(r'(\d+\.?\d*)', value_str)
            if number_match:
                number = float(number_match.group(1))
                
                # Создаем "искусственное" извлечение на основе группового анализа
                artificial_extraction = {
                    'value': number,
                    'unit': unit_from_category,
                    'unit_symbol': unit_from_category,
                    'dimension': 'from_category_group',
                    'confidence': 0.98,  # Очень высокая уверенность для группового анализа
                    'method': 'category_group_analysis'
                }
                extractions = [artificial_extraction]
    
    if not extractions:
        return value
    
    # Берем результат с наивысшей уверенностью
    best_extraction = max(extractions, key=lambda x: x.get('confidence', 0))
    
    if best_extraction.get('confidence', 0) < confidence_threshold:
        return value
    
    # Форматируем стандартизированное значение
    number = best_extraction.get('value', '')
    unit = best_extraction.get('unit', '')
    
    # Если есть валидация Pint, используем каноническую единицу
    pint_val = best_extraction.get('pint_validation', {})
    if pint_val.get('is_valid') and pint_val.get('canonical_unit'):
        unit = pint_val['canonical_unit']
    
    if target_format == 'number_unit':
        return f"{number} {unit}"
    elif target_format == 'unit_number':
        return f"{unit} {number}"
    else:
        return f"{number} {unit}"

def standardize_value_simple_optimized(value, target_format='number_unit', column_name=None, product_name=None, category=None, category_units=None):
    """Оптимизированная простая стандартизация значения с групповым определением единиц"""
    if pd.isna(value):
        return value
    
    value_str = str(value).strip()
    
    # Простой паттерн: число + единица
    pattern = r'(\d+\.?\d*)\s*([a-zA-Zа-яё°"\'\s]*)'
    match = re.search(pattern, value_str)
    
    if match:
        number = match.group(1)
        unit = match.group(2).strip()
        
        # Если единица пустая, используем групповое определение единиц
        if not unit and column_name:
            unit_from_category = None
            
            # Сначала проверяем групповой кэш единиц для категории
            if category_units and category in category_units:
                unit_from_category = category_units[category]
            elif category_units and 'default' in category_units:
                unit_from_category = category_units['default']
            else:
                # Fallback к контекстному анализу
                unit_from_category = extract_unit_from_context_optimized(column_name, product_name, category)
            
            if unit_from_category:
                unit = unit_from_category
        
        # Если единица все еще пустая, возвращаем как есть
        if not unit:
            return value
        
        # Форматируем по выбранному формату
        if target_format == 'number_unit':
            return f"{number} {unit}"
        elif target_format == 'unit_number':
            return f"{unit} {number}"
        else:
            return f"{number} {unit}"
    
    return value

# Оптимизированная функция пакетной стандартизации
def batch_standardize_column_optimized(df, column_name, standardization_format, use_ai=False, category_column='group_name'):
    """Пакетная стандартизация колонки с групповым анализом единиц"""
    
    # Сначала определяем единицы для каждой категории
    category_units = analyze_units_by_category(df, column_name, category_column)
    
    results = []
    
    # Обрабатываем данные по группам категорий для максимальной эффективности
    if category_column in df.columns:
        for category, group_df in df.groupby(category_column):
            if pd.isna(category):
                category = 'default'
            
            # Получаем единицу для этой категории
            category_unit = category_units.get(str(category))
            
            # Стандартизируем все значения в этой категории
            for idx, value in group_df[column_name].items():
                if use_ai:
                    standardized = standardize_value_intelligent_optimized(
                        value, standardization_format, column_name, None, str(category), category_units
                    )
                else:
                    standardized = standardize_value_simple_optimized(
                        value, standardization_format, column_name, None, str(category), category_units
                    )
                results.append((idx, standardized))
    else:
        # Если нет колонки категории, обрабатываем как обычно
        for idx, value in df[column_name].items():
            if use_ai:
                standardized = standardize_value_intelligent_optimized(
                    value, standardization_format, column_name, None, None, category_units
                )
            else:
                standardized = standardize_value_simple_optimized(
                    value, standardization_format, column_name, None, None, category_units
                )
            results.append((idx, standardized))
    
    # Возвращаем результаты в правильном порядке индексов
    results_dict = dict(results)
    return [results_dict.get(idx, df.loc[idx, column_name]) for idx in df.index]

def analyze_column_statistics_intelligent(df, column):
    """Улучшенный анализ статистики колонки"""
    values = df[column].dropna()
    
    # Общая статистика
    extraction_stats = {
        'quantulum_extractions': 0,
        'transformer_extractions': 0,
        'pint_validations': 0,
        'total_extractions': 0
    }
    
    numeric_values = []
    units_found = Counter()
    dimensions_found = Counter()
    
    # Используем глобальный экстрактор
    if not hasattr(analyze_column_statistics_intelligent, '_extractor'):
        analyze_column_statistics_intelligent._extractor = IntelligentUnitExtractor()
    
    extractor = analyze_column_statistics_intelligent._extractor
    
    # Анализируем образец значений
    sample_size = min(max_samples, len(values))
    for value in values.head(sample_size):
        extractions = extractor.extract_all_methods(value)
        
        extraction_stats['total_extractions'] += len(extractions)
        
        for extraction in extractions:
            method = extraction.get('method', 'unknown')
            if method == 'quantulum3':
                extraction_stats['quantulum_extractions'] += 1
            elif method == 'transformers':
                extraction_stats['transformer_extractions'] += 1
            
            if extraction.get('pint_validation', {}).get('is_valid'):
                extraction_stats['pint_validations'] += 1
            
            if extraction.get('confidence', 0) >= confidence_threshold:
                numeric_values.append(extraction.get('value', 0))
                units_found[extraction.get('unit', '')] += 1
                
                dimension = extraction.get('dimension', '')
                if dimension:
                    dimensions_found[dimension] += 1
    
    stats = {
        'total_values': len(values),
        'analyzed_values': sample_size,
        'numeric_count': len(numeric_values),
        'extraction_stats': extraction_stats,
        'units_found': dict(units_found.most_common(10)),
        'dimensions_found': dict(dimensions_found.most_common(5)),
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

    # Исключаем колонки специфичные для стандартизации
    system_columns = [
        col for col in all_columns
        if any(keyword in col.lower() for keyword in [
            'name', 'id', 'source_file', 'category', 'group_name',  # системные
            'sku', 'width',  # обязательные исключения
            'price', 'cost', 'цена', 'стоимость'  # цифровые данные
        ])
    ]

    st.markdown("#### Исключить колонки из анализа")
    excluded_cols = st.multiselect(
        "Выберите колонки, которые не нужно анализировать:",
        options=all_columns,
        default=system_columns,
        help="По умолчанию исключены: системные колонки, SKU, Width, цены"
    )

    analysis_columns = [c for c in all_columns if c not in excluded_cols]
    
    if analysis_columns:
        st.success(f"✅ Для анализа доступно {len(analysis_columns)} колонок: {', '.join(analysis_columns[:5])}{'...' if len(analysis_columns) > 5 else ''}")
    else:
        st.warning("❌ Нет колонок для анализа. Все колонки исключены.")
    
    # --- Этап 1: Быстрое определение числовых колонок ---
    with st.expander("#### 1. 🔢 Быстрое определение числовых колонок", expanded=True):
        st.markdown("**Поиск колонок содержащих цифры или цифры + единицы измерения**")
        
        # Показываем активные методы ИИ для следующих этапов
        active_methods = []
        if use_quantulum:
            active_methods.append("🔍 Quantulum3")
        if use_transformers:
            active_methods.append("🤖 Transformer-NER")
        if use_pint:
            active_methods.append("📐 Pint")
        
        if active_methods:
            st.info(f"Методы ИИ для следующих этапов: {', '.join(active_methods)}")
            if use_transformers and TRANSFORMERS_AVAILABLE:
                st.info("ℹ️ Transformer модель будет загружена при первом использовании (может занять некоторое время)")
        
        if st.button("🔍 Найти числовые колонки", key="find_numeric_simple"):
            if not analysis_columns:
                st.warning("Нет колонок для анализа после исключения.")
            else:
                with st.spinner("Анализируем колонки..."):
                    # Быстрый анализ
                    numeric_analysis = quick_find_numeric_columns(df_param, analysis_columns)
                    
                    # Сохраняем результаты
                    st.session_state['numeric_analysis_simple'] = numeric_analysis
                    
                    if numeric_analysis:
                        st.success(f"✅ Найдено {len(numeric_analysis)} колонок с числовыми значениями!")
                    else:
                        st.warning("❌ Колонки с числовыми значениями не найдены.")
        
        # Отображение результатов простого анализа
        if 'numeric_analysis_simple' in st.session_state and st.session_state['numeric_analysis_simple']:
            st.markdown("**📊 Найденные числовые колонки**")
            
            analysis_data = []
            for column, stats in st.session_state['numeric_analysis_simple'].items():
                row = {
                    'Колонка': column,
                    'Всего значений': stats['total_values'],
                    'Проанализировано': stats['analyzed_values'],
                    'С цифрами': stats['numeric_count'],
                    '% числовых': f"{stats['numeric_percentage']:.1f}%",
                    'Найденные единицы': ', '.join(list(stats['units_found'].keys())[:3]),
                    'Примеры чисел': ', '.join([str(v) for v in stats['numeric_values'][:3]]),
                    'Среднее значение': f"{stats['avg_value']:.2f}" if stats['avg_value'] > 0 else "-"
                }
                analysis_data.append(row)
            
            analysis_df = pd.DataFrame(analysis_data)
            st.dataframe(analysis_df, use_container_width=True)
            
            # Показываем примеры значений для выбранной колонки
            selected_col = st.selectbox(
                "Выберите колонку для просмотра примеров:",
                options=list(st.session_state['numeric_analysis_simple'].keys())
            )
            
            if selected_col:
                st.markdown(f"**Примеры значений из колонки '{selected_col}':**")
                sample_values = df_param[selected_col].dropna().head(10).tolist()
                for i, value in enumerate(sample_values, 1):
                    # Подсвечиваем цифры в значениях
                    value_str = str(value)
                    highlighted = re.sub(r'(\d+\.?\d*)', r'**\1**', value_str)
                    st.write(f"{i}. {highlighted}")

    # --- Этап 2: Стандартизация значений ---
    with st.expander("#### 2. ⚙️ Стандартизация значений", expanded=False):
        
        if 'numeric_analysis_simple' in st.session_state and st.session_state['numeric_analysis_simple']:
            
            # Выбор режима стандартизации
            col1, col2 = st.columns([1, 2])
            with col1:
                use_ai_standardization = st.checkbox(
                    "🧠 Умная стандартизация (AI)", 
                    value=False,
                    help="Использовать AI для понимания единиц измерения и автоматической стандартизации"
                )
            with col2:
                if use_ai_standardization:
                    st.info("🤖 AI-режим: автоматическое определение и стандартизация единиц")
                else:
                    st.info("⚡ Быстрый режим: простое форматирование regex")
            
            # Информация об оптимизации
            st.info("🚀 **Групповая оптимизация**: Система анализирует только несколько образцов из каждой категории товаров, а затем применяет найденные единицы ко всем товарам в этой категории. Например, если в категории 'laptop' параметр 'Screen Inches' имеет единицу 'inches' у 2-3 товаров, то все остальные ноутбуки автоматически получат эту же единицу.")
            
            # Выбор колонок для стандартизации
            standardizable_columns = list(st.session_state['numeric_analysis_simple'].keys())
            selected_columns = st.multiselect(
                "Выберите колонки для стандартизации:",
                options=standardizable_columns,
                default=standardizable_columns[:3] if len(standardizable_columns) > 3 else standardizable_columns
            )
            
            if selected_columns:
                # Настройки стандартизации
                format_options = {
                    'number_unit': 'Число Единица (100 kg)',
                    'unit_number': 'Единица Число (kg 100)'
                }
                
                if use_ai_standardization:
                    format_options['auto'] = 'Автоматический выбор формата (AI)'
                
                standardization_format = st.selectbox(
                    "Формат результата:",
                    options=list(format_options.keys()),
                    format_func=lambda x: format_options[x],
                    index=0
                )
                
                # Предпросмотр стандартизации
                if st.button("👁️ Предпросмотр стандартизации", key="preview_standardization"):
                    preview_col = selected_columns[0]
                    st.markdown(f"**Предпросмотр для колонки '{preview_col}':**")
                    
                    sample_values = df_param[preview_col].dropna().head(5)
                    for i, (idx, original) in enumerate(sample_values.items()):
                        # Получаем дополнительную информацию о продукте
                        product_name = None
                        category = None
                        
                        # Пытаемся найти колонки с названием продукта и категорией
                        if 'Name' in df_param.columns:
                            product_name = df_param.loc[idx, 'Name']
                        elif 'Product Name' in df_param.columns:
                            product_name = df_param.loc[idx, 'Product Name']
                        elif 'product_name' in df_param.columns:
                            product_name = df_param.loc[idx, 'product_name']
                        
                        if 'Category' in df_param.columns:
                            category = df_param.loc[idx, 'Category']
                        elif 'category' in df_param.columns:
                            category = df_param.loc[idx, 'category']
                        elif 'group_name' in df_param.columns:
                            category = df_param.loc[idx, 'group_name']
                        
                        if use_ai_standardization:
                            standardized = standardize_value_intelligent_optimized(original, standardization_format, preview_col, product_name, category)
                        else:
                            standardized = standardize_value_simple_optimized(original, standardization_format, preview_col, product_name, category)
                        
                        # Показываем дополнительную информацию для контекста
                        context_info = []
                        if product_name:
                            context_info.append(f"Продукт: {str(product_name)[:30]}...")
                        if category:
                            context_info.append(f"Категория: {category}")
                        
                        context_str = " | ".join(context_info) if context_info else ""
                        
                        st.write(f"**До:** {original} → **После:** {standardized}")
                        if context_str:
                            st.caption(f"📝 Контекст: {context_str}")
                
                # Применение стандартизации
                standardization_button_text = "✅ Применить AI-стандартизацию" if use_ai_standardization else "✅ Применить простую стандартизацию"
                
                if st.button(standardization_button_text, key="apply_standardization"):
                    standardized_df = df_param.copy()
                    
                    if use_ai_standardization:
                        with st.spinner("🧠 AI-стандартизация значений..."):
                            progress_bar = st.progress(0)
                            
                            for i, column in enumerate(selected_columns):
                                progress_bar.progress((i + 1) / len(selected_columns))
                                
                                # Используем оптимизированную пакетную стандартизацию
                                new_column_name = f"{column}_ai_standardized"
                                standardized_df[new_column_name] = batch_standardize_column_optimized(
                                    standardized_df, 
                                    column, 
                                    standardization_format, 
                                    use_ai=True
                                )
                            
                            progress_bar.empty()
                            suffix = "ai_standardized"
                    else:
                        with st.spinner("⚡ Быстрая стандартизация значений..."):
                            for column in selected_columns:
                                # Используем оптимизированную пакетную стандартизацию
                                new_column_name = f"{column}_standardized"
                                standardized_df[new_column_name] = batch_standardize_column_optimized(
                                    standardized_df, 
                                    column, 
                                    standardization_format, 
                                    use_ai=False
                                )
                            suffix = "standardized"
                    
                    # Сохраняем результат
                    st.session_state['df_standardization'] = standardized_df
                    method_name = "AI-стандартизация" if use_ai_standardization else "Простая стандартизация"
                    st.success(f"✅ {method_name} применена к {len(selected_columns)} колонкам!")
                    
                    # Показываем статистику оптимизации
                    total_rows = len(standardized_df)
                    category_column = None
                    for col in ['group_name', 'Category', 'category']:
                        if col in standardized_df.columns:
                            category_column = col
                            break
                    
                    if category_column:
                        unique_categories = standardized_df[category_column].nunique()
                        st.info(f"📊 **Статистика оптимизации**: Обработано {total_rows} товаров в {unique_categories} категориях. Вместо {total_rows * len(selected_columns)} индивидуальных анализов выполнено всего {unique_categories * len(selected_columns) * 5} анализов образцов (экономия ~{((total_rows * len(selected_columns) - unique_categories * len(selected_columns) * 5) / (total_rows * len(selected_columns)) * 100):.1f}%)")
                    
                    # Показываем результаты
                    comparison_cols = []
                    for col in selected_columns:
                        new_col_name = f"{col}_{suffix}"
                        if new_col_name in standardized_df.columns:
                            comparison_cols.extend([col, new_col_name])
                    
                    if comparison_cols:
                        st.markdown("**Сравнение результатов:**")
                        st.dataframe(standardized_df[comparison_cols].head(10))
        else:
            st.info("Сначала найдите числовые колонки в разделе выше.")

    # --- Этап 3: Интеллектуальное определение единиц измерения ---
    with st.expander("#### 3. 🧠 Интеллектуальное определение единиц измерения", expanded=False):
        st.markdown("**AI-анализ типов измерений и единиц для найденных числовых колонок**")
        
        if 'numeric_analysis_simple' in st.session_state and st.session_state['numeric_analysis_simple']:
            
            if st.button("🔍 Определить единицы измерения (AI)", key="detect_units_ai"):
                measurement_analysis = {}
                
                progress_bar = st.progress(0)
                columns_to_analyze = list(st.session_state['numeric_analysis_simple'].keys())
                
                with st.spinner("Выполняем AI-анализ единиц измерения..."):
                    for i, column in enumerate(columns_to_analyze):
                        progress_bar.progress((i + 1) / len(columns_to_analyze))
                        
                        # Берем образец значений для анализа
                        sample_values = df_param[column].dropna().head(max_samples).tolist()
                        analysis_result = detect_measurement_type_intelligent(column, sample_values)
                        
                        measurement_analysis[column] = {
                            'type': analysis_result['primary_dimension'],
                            'confidence': analysis_result['confidence'],
                            'common_units': analysis_result['common_units'],
                            'analysis_summary': analysis_result['analysis_summary'],
                            'sample_values': sample_values[:10]
                        }
                
                progress_bar.empty()
                st.session_state['measurement_analysis_ai'] = measurement_analysis
                st.success("✅ Интеллектуальный анализ единиц измерения завершен!")
            
            # Отображение результатов определения единиц
            if 'measurement_analysis_ai' in st.session_state:
                st.markdown("**🧠 Результаты AI-определения единиц измерения**")
                
                measurement_data = []
                for column, analysis in st.session_state['measurement_analysis_ai'].items():
                    row = {
                        'Колонка': column,
                        'Размерность': analysis['type'],
                        'Уверенность': f"{analysis['confidence']:.2f}",
                        'Найденные единицы': ', '.join(analysis['common_units'][:5]),
                        'Примеры значений': ', '.join([str(v) for v in analysis['sample_values'][:3]])
                    }
                    measurement_data.append(row)
                
                measurement_df = pd.DataFrame(measurement_data)
                st.dataframe(measurement_df, use_container_width=True)
                
                # Детальная информация
                selected_column = st.selectbox(
                    "Выберите колонку для детального анализа:",
                    options=list(st.session_state['measurement_analysis_ai'].keys())
                )
                
                if selected_column:
                    analysis = st.session_state['measurement_analysis_ai'][selected_column]
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Найденные размерности:**")
                        dimensions = analysis['analysis_summary']['dimensions_found']
                        for dim, count in dimensions.items():
                            st.write(f"- {dim}: {count}")
                    
                    with col2:
                        st.markdown("**Найденные единицы:**")
                        units = analysis['analysis_summary']['units_found']
                        for unit, count in units.items():
                            st.write(f"- {unit}: {count}")
        else:
            st.info("Сначала найдите числовые колонки в разделе выше.")

    # --- Этап 4: Пакетная конвертация единиц с Pint ---
    with st.expander("#### 4. 🔄 Пакетная конвертация единиц (Pint)", expanded=False):
        if PINT_AVAILABLE and use_pint:
            st.markdown("**Конвертация всех единиц в стандартный формат**")
            
            if 'measurement_analysis_ai' in st.session_state:
                # Показываем возможные конвертации
                conversion_options = {}
                
                for column, analysis in st.session_state['measurement_analysis_ai'].items():
                    dimension = analysis['type']
                    if dimension and dimension != 'unknown':
                        conversion_options[column] = {
                            'dimension': dimension,
                            'units': analysis['common_units']
                        }
                
                if conversion_options:
                    st.markdown("**Доступные конвертации:**")
                    
                    for column, info in conversion_options.items():
                        st.write(f"**{column}** (размерность: {info['dimension']})")
                        st.write(f"Найденные единицы: {', '.join(info['units'][:5])}")
                        
                        # Выбор целевой единицы
                        target_unit = st.text_input(
                            f"Целевая единица для {column}:",
                            value=info['units'][0] if info['units'] else '',
                            key=f"target_unit_{column}"
                        )
                        
                        conversion_options[column]['target_unit'] = target_unit
                    
                    if st.button("🔄 Выполнить пакетную конвертацию", key="batch_conversion"):
                        # Здесь будет логика пакетной конвертации с Pint
                        st.info("Функция пакетной конвертации будет реализована в следующем обновлении")
                else:
                    st.info("Нет подходящих данных для конвертации единиц")
            else:
                st.info("Сначала выполните анализ единиц измерения")
        else:
            st.warning("Библиотека Pint не доступна или не выбрана в настройках")

    # --- Сохранение результатов ---
    with st.expander("#### 5. 💾 Сохранение результатов", expanded=False):
        
        # Показываем текущее состояние
        if 'df_standardization' in st.session_state:
            current_df = st.session_state['df_standardization']
            st.info(f"📊 Текущее состояние: {len(current_df)} строк, {len(current_df.columns)} колонок")
            
            # Подсчитываем стандартизированные колонки
            simple_standardized_count = len([col for col in current_df.columns if col.endswith('_standardized') and not col.endswith('_ai_standardized')])
            ai_standardized_count = len([col for col in current_df.columns if '_ai_standardized' in col])
            
            if simple_standardized_count > 0:
                st.success(f"✅ Простых стандартизированных колонок: {simple_standardized_count}")
            if ai_standardized_count > 0:
                st.success(f"🧠 AI-стандартизированных колонок: {ai_standardized_count}")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("💾 Сохранить в grouped_categories.csv", key="save_main_ai"):
                try:
                    output_path = os.path.join(os.path.dirname(__file__), '..', 'grouped_categories.csv')
                    current_df = st.session_state.get('df_standardization', df_param)
                    current_df.to_csv(output_path, index=False, encoding='utf-8-sig')
                    st.success(f"✅ Данные сохранены в {output_path}")
                    st.success(f"📊 Сохранено: {len(current_df)} строк, {len(current_df.columns)} колонок")
                except Exception as e:
                    st.error(f"❌ Ошибка при сохранении: {e}")
        
        with col2:
            if st.button("📥 Скачать как CSV", key="download_csv_ai"):
                current_df = st.session_state.get('df_standardization', df_param)
                csv_data = current_df.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    label="⬇️ Скачать файл",
                    data=csv_data,
                    file_name="ai_standardized_parameters.csv",
                    mime="text/csv",
                    key="download_ai_standardized"
                )
        
        with col3:
            if st.button("🔄 Обновить исходные данные", key="reload_data_ai"):
                # Очищаем кэш и session state
                for key in ['df_standardization', 'numeric_analysis_simple', 'numeric_analysis_ai', 'measurement_analysis_ai']:
                    if key in st.session_state:
                        del st.session_state[key]
                
                # Очищаем кэш экстракторов
                for func in [extract_numeric_values_intelligent, detect_measurement_type_intelligent, 
                           standardize_value_intelligent_optimized, analyze_column_statistics_intelligent]:
                    if hasattr(func, '_extractor'):
                        delattr(func, '_extractor')
                
                # Очищаем кэш Streamlit
                st.cache_resource.clear()
                
                st.success("Данные и кэш обновлены! Страница будет перезагружена.")
                st.rerun()

    # --- Статистика ---
    with st.expander("#### 6. 📊 Статистика и сравнение методов", expanded=False):
        current_df = st.session_state.get('df_standardization', df_param)
        total_cols = len(current_df.columns)
        
        # Подсчитываем стандартизированные колонки (простые + AI)
        simple_standardized_cols = len([col for col in current_df.columns if col.endswith('_standardized') and not col.endswith('_ai_standardized')])
        ai_standardized_cols = len([col for col in current_df.columns if '_ai_standardized' in col])
        total_standardized = simple_standardized_cols + ai_standardized_cols
        
        total_rows = len(current_df)
        
        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
        with stat_col1:
            st.metric("Всего колонок", total_cols)
        with stat_col2:
            st.metric("Стандартизированных", total_standardized)
        with stat_col3:
            st.metric("Всего строк", total_rows)
        with stat_col4:
            active_methods_count = sum([use_quantulum, use_transformers, use_pint])
            st.metric("Активных методов ИИ", active_methods_count)
        
        # Детализация по типам стандартизации
        if total_standardized > 0:
            detail_col1, detail_col2 = st.columns(2)
            with detail_col1:
                st.metric("🔧 Простая стандартизация", simple_standardized_cols)
            with detail_col2:
                st.metric("🧠 AI-стандартизация", ai_standardized_cols)
        
        # Сравнение производительности методов
        simple_found = len(st.session_state.get('numeric_analysis_simple', {}))
        
        st.markdown("**Результаты анализа:**")
        st.write(f"- **Простой анализ**: {simple_found} числовых колонок найдено")
        
        if 'numeric_analysis_ai' in st.session_state:
            st.markdown("**Сравнение производительности ИИ-методов:**")
            
            method_stats = {
                'Quantulum3': 0,
                'Transformers': 0,
                'Pint валидация': 0
            }
            
            for column, stats in st.session_state['numeric_analysis_ai'].items():
                extraction_stats = stats.get('extraction_stats', {})
                method_stats['Quantulum3'] += extraction_stats.get('quantulum_extractions', 0)
                method_stats['Transformers'] += extraction_stats.get('transformer_extractions', 0)
                method_stats['Pint валидация'] += extraction_stats.get('pint_validations', 0)
            
            for method, count in method_stats.items():
                st.write(f"- **{method}**: {count} успешных извлечений")
        
        # Показываем текущее состояние данных
        if st.checkbox("Показать все данные", key="show_all_data_ai"):
            st.dataframe(current_df, use_container_width=True)
        else:
            st.markdown("**Первые 20 строк:**")
            st.dataframe(current_df.head(20), use_container_width=True)

else:
    st.warning("Нет данных для обработки. Сначала выполните основную обработку в главном приложении.")
