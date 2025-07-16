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
                    # Новый фильтр: игнорируем 'x', '-', ':', '' и неалфавитные "единицы"
                    if (
                        unit_clean
                        and len(unit_clean) < 15
                        and unit_clean.lower() not in ['x', '-', ':', '']
                        and re.search(r'[a-zA-Zа-яё]', unit_clean)
                    ):
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
    def __init__(self, use_quantulum=True, use_transformers=True, use_pint=True):
        self._use_quantulum = use_quantulum
        self._use_transformers = use_transformers
        self._use_pint = use_pint
        self.transformer_pipeline = get_transformer_pipeline() if self._use_transformers else None

    def extract_with_quantulum(self, text):
        if not QUANTULUM_AVAILABLE or not self._use_quantulum:
            return []
        try:
            parsed = quantulum3.parser.parse(str(text))
            results = []
            for quantity in parsed:
                if quantity.value and quantity.unit:
                    unit_name = quantity.unit.name
                    unit_symbol = getattr(quantity.unit, 'symbol', unit_name)
                    pint_validation = self.validate_with_pint(unit_symbol)
                    if pint_validation and pint_validation.get('is_valid'):
                        canonical_unit = pint_validation.get('canonical_unit', unit_symbol)
                        dimension = pint_validation.get('dimensionality', quantity.unit.dimension.name if quantity.unit.dimension else '')
                        confidence = 0.95
                    else:
                        canonical_unit = unit_symbol
                        dimension = quantity.unit.dimension.name if quantity.unit.dimension else ''
                        confidence = 0.8
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
        except Exception:
            return []

    def extract_with_transformers(self, text):
        if not TRANSFORMERS_AVAILABLE or not self._use_transformers or not self.transformer_pipeline:
            return []
        try:
            pattern = r"(\d+(?:\.\d+)?)\s*([a-zA-Zа-яё°'\"\s]+)"
            matches = re.findall(pattern, str(text).lower())
            results = []
            for value, unit in matches:
                try:
                    numeric_value = float(value)
                    unit_clean = unit.strip()
                    if len(unit_clean) > 0 and len(unit_clean) < 20:
                        pint_validation = self.validate_with_pint(unit_clean)
                        if pint_validation and pint_validation.get('is_valid'):
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
                            results.append({
                                'value': numeric_value,
                                'unit': unit_clean,
                                'unit_symbol': unit_clean,
                                'dimension': 'unknown',
                                'confidence': 0.4,
                                'method': 'transformers_only'
                            })
                except ValueError:
                    continue
            return results
        except Exception:
            return []

    def validate_with_pint(self, unit_string):
        if not PINT_AVAILABLE or not self._use_pint:
            return None
        try:
            unit_clean = re.sub(r'[^\w°]', '', str(unit_string).strip())
            try:
                unit = ureg.parse_expression(unit_clean)
                
                # ИСПРАВЛЕНИЕ: Получаем человекочитаемое название размерности
                dimensionality_str = str(unit.dimensionality)
                
                # Улучшаем базовые размерности для читаемости
                if dimensionality_str == '[mass] * [length] ** 2 / [time] ** 3':
                    dimension_name = 'power'
                elif dimensionality_str == '[mass] * [length] ** 2 / [time] ** 2':
                    dimension_name = 'energy'
                elif dimensionality_str == '[length]':
                    dimension_name = 'length'
                elif dimensionality_str == '[mass]':
                    dimension_name = 'mass'
                elif dimensionality_str == '[time]':
                    dimension_name = 'time'
                elif dimensionality_str == '' or dimensionality_str == '1':
                    dimension_name = 'dimensionless'
                else:
                    dimension_name = dimensionality_str
                
                return {
                    'unit_str': str(unit),
                    'dimensionality': dimension_name,  # Человекочитаемое название
                    'is_valid': True,
                    'canonical_unit': str(unit.units),  # БЕЗ to_base_units()!
                    'method': 'pint_exact'
                }
            except:
                pass
            try:
                for unit_name in ureg._units.keys():
                    if unit_clean.lower() in str(unit_name).lower() or str(unit_name).lower() in unit_clean.lower():
                        unit = ureg.parse_expression(unit_name)
                        
                        # Та же логика для fuzzy поиска
                        dimensionality_str = str(unit.dimensionality)
                        if dimensionality_str == '[mass] * [length] ** 2 / [time] ** 3':
                            dimension_name = 'power'
                        elif dimensionality_str == '[mass] * [length] ** 2 / [time] ** 2':
                            dimension_name = 'energy'
                        elif dimensionality_str == '[length]':
                            dimension_name = 'length'
                        elif dimensionality_str == '[mass]':
                            dimension_name = 'mass'
                        elif dimensionality_str == '[time]':
                            dimension_name = 'time'
                        elif dimensionality_str == '' or dimensionality_str == '1':
                            dimension_name = 'dimensionless'
                        else:
                            dimension_name = dimensionality_str
                        
                        return {
                            'unit_str': str(unit),
                            'dimensionality': dimension_name,  # Человекочитаемое название
                            'is_valid': True,
                            'canonical_unit': str(unit.units),  # БЕЗ to_base_units()!
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
        all_results = []
        all_results.extend(self.extract_with_quantulum(text))
        all_results.extend(self.extract_with_transformers(text))
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

def detect_measurement_type_intelligent(column_name, values_sample, use_quantulum=None, use_transformers=None, use_pint=None):
    """Интеллектуальное определение типа измерения с учетом контекста колонки
    use_quantulum/use_transformers/use_pint: если None — использовать глобальные переменные, если True/False — форсировать режим"""
    # Определяем какие методы использовать
    uq = use_quantulum if use_quantulum is not None else globals().get('use_quantulum', True)
    ut = use_transformers if use_transformers is not None else globals().get('use_transformers', True)
    up = use_pint if use_pint is not None else globals().get('use_pint', True)
    extractor = IntelligentUnitExtractor(use_quantulum=uq, use_transformers=ut, use_pint=up)
    dimension_counter = Counter()
    unit_counter = Counter()
    context_unit = extract_unit_from_context_optimized(column_name, None, None)
    if context_unit:
        pint_validation = extractor.validate_with_pint(context_unit)
        if pint_validation and pint_validation.get('is_valid'):
            dimension = pint_validation.get('dimensionality', 'unknown')
            canonical_unit = pint_validation.get('canonical_unit', context_unit)
            dimension_counter[dimension] = len(values_sample)
            unit_counter[canonical_unit] = len(values_sample)
            numeric_count = 0
            for value in values_sample[:max_samples]:
                if pd.notna(value):
                    value_str = str(value).strip()
                    if re.search(r'\d+\.?\d*', value_str):
                        numeric_count += 1
            if numeric_count / len(values_sample) > 0.5:
                return {
                    'primary_dimension': dimension,
                    'confidence': 0.95,  # ПОВЫШЕННАЯ уверенность для контекста
                    'common_units': [canonical_unit],
                    'analysis_summary': {
                        'dimensions_found': {dimension: len(values_sample)},
                        'units_found': {canonical_unit: len(values_sample)},
                        'context_unit': context_unit,
                        'method': 'context_analysis_priority'  # Указываем приоритетный метод
                    }
                }
    for value in values_sample[:max_samples]:
        if pd.notna(value):
            extractions = extractor.extract_all_methods(value)
            for extraction in extractions:
                if extraction.get('confidence', 0) >= confidence_threshold:
                    dimension = extraction.get('dimension', 'unknown')
                    if dimension and dimension != 'unknown':
                        dimension_counter[dimension] += 1
                    unit = extraction.get('unit', '')
                    if unit:
                        unit_counter[unit] += 1
                    pint_val = extraction.get('pint_validation', {})
                    if pint_val.get('is_valid'):
                        pint_dimension = pint_val.get('dimensionality', '')
                        if pint_dimension:
                            dimension_counter[pint_dimension] += 2
    most_common_dimension = dimension_counter.most_common(1)
    most_common_unit = unit_counter.most_common(3)
    return {
        'primary_dimension': most_common_dimension[0][0] if most_common_dimension else 'unknown',
        'confidence': most_common_dimension[0][1] / len(values_sample) if most_common_dimension else 0,
        'common_units': [unit for unit, count in most_common_unit],
        'analysis_summary': {
            'dimensions_found': dict(dimension_counter),
            'units_found': dict(unit_counter),
            'context_unit': context_unit,
            'method': 'mixed_analysis'
        }
    }

def analyze_product_context_for_units(column_name, product_name=None, category=None, description=None):
    """AI-анализ контекста продукта для определения логичных единиц измерения"""
    
    if not PINT_AVAILABLE:
        return None
    
    # Приводим к нижнему регистру для анализа
    column_lower = column_name.lower()
    product_lower = str(product_name).lower() if product_name else ""
    category_lower = str(category).lower() if category else ""
    description_lower = str(description).lower() if description else ""
    
    # Объединяем весь контекст
    full_context = f"{column_lower} {product_lower} {category_lower} {description_lower}"
    
    # === ПРАВИЛА ДЛЯ РАЗНЫХ ТИПОВ ПРОДУКТОВ ===
    
    # 1. НОУТБУКИ / КОМПЬЮТЕРЫ
    if any(tech_word in full_context for tech_word in [
        'laptop', 'notebook', 'computer', 'pc', 'ноутбук', 'компьютер'
    ]):
        # Размеры ноутбуков обычно в мм
        if any(dim_word in column_lower for dim_word in ['dimension', 'size', 'размер']):
            return 'mm'
            
        # Экран ноутбука в дюймах
        if any(screen_word in column_lower for screen_word in ['screen', 'display', 'экран']):
            if 'resolution' not in column_lower:  # Не разрешение
                return 'inch'
                
        # Разрешение экрана в пикселях
        if any(res_word in column_lower for res_word in ['resolution', 'разрешение']):
            return 'pixel'
            
        # Вес ноутбука в кг
        if any(weight_word in column_lower for weight_word in ['weight', 'вес']):
            return 'kg'
            
        # Память в GB
        if any(mem_word in column_lower for mem_word in ['ram', 'memory', 'storage', 'память']):
            return 'GB'
    
    # 2. ТЕЛЕФОНЫ / СМАРТФОНЫ
    if any(phone_word in full_context for phone_word in [
        'phone', 'smartphone', 'mobile', 'телефон', 'смартфон'
    ]):
        # Размеры телефонов обычно в мм
        if any(dim_word in column_lower for dim_word in ['dimension', 'size', 'width', 'height', 'length']):
            return 'mm'
            
        # Экран телефона в дюймах
        if any(screen_word in column_lower for screen_word in ['screen', 'display']):
            if 'resolution' not in column_lower:
                return 'inch'
                
        # Разрешение в пикселях
        if 'resolution' in column_lower:
            return 'pixel'
            
        # Вес телефона в граммах (обычно легкие)
        if 'weight' in column_lower:
            return 'g'
    
    # 3. ТЕЛЕВИЗОРЫ / МОНИТОРЫ
    if any(tv_word in full_context for tv_word in [
        'tv', 'television', 'monitor', 'телевизор', 'монитор'
    ]):
        # Экран ТВ/монитора в дюймах
        if any(screen_word in column_lower for screen_word in ['screen', 'display', 'size']):
            if 'resolution' not in column_lower:
                return 'inch'
                
        # Разрешение в пикселях
        if 'resolution' in column_lower:
            return 'pixel'
            
        # Размеры ТВ в мм или см
        if any(dim_word in column_lower for dim_word in ['dimension', 'width', 'height']):
            return 'mm'
    
    # 4. БЫТОВАЯ ТЕХНИКА
    if any(appliance_word in full_context for appliance_word in [
        'refrigerator', 'washing', 'dishwasher', 'oven', 'холодильник', 'стиральная', 'посудомойка'
    ]):
        # Размеры крупной техники в см
        if any(dim_word in column_lower for dim_word in ['dimension', 'size', 'width', 'height']):
            return 'cm'
            
        # Вес крупной техники в кг
        if 'weight' in column_lower:
            return 'kg'
            
        # Мощность в ваттах
        if any(power_word in column_lower for power_word in ['power', 'мощность']):
            return 'W'
    
    # 5. АВТОМОБИЛИ
    if any(car_word in full_context for car_word in [
        'car', 'auto', 'vehicle', 'машина', 'автомобиль'
    ]):
        # Размеры авто в метрах
        if any(dim_word in column_lower for dim_word in ['length', 'width', 'height']):
            return 'm'
            
        # Мощность двигателя в лошадиных силах или кВт
        if any(power_word in column_lower for power_word in ['power', 'engine', 'мощность']):
            return 'kW'
    
    # === ОБЩИЕ ПРАВИЛА ПО ТИПУ ПАРАМЕТРА ===
    
    # Разрешение экрана всегда в пикселях
    if any(res_word in column_lower for res_word in ['resolution', 'разрешение']) or \
       re.search(r'\d+x\d+', str(column_name)):  # Паттерн 1920x1080
        return 'pixel'
    
    # Экраны обычно в дюймах
    if any(screen_word in column_lower for screen_word in ['screen', 'display', 'inch', 'дюйм']):
        return 'inch'
    
    # Вес
    if any(weight_word in column_lower for weight_word in ['weight', 'вес']):
        # Определяем по контексту: легкие устройства в граммах, тяжелые в кг
        if any(light_device in full_context for light_device in ['phone', 'tablet', 'телефон']):
            return 'g'
        else:
            return 'kg'
    
    # Мощность
    if any(power_word in column_lower for power_word in ['power', 'watt', 'мощность']):
        return 'W'
    
    # Память/хранилище
    if any(mem_word in column_lower for mem_word in ['memory', 'storage', 'ram', 'ssd', 'память']):
        return 'GB'
    
    # Размеры - определяем по контексту устройства
    if any(dim_word in column_lower for dim_word in ['dimension', 'size', 'width', 'height', 'length']):
        # Мелкие устройства в мм
        if any(small_device in full_context for small_device in [
            'phone', 'tablet', 'laptop', 'телефон', 'планшет', 'ноутбук'
        ]):
            return 'mm'
        # Средние устройства в см
        elif any(medium_device in full_context for medium_device in [
            'tv', 'monitor', 'телевизор', 'монитор'
        ]):
            return 'cm'
        # Крупные объекты в метрах
        else:
            return 'm'
    
    return None

def extract_unit_from_context_optimized(column_name, product_name=None, category=None, description=None):
    """Улучшенное извлечение единицы измерения с AI-анализом контекста продукта"""
    
    if not PINT_AVAILABLE:
        return None
    
    # Приводим к нижнему регистру для анализа
    column_lower = column_name.lower()
    
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
    words = re.findall(r'\b\w+\b', column_lower)
    for word in words:
        if len(word) > 1:  # Пропускаем односимвольные слова
            try:
                ureg.parse_expression(word)
                return word
            except:
                continue
    
    # 3. НОВЫЙ: AI-анализ контекста продукта
    ai_unit = analyze_product_context_for_units(column_name, product_name, category, description)
    if ai_unit:
        try:
            ureg.parse_expression(ai_unit)
            return ai_unit
        except:
            pass
    
    # 4. Базовые правила (fallback)
    context_text = f"{column_lower} {str(product_name).lower() if product_name else ''} {str(category).lower() if category else ''}"
    
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
def analyze_units_by_category(df, column_name, category_column='group_name', sample_size=3):
    """Улучшенный анализ единиц измерения по категориям с AI-контекстом продукта"""
    
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
            # Если нет колонки категории, используем общий AI-анализ
            sample_row = df.iloc[0] if len(df) > 0 else {}
            product_name = sample_row.get('product_name', sample_row.get('name', ''))
            description = sample_row.get('description', '')
            ai_unit = analyze_product_context_for_units(column_name, product_name, None, description)
            return {'default': ai_unit or extract_unit_from_context_optimized(column_name, product_name, None, description)}
    
    for category, group_df in df.groupby(category_column):
        if pd.isna(category):
            continue
            
        # Берем образцы товаров из каждой категории для AI-анализа
        sample_rows = group_df.head(sample_size)
        
        if len(sample_rows) == 0:
            continue
        
        # === НОВЫЙ: AI-АНАЛИЗ 2-3 ТОВАРОВ ИЗ КАТЕГОРИИ ===
        ai_suggestions = []
        
        for idx, row in sample_rows.iterrows():
            product_name = row.get('product_name', row.get('name', ''))
            description = row.get('description', '')
            
            # AI-анализ контекста каждого товара
            ai_unit = analyze_product_context_for_units(
                column_name, product_name, str(category), description
            )
            
            if ai_unit:
                ai_suggestions.append(ai_unit)
        
        # Выбираем наиболее частую AI-рекомендацию
        category_unit = None
        if ai_suggestions:
            unit_counter = Counter(ai_suggestions)
            most_common_ai_unit = unit_counter.most_common(1)
            if most_common_ai_unit:
                category_unit = most_common_ai_unit[0][0]
        
        # === FALLBACK: АНАЛИЗ ЗНАЧЕНИЙ В КОЛОНКЕ ===
        if not category_unit:
            # Анализируем сами значения в колонке
            sample_values = group_df[column_name].dropna().head(sample_size)
            
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
        
        # === ПОСЛЕДНИЙ FALLBACK: КОНТЕКСТНЫЙ АНАЛИЗ ===
        if not category_unit:
            # Используем общий контекстный анализ с информацией о товаре
            sample_row = sample_rows.iloc[0]
            product_name = sample_row.get('product_name', sample_row.get('name', ''))
            description = sample_row.get('description', '')
            category_unit = extract_unit_from_context_optimized(column_name, product_name, str(category), description)
        
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

def standardize_value_intelligent_optimized(value, target_format=None, column_name=None, product_name=None, category=None, category_units=None, description=None):
    """Оптимизированная интеллектуальная стандартизация с AI-контекстом продукта"""
    if pd.isna(value):
        return value
    
    # Используем глобальный экстрактор
    if not hasattr(standardize_value_intelligent_optimized, '_extractor'):
        standardize_value_intelligent_optimized._extractor = IntelligentUnitExtractor()
    
    extractor = standardize_value_intelligent_optimized._extractor
    extractions = extractor.extract_all_methods(value)
    
    # Если AI не нашел единицы в значении, используем контекстный анализ
    if not extractions and column_name:
        unit_from_context = None
        
        # Сначала проверяем групповой кэш единиц для категории
        if category_units and category in category_units:
            unit_from_context = category_units[category]
        elif category_units and 'default' in category_units:
            unit_from_context = category_units['default']
        else:
            # AI-контекстный анализ с полной информацией о продукте
            unit_from_context = analyze_product_context_for_units(column_name, product_name, category, description)
            if not unit_from_context:
                unit_from_context = extract_unit_from_context_optimized(column_name, product_name, category, description)
        
        if unit_from_context:
            # Пытаемся извлечь число из значения
            value_str = str(value).strip()
            number_match = re.search(r'(\d+\.?\d*)', value_str)
            if number_match:
                number = float(number_match.group(1))
                
                # Создаем "искусственное" извлечение на основе AI-контекстного анализа
                artificial_extraction = {
                    'value': number,
                    'unit': unit_from_context,
                    'unit_symbol': unit_from_context,
                    'dimension': 'ai_product_context',
                    'confidence': 0.95,  # Высокая уверенность для AI-контекстного анализа
                    'method': 'ai_product_context_analysis'
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

def standardize_value_simple_optimized(value, target_format='number_unit', column_name=None, product_name=None, category=None, category_units=None, description=None):
    """Оптимизированная простая стандартизация с AI-контекстом продукта"""
    if pd.isna(value):
        return value
    
    value_str = str(value).strip()
    
    # Простой паттерн: число + единица
    pattern = r'(\d+\.?\d*)\s*([a-zA-Zа-яё°"\'\s]*)'
    match = re.search(pattern, value_str)
    
    if match:
        number = match.group(1)
        unit = match.group(2).strip()
        
        # Если единица пустая, используем AI-контекстный анализ
        if not unit and column_name:
            unit_from_context = None
            
            # Сначала проверяем групповой кэш единиц для категории
            if category_units and category in category_units:
                unit_from_context = category_units[category]
            elif category_units and 'default' in category_units:
                unit_from_context = category_units['default']
            else:
                # AI-контекстный анализ с полной информацией о продукте
                unit_from_context = analyze_product_context_for_units(column_name, product_name, category, description)
                if not unit_from_context:
                    unit_from_context = extract_unit_from_context_optimized(column_name, product_name, category, description)
            
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

# Оптимизированная функция пакетной стандартизации
def batch_standardize_column_optimized(df, column_name, standardization_format, use_ai=False, category_column='group_name'):
    """Пакетная стандартизация колонки с улучшенным AI-контекстным анализом"""
    
    # Сначала определяем единицы для каждой категории с AI-анализом продуктов
    category_units = analyze_units_by_category(df, column_name, category_column, sample_size=3)
    
    results = []
    
    # Обрабатываем данные по группам категорий для максимальной эффективности
    if category_column in df.columns:
        for category, group_df in df.groupby(category_column):
            if pd.isna(category):
                category = 'default'
            
            # Стандартизируем все значения в этой категории
            for idx, row in group_df.iterrows():
                value = row[column_name]
                product_name = row.get('product_name', row.get('name', ''))
                description = row.get('description', '')
                
                if use_ai:
                    standardized = standardize_value_intelligent_optimized(
                        value, standardization_format, column_name, product_name, str(category), category_units, description
                    )
                else:
                    standardized = standardize_value_simple_optimized(
                        value, standardization_format, column_name, product_name, str(category), category_units, description
                    )
                results.append((idx, standardized))
    else:
        # Если нет колонки категории, обрабатываем как обычно
        for idx, row in df.iterrows():
            value = row[column_name]
            product_name = row.get('product_name', row.get('name', ''))
            description = row.get('description', '')
            
            if use_ai:
                standardized = standardize_value_intelligent_optimized(
                    value, standardization_format, column_name, product_name, None, category_units, description
                )
            else:
                standardized = standardize_value_simple_optimized(
                    value, standardization_format, column_name, product_name, None, category_units, description
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
            st.success("🧠 **НОВОЕ**: AI анализирует 2-3 товара из каждой категории для точного определения единиц измерения")
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
                
                # Проверяем где найти колонку - в исходных данных или в обработанных
                if selected_col in df_param.columns:
                    # Колонка еще не была стандартизирована
                    sample_values = df_param[selected_col].dropna().head(10).tolist()
                elif 'df_standardization' in st.session_state:
                    # Ищем стандартизированную версию колонки
                    standardized_col = None
                    for col in st.session_state['df_standardization'].columns:
                        if selected_col in col and "(" in col and ")" in col:
                            standardized_col = col
                            break
                    
                    if standardized_col:
                        sample_values = st.session_state['df_standardization'][standardized_col].dropna().head(10).tolist()
                        st.info(f"Показаны значения из стандартизированной колонки: {standardized_col}")
                    else:
                        # Пытаемся найти в обработанных данных
                        if selected_col in st.session_state['df_standardization'].columns:
                            sample_values = st.session_state['df_standardization'][selected_col].dropna().head(10).tolist()
                        else:
                            st.warning("Колонка не найдена ни в исходных, ни в обработанных данных")
                            sample_values = []
                else:
                    st.warning("Данные не найдены")
                    sample_values = []
                
                if sample_values:
                    for i, value in enumerate(sample_values, 1):
                        # Подсвечиваем цифры в значениях
                        value_str = str(value)
                        highlighted = re.sub(r'(\d+\.?\d*)', r'**\1**', value_str)
                        st.write(f"{i}. {highlighted}")

    # --- Этап 2: Интеллектуальное определение единиц измерения ---
    with st.expander("#### 2. 🧠 Интеллектуальное определение единиц измерения", expanded=False):
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
                        analysis_result = detect_measurement_type_intelligent(column, sample_values, use_quantulum=True, use_transformers=True, use_pint=True)
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
                        'Примеры значений': ', '.join([str(v) for v in analysis['sample_values'][:3]]),
                        'Метод анализа': analysis['analysis_summary'].get('method', 'unknown'),
                        'Контекстная единица': analysis['analysis_summary'].get('context_unit', 'Нет')
                    }
                    measurement_data.append(row)
                
                measurement_df = pd.DataFrame(measurement_data)
                st.dataframe(measurement_df, use_container_width=True)
                
                # Показываем статистику успешности
                successful_analyses = sum(1 for analysis in st.session_state['measurement_analysis_ai'].values() if analysis['confidence'] > 0)
                total_analyses = len(st.session_state['measurement_analysis_ai'])
                success_rate = (successful_analyses / total_analyses * 100) if total_analyses > 0 else 0
                
                st.info(f"📈 **Статистика анализа**: {successful_analyses} из {total_analyses} колонок успешно проанализированы ({success_rate:.1f}% успешности)")
                
                # Детальная информация
                selected_column = st.selectbox(
                    "Выберите колонку для детального анализа:",
                    options=list(st.session_state['measurement_analysis_ai'].keys())
                )
                
                if selected_column:
                    analysis = st.session_state['measurement_analysis_ai'][selected_column]
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown("**Найденные размерности:**")
                        dimensions = analysis['analysis_summary']['dimensions_found']
                        for dim, count in dimensions.items():
                            st.write(f"• {dim}: {count}")
                    
                    with col2:
                        st.markdown("**Найденные единицы:**")
                        units = analysis['analysis_summary']['units_found']
                        for unit, count in units.items():
                            st.write(f"• {unit}: {count}")
                    
                    with col3:
                        st.markdown("**Образцы значений:**")
                        for value in analysis['sample_values'][:5]:
                            st.write(f"• {value}")
        else:
            st.info("Сначала найдите числовые колонки в разделе выше.")

    # --- Этап 3: Стандартизация значений ---
    with st.expander("#### 3. ⚙️ Стандартизация значений", expanded=False):
        st.markdown("**Стандартизация всех найденных числовых колонок (исключите ненужные)**")
        
        # Выбор типа стандартизации
        use_ai_standardization = st.checkbox(
            "🧠 Умная стандартизация (AI)", 
            value=False,
            help="AI автоматически определяет и добавляет единицы измерения к числам"
        )
        
        if use_ai_standardization:
            st.info("ℹ️ AI-режим: автоматическое определение и стандартизация единиц + анализ контекста продуктов")
        else:
            st.info("⚡ Быстрый режим: простое regex форматирование + AI-контекст продуктов")
        
        # Выбор колонок для исключения из стандартизации
        if 'numeric_analysis_simple' in st.session_state and st.session_state['numeric_analysis_simple']:
            available_columns = list(st.session_state['numeric_analysis_simple'].keys())
            
            # Определяем колонки с низкой уверенностью для автоматического исключения
            columns_with_low_confidence = []
            if 'measurement_analysis_ai' in st.session_state:
                for column, analysis in st.session_state['measurement_analysis_ai'].items():
                    if analysis['confidence'] < 0.3:  # Низкая уверенность
                        columns_with_low_confidence.append(column)
            
            st.markdown("**Выберите колонки для ИСКЛЮЧЕНИЯ из стандартизации:**")
            excluded_columns = st.multiselect(
                "Колонки для исключения:",
                options=available_columns,
                default=columns_with_low_confidence,
                help="По умолчанию исключены колонки с низкой уверенностью определения единиц (< 30%)"
            )
            
            # Определяем колонки для стандартизации (все кроме исключенных)
            selected_columns = [col for col in available_columns if col not in excluded_columns]
            
            if selected_columns:
                st.success(f"✅ Для стандартизации выбрано {len(selected_columns)} колонок из {len(available_columns)}")
                if excluded_columns:
                    st.info(f"❌ Исключено {len(excluded_columns)} колонок: {', '.join(excluded_columns[:3])}{'...' if len(excluded_columns) > 3 else ''}")
                
                # Показываем список колонок для стандартизации
                with st.expander("📋 Колонки для стандартизации", expanded=False):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Будут стандартизированы:**")
                        for col in selected_columns:
                            confidence = ""
                            if 'measurement_analysis_ai' in st.session_state and col in st.session_state['measurement_analysis_ai']:
                                conf_val = st.session_state['measurement_analysis_ai'][col]['confidence']
                                confidence = f" (уверенность: {conf_val:.2f})"
                            st.write(f"✅ {col}{confidence}")
                    
                    with col2:
                        if excluded_columns:
                            st.markdown("**Исключены из стандартизации:**")
                            for col in excluded_columns:
                                confidence = ""
                                if 'measurement_analysis_ai' in st.session_state and col in st.session_state['measurement_analysis_ai']:
                                    conf_val = st.session_state['measurement_analysis_ai'][col]['confidence']
                                    confidence = f" (уверенность: {conf_val:.2f})"
                                st.write(f"❌ {col}{confidence}")
            else:
                st.warning("⚠️ Все колонки исключены из стандартизации")
            
            if selected_columns:
                # Устанавливаем формат по умолчанию
                standardization_format = 'number_unit'  # Число Единица (100 kg)
                
                # Предпросмотр стандартизации
                if st.button("👁️ Предпросмотр стандартизации", key="preview_standardization"):
                    st.markdown("**🔍 Предпросмотр стандартизации для всех колонок:**")
                    
                    # Создаем компактную таблицу предпросмотра
                    preview_data = []
                    
                    # Сначала определяем единицы по категориям один раз
                    category_units_cache = {}
                    
                    for column in selected_columns:
                        # Получаем единицы для этой колонки (кэшируем для эффективности)
                        if column not in category_units_cache:
                            category_units_cache[column] = analyze_units_by_category(df_param, column, 'group_name', 3)
                        category_units = category_units_cache[column]
                        
                        # Берем первое непустое значение из колонки
                        sample_value = None
                        for value in df_param[column].dropna().head(10):
                            if pd.notna(value) and str(value).strip():
                                sample_value = value
                                break
                        
                        if sample_value is not None:
                            # Получаем информацию о продукте для контекста
                            value_row = df_param[df_param[column] == sample_value]
                            if not value_row.empty:
                                product_name = value_row.get('product_name', pd.Series([None])).iloc[0]
                                category = value_row.get('group_name', pd.Series([None])).iloc[0]
                                description = value_row.get('description', pd.Series([None])).iloc[0]
                            else:
                                product_name = None
                                category = None
                                description = None
                            
                            # Выполняем стандартизацию
                            if use_ai_standardization:
                                standardized = standardize_value_intelligent_optimized(
                                    sample_value, standardization_format, column, product_name, category, category_units, description
                                )
                            else:
                                standardized = standardize_value_simple_optimized(
                                    sample_value, standardization_format, column, product_name, category, category_units, description
                                )
                            
                            # Добавляем в таблицу предпросмотра
                            preview_data.append({
                                'Колонка': column,
                                'До': str(sample_value),
                                'После': str(standardized),
                                'Изменение': "✅ Добавлена единица" if str(sample_value) != str(standardized) else "ℹ️ Без изменений"
                            })
                        else:
                            # Если не найдено значений
                            preview_data.append({
                                'Колонка': column,
                                'До': "—",
                                'После': "—", 
                                'Изменение': "❌ Нет данных"
                            })
                    
                    # Отображаем компактную таблицу
                    if preview_data:
                        preview_df = pd.DataFrame(preview_data)
                        st.dataframe(preview_df, use_container_width=True, hide_index=True)
                        
                        # Показываем статистику
                        changed_count = len([row for row in preview_data if "✅" in row['Изменение']])
                        unchanged_count = len([row for row in preview_data if "ℹ️" in row['Изменение']])
                        error_count = len([row for row in preview_data if "❌" in row['Изменение']])
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("✅ Будут изменены", changed_count)
                        with col2:
                            st.metric("ℹ️ Без изменений", unchanged_count)
                        with col3:
                            st.metric("❌ Ошибки", error_count)
                    else:
                        st.warning("Нет данных для предпросмотра")
                
                # Применение стандартизации
                button_text = "🧠 Применить AI-стандартизацию" if use_ai_standardization else "⚡ Применить простую стандартизацию"
                
                if st.button(button_text, key="apply_standardization"):
                    with st.spinner("Выполняем стандартизацию..."):
                        progress_bar = st.progress(0)
                        
                        for i, column in enumerate(selected_columns):
                            progress_bar.progress((i + 1) / len(selected_columns))
                            
                            # Выполняем стандартизацию
                            standardized_values = batch_standardize_column_optimized(
                                df_param, column, standardization_format, use_ai_standardization, 'group_name'
                            )
                            
                            # Определяем единицу измерения для заголовка
                            unit_for_header = None
                            if 'measurement_analysis_ai' in st.session_state and column in st.session_state['measurement_analysis_ai']:
                                common_units = st.session_state['measurement_analysis_ai'][column]['common_units']
                                if common_units:
                                    unit_for_header = common_units[0]  # Берем первую (наиболее частую) единицу
                            
                            # Если нет единицы из AI-анализа, пытаемся определить из контекста
                            if not unit_for_header:
                                category_units = analyze_units_by_category(df_param, column, 'group_name', 3)
                                if category_units:
                                    unit_for_header = list(category_units.values())[0] if category_units else None
                            
                            # Создаем новое имя колонки с единицей в скобках (если её нет)
                            if unit_for_header and f"({unit_for_header})" not in column:
                                new_column_name = f"{column} ({unit_for_header})"
                            else:
                                new_column_name = column
                            
                            # ЗАМЕНЯЕМ старую колонку на стандартизированную (не добавляем новую)
                            st.session_state['df_standardization'][new_column_name] = standardized_values
                            
                            # Удаляем старую колонку если имя изменилось
                            if new_column_name != column and column in st.session_state['df_standardization'].columns:
                                st.session_state['df_standardization'].drop(columns=[column], inplace=True)
                        
                        progress_bar.empty()
                        
                        # Показываем статистику
                        new_cols_count = len(selected_columns)
                        method_name = "AI-стандартизации" if use_ai_standardization else "простой стандартизации"
                        
                        st.success(f"✅ {method_name} завершена!")
                        st.info(f"📊 Заменено {new_cols_count} колонок стандартизированными версиями с единицами измерения")
                        
                        # Показываем какие колонки были заменены
                        replaced_columns = []
                        for column in selected_columns:
                            # Ищем новую версию с единицами
                            standardized_version = None
                            for col in st.session_state['df_standardization'].columns:
                                if column in col and "(" in col and ")" in col:
                                    standardized_version = col
                                    break
                            if standardized_version:
                                replaced_columns.append(f"{column} → {standardized_version}")
                        
                        if replaced_columns:
                            with st.expander("📋 Замененные колонки", expanded=True):
                                st.markdown("**Колонки заменены на стандартизированные версии:**")
                                for replacement in replaced_columns:
                                    st.write(f"• {replacement}")
                        
                        # Показываем примеры результатов
                        st.markdown("**Примеры результатов:**")
                        
                        # Находим стандартизированные колонки с единицами
                        standardized_columns_with_units = [
                            col for col in st.session_state['df_standardization'].columns 
                            if any(orig_col in col for orig_col in selected_columns) and "(" in col and ")" in col
                        ]
                        
                        for column in standardized_columns_with_units[:2]:  # Показываем для первых 2 колонок
                            # Находим оригинальное имя колонки
                            original_col = None
                            for orig in selected_columns:
                                if orig in column:
                                    original_col = orig
                                    break
                            
                            if original_col and original_col in df_param.columns:
                                sample_comparison = []
                                original_values = df_param[original_col].dropna().head(3).tolist()
                                standardized_values = st.session_state['df_standardization'][column].dropna().head(3).tolist()
                                
                                for orig, stand in zip(original_values, standardized_values):
                                    sample_comparison.append({
                                        'Оригинал': orig,
                                        'Стандартизировано': stand,
                                        'Колонка': column
                                    })
                                
                                if sample_comparison:
                                    comparison_df = pd.DataFrame(sample_comparison)
                                    st.dataframe(comparison_df, use_container_width=True)
            else:
                st.warning("Выберите колонки для стандартизации")
        else:
            st.info("Сначала найдите числовые колонки в этапе 1.")


    # --- Сохранение результатов ---
    with st.expander("#### 5. 💾 Сохранение результатов", expanded=False):
        
        # Показываем текущее состояние
        if 'df_standardization' in st.session_state:
            current_df = st.session_state['df_standardization']
            
            # Подсчитываем стандартизированные колонки (с единицами в скобках)
            standardized_columns = [col for col in current_df.columns if "(" in col and ")" in col]
            total_columns = len(current_df.columns)
            
            st.info(f"📊 Готовая таблица: {len(current_df)} строк, {total_columns} колонок")
            
            if standardized_columns:
                st.success(f"✅ Стандартизированных колонок: {len(standardized_columns)} из {total_columns}")
                st.info(f"📝 Примеры: {', '.join(standardized_columns[:3])}{'...' if len(standardized_columns) > 3 else ''}")
        else:
            st.warning("Сначала выполните стандартизацию")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("💾 Сохранить в standard.csv", key="save_main_ai"):
                try:
                    output_path = os.path.join(os.path.dirname(__file__), '..', 'standard.csv')
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
                    file_name="standard.csv",
                    mime="text/csv",
                    key="download_ai_standardized"
                )
        
        with col3:
            if st.button("🔄 Обновить исходные данные", key="reload_data_ai"):
                # Очищаем кэш и session state
                for key in ['df_standardization', 'numeric_analysis_simple', 'measurement_analysis_ai', 'standardization_results']:
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


else:
    st.warning("Нет данных для обработки. Сначала выполните основную обработку в главном приложении.")
