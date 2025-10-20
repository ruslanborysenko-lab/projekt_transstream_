import streamlit as st
import pandas as pd
from openai import OpenAI
from dotenv import dotenv_values
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import numpy as np

env = dotenv_values(".env")
#if "OPENAI_API_KEY" in st.secrets:
#    env["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
openai_client = OpenAI(api_key=env["OPENAI_API_KEY"])


# Настройка заголовка приложения
st.title("Analysis of the number of employees in the warehouse")

# Простое CSS для центрирования таблиц - убираем лишние стили, мешающие нативному масштабированию
st.markdown("""
<style>
/* Центрирование содержимого таблиц */
div[data-testid="stDataFrame"] th {
    background-color: #f0f2f6;
    font-weight: bold;
    text-align: center;
    padding: 8px;
}

div[data-testid="stDataFrame"] td {
    text-align: center;
    padding: 8px;
}
</style>
""", unsafe_allow_html=True)

# Кэшируем загрузку данных для оптимизации производительности
@st.cache_data
def load_data():
    # Загружаем данные из Excel файла
    df = pd.read_excel("df.xlsx")
    # Используем первую строку как заголовки колонок
    df.columns = df.iloc[0]  
    # Удаляем первую строку, которая стала заголовками
    df = df.drop(df.index[0])  
    # Сбрасываем индексы для корректной нумерации
    df = df.reset_index(drop=True)  
    return df

# Загружаем данные
df = load_data()

# Выводим DataFrame на главную страницу с центрированием
st.subheader("Warehouse Operations and Employee Data")

# Отображаем основную таблицу с полным контролем стиля
# Преобразуем все числовые колонки в числовой тип для правильного выравнивания
df_display = df.copy()
for col in df_display.columns[1:]:  # Пропускаем колонку Month
    df_display[col] = pd.to_numeric(df_display[col], errors='coerce').fillna(0).astype(int)

# Конфигурируем колонки для правильного отображения
column_config = {
    "Month": st.column_config.TextColumn(
        "Month",
        width="medium",
        help="Месяц",
    )
}

# Настройка для числовых колонок
for col in df_display.columns[1:]:
    column_config[col] = st.column_config.NumberColumn(
        col,
        width="small",
        format="%d",
        help=f"Количество: {col}",
    )

st.dataframe(
    df, 
    use_container_width=True, 
    column_config=column_config,
    hide_index=True
)

# Модифицируем функцию для работы с OpenAI API
def optimize_employees_with_ai(df):
    """
    Функция отправляет данные в OpenAI для получения рекомендаций
    по оптимальному количеству сотрудников
    """
    # Подготавливаем данные для отправки в формате строки
    data_text = df.to_string(index=False)
    
    # Новый улучшенный промпт с обновленными операциями
    prompt = f"""
ОПТИМИЗИРУЙ количество сотрудников на складе!
    Используй следующие правила для расчета оптимального количества сотрудников.
    Склад - площадью 1500 квадратных метров.
    Рабочая неделя: 5 дней, рабочий день: 8 часов. Итого: 160 часов в месяц.
    Склад - площадью 1500 квадратных метров.
    Используется в основе расчета 5-дневная рабочая неделя, 8-часовой рабочий день.
    
    Название рабочих мест:
    - Loader - грузчик для ручного труда
    - Forklift_Operator - оператор погрузчика для паллетных перегрузок
    - Operation_manager - работник офиса, обеспечивающий операционную поддержку всех складских процессов, документальную работу, а так же обеспечение дополнительных услуг и сервиса на складе. ВАЖНО: Operation manager может выполнять одновременно несколько функций: Warehouse Logistics Specialist, Warehouse Shift Coordinator, Warehouse Logistics Assistant при малых объемах работ.
    - Sales - особа, ответственная за продажи складских услуг
- Director - руководитель фирмы
Описание складских операций.
Direct_Overloading_20, Cross_Docking_20, Direct_Overloading_40, Cross_Docking_40 - операции по ручной перегрузке товаров. Производятся грузчиками для ручного труда (Loader). Одна операция производится 4 грузчиками. На одну операцию Direct_Overloading_20, Direct_Overloading_40 необходимо - 3 часа. На одну операцию Cross_Docking_20, Cross_Docking_40 необходимо - 5 часов. Важно: при большом объеме операций грузчики могут объединяться в бригады по 4 человека и бригады могут выполнять операции параллельно. Одновременно можно выполнять максимум 3 операции.

ОБЯЗАТЕЛЬНАЯ ОПТИМИЗАЦИЯ LOADER:
Текущие Loader ИЗБЫТОЧНЫ! Рассчитай правильно:
- Ручные операции = Direct_Overloading_20 + Cross_Docking_20 + Direct_Overloading_40 + Cross_Docking_40
- Время = (Direct_Overloading_20 + Direct_Overloading_40) × 3ч + (Cross_Docking_20 + Cross_Docking_40) × 5ч
- НОВОЕ количество Loader = МАКСИМУМ(2, (Время ÷ 160 часов в месяц) × 4)

ПРИМЕР ОБЯЗАТЕЛЬНЫХ РАСЧЕТОВ:
- Май: (6+40)×3 + (2+0)×5 = 148 часов → 148÷160×4 = 3.7 ≈ 4 Loader (было 10 - УМЕНЬШИ!)
- Июнь: (10+38)×3 + (2+0)×5 = 154 часов → 154÷160×4 = 3.85 ≈ 4 Loader (было 9 - УМЕНЬШИ!)

Pallet_Direct_Overloading, Pallet_Cross_Docking - операции по перегрузке паллетного груза. Одна операция производится одним водителем погрузчика (Forklift_Operator) и одним грузчиком для ручного труда (Loader). На одну операцию Pallet_Direct_Overloading необходимо - 1 час.  На одну операцию Pallet_Cross_Docking необходимо - 2 часа. Важно: операцию по перегрузке паллет производят параллельно с ручной перегрузкой товара.

ОБЯЗАТЕЛЬНАЯ ОПТИМИЗАЦИЯ FORKLIFT_OPERATOR:
Текущие Forklift_Operator НЕПРАВИЛЬНЮ! Рассчитай снова:
- Паллетное время = Pallet_Direct_Overloading × 1ч + Pallet_Cross_Docking × 2ч
- НОВОЕ Forklift_Operator = МАКСИМУМ(1, Паллетное_время ÷ 160)

ПРИМЕРЫ ОБЯЗАТЕЛЬНЫХ ИСПРАВЛЕНИЙ:
- Май: 73×1 + 93×2 = 259ч → 259÷160 = 1.6 ≈ 2 Forklift_Operator (было 0 - ДОБАВЬ!)
- Июнь: 61×1 + 156×2 = 373ч → 373÷160 = 2.3 ≈ 2 Forklift_Operator (было 0 - ДОБАВЬ!)
- Июль: 116×1 + 147×2 = 410ч → 410÷160 = 2.6 ≈ 3 Forklift_Operator (было 0 - ДОБАВЬ!)

ОБЯЗАТЕЛЬНАЯ ОПТИМИЗАЦИЯ OPERATION_MANAGER:
Текущие Operation_manager ИЗБЫТОЧНЫ! Оптимизируй:
- Офисные операции = Other_revenue + Reloading_Service + Goods_Storage + Additional_Service
- НОВОЕ Operation_manager = МАКСИМУМ(2, МИНИМУМ(5, Офисные_операции ÷ 150))

ПРИМЕРЫ ОПТИМИЗАЦИИ:
- Май: 75+233+137+65 = 510 → 510÷150 = 3.4 ≈ 3 Operation_manager (было 2 - УВЕЛИЧЬ!)
- Июнь: 145+295+225+79 = 744 → 744÷150 = 5 Operation_manager (было 3 - УВЕЛИЧЬ!)
Other_revenue - операция по оформлению документов. Выполняется работниками офиса (Operation_manager).
Reloading_Service - операции по оформлению документов прихода на склад и выпуска товаров со склада. Выполняется работниками офиса (Operation_manager).
Goods_Storage - операции по складскому обслуживанию. Выполняется работниками офиса (Operation_manager).
Additional_Service - операции по обеспечению дополнительных складских услуг и сервиса. Выполняется работниками офиса (Operation_manager).
    
ОПТИМИЗИРУЙ количество сотрудников СОГЛАСНО ФОРМУЛАМ ВЫШЕ!

Исходные данные: {data_text}

ОБЯЗАТЕЛЬНО ИСПРАВЬ ОШИБКИ В КОЛИЧЕСТВЕ СОТРУДНИКОВ!
- Director: всегда 1
- Sales: всегда 1  
- Operation_manager: ПО ФОРМУЛЕ выше!
- Loader: ПО ФОРМУЛЕ выше!
- Forklift_Operator: ПО ФОРМУЛЕ выше!

ВОЗВРАТИ ПОЛНУЮ ТАБЛИЦУ (ВСЕ 16 колонок) С ОПТИМИЗИРОВАННЫМИ ЧИСЛАМИ:
Month Direct_Overloading_20 Cross_Docking_20 Direct_Overloading_40 Cross_Docking_40 Pallet_Direct_Overloading Pallet_Cross_Docking Other_revenue Reloading_Service Goods_Storage Additional_Service Director Sales Operation_manager Loader Forklift_Operator
    """
    
    try:
        # Отправляем запрос к OpenAI API с моделью GPT-4o для оптимизации
        response = openai_client.chat.completions.create(
            model="gpt-4o",  # Используем GPT-4o для лучшей оптимизации
            messages=[
                {"role": "system", "content": "Ты эксперт по оптимизации складских операций. Возвращай только числовые данные таблицы без дополнительных объяснений."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,  
            temperature=0.1   # Снижена для более стабильных результатов
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"Error when calling OpenAI API: {str(e)}"

# Функция для анализа различий между исходными и оптимизированными данными
def analyze_differences(original_df, optimized_df):
    """
    Анализирует различия между исходными и оптимизированными DataFrame
    """
    try:
        # Создаем детальный анализ различий
        analysis = []
        
        # Проверяем все колонки с числовыми данными (количество сотрудников) - обновленные названия
        employee_columns = ['Director', 'Sales', 'Operation_manager', 'Loader', 'Forklift_Operator']
        
        analysis.append("Analysis of average values for each job position (May–September):")
        analysis.append("")
        
        for col in employee_columns:
            if col in original_df.columns and col in optimized_df.columns:
                try:
                    # Преобразуем в числовой формат и вычисляем средние значения
                    orig_mean = pd.to_numeric(original_df[col], errors='coerce').mean()
                    opt_mean = pd.to_numeric(optimized_df[col], errors='coerce').mean()
                    
                    if pd.notna(orig_mean) and pd.notna(opt_mean):  # Проверяем, что значения не NaN
                        diff_abs = opt_mean - orig_mean
                        if orig_mean > 0:
                            diff_percent = (diff_abs / orig_mean) * 100
                            percent_str = f" ({diff_percent:+.1f}%)"
                        else:
                            percent_str = ""
                        
                        if diff_abs > 0.1:  # Учитываем погрешность для средних значений
                            direction = "⬆️ Increase"
                        elif diff_abs < -0.1:
                            direction = "⬇️ Decrease"
                        else:
                            direction = "➡️ No significant change"
                        
                        analysis.append(f"**{col}**: {direction}")
                        analysis.append(f"   - Average before: {orig_mean:.1f} чел.")
                        analysis.append(f"   - Average after: {opt_mean:.1f} чел.")
                        analysis.append(f"   - Average difference: {diff_abs:+.1f} чел.{percent_str}")
                        analysis.append("")
                except Exception as e:
                    analysis.append(f"**{col}**: Analysis error - {str(e)}")
                    analysis.append("")
        
        return "\n".join(analysis)
        
    except Exception as e:
        return f"Error analyzing data: {str(e)}"

# Функция для получения оптимизированного DataFrame
def get_optimized_dataframe(original_df, optimized_data):
    """
    Преобразует оптимизированные данные в DataFrame
    """
    try:
        import io
        # Парсим данные, полученные от AI
        lines = optimized_data.strip().split('\n')
        
        data_rows = []
        valid_months = ['May', 'June', 'July', 'August', 'September']
        
        for line in lines:
            if line.strip():  # Пропускаем пустые строки
                parts = line.split()
                
                # Фильтрация: пропускаем заголовки и некорректные строки
                if (len(parts) >= 15 and 
                    len(parts[0]) <= 10 and  # Не слишком длинное слово
                    not parts[0] == 'Month' and  # Не заголовок
                    not all(part.isalpha() and len(part) > 4 for part in parts[:5])):  # Не строка заголовков
                    
                    # Если первое слово - месяц, используем как есть
                    if parts[0] in valid_months:
                        if len(parts) >= 16:  # Полная таблица
                            data_rows.append(parts[:len(original_df.columns)])
                        elif len(parts) == 15:  # Без месяца - добавляем
                            data_rows.append([parts[0]] + parts[1:len(original_df.columns)])
                    # Если нет месяца в начале, добавляем его
                    elif len(parts) == 15:
                        month_idx = len(data_rows)
                        if month_idx < len(valid_months):
                            row = [valid_months[month_idx]] + parts[:15]
                            data_rows.append(row[:len(original_df.columns)])
        
        if data_rows:
            optimized_df = pd.DataFrame(data_rows, columns=original_df.columns)
            return optimized_df
        else:
            # Если не удалось распарсить, возвращаем исходные данные
            return original_df
    except Exception as e:
        st.error(f"Error processing optimized data: {str(e)}")
        return original_df

# Функция для прогнозирования операций с помощью линейной регрессии
def predict_future_operations(df, target_month, use_optimized_data=True):
    """
    Прогнозирует количество операций и сотрудников на указанный месяц
    """
    try:
        # Данные для прогнозирования (уже оптимизированные если необходимо)
        if use_optimized_data and st.session_state.get('optimization_data'):
            optimized_df = get_optimized_dataframe(df, st.session_state.optimization_data)
            if len(optimized_df) > 0 and len(optimized_df.columns) == len(df.columns):
                df = optimized_df
        
        # Преобразуем месяцы в числовой формат
        month_mapping = {
            'May': 5, 'June': 6, 'July': 7, 'August': 8, 'September': 9,
            'October': 10, 'November': 11, 'December': 12
        }
        
        # Подготавливаем данные для обучения - очищаем от пробелов
        months_numeric = []
        for month in df[df.columns[0]]:
            month_clean = str(month).strip()  # Убираем пробелы
            if month_clean in month_mapping:
                months_numeric.append(month_mapping[month_clean])
            else:
                st.error(f"Unknown month: '{month_clean}'. Available: {list(month_mapping.keys())}")
                return None, None
        X = np.array(months_numeric).reshape(-1, 1)  # Месяцы как признак
        
        # Прогнозируем каждый тип операций - обновленные колонки
        predictions = {}
        operation_columns = ['Direct_Overloading_20', 'Cross_Docking_20', 'Direct_Overloading_40', 'Cross_Docking_40', 'Pallet_Direct_Overloading', 'Pallet_Cross_Docking', 'Other_revenue', 'Reloading_Service', 'Goods_Storage', 'Additional_Service']
        employee_columns = ['Director', 'Sales', 'Operation_manager', 'Loader', 'Forklift_Operator']
        
        target_month_num = month_mapping[target_month]
        X_pred = np.array([[target_month_num]])
        
        # Прогнозируем операции с помощью чистой линейной регрессии без коррекции роста
        for col in operation_columns:
            if col in df.columns:
                y = pd.to_numeric(df[col], errors='coerce')
                model = LinearRegression()
                model.fit(X, y)
                pred_value = model.predict(X_pred)[0]
                # Используем чистую линейную регрессию без дополнительных коэффициентов роста
                predictions[col] = max(0, int(round(pred_value)))
        
        # Прогнозируем количество сотрудников на основе роста/падения операций от базового уровня (сентябрь)
        # Получаем базовые данные (последний месяц - сентябрь)
        baseline_data = df.iloc[-1]  # Последняя строка - сентябрь
        
        for col in employee_columns:
            if col in df.columns:
                # Получаем базовое количество сотрудников из сентября
                baseline_employees = pd.to_numeric(baseline_data[col], errors='coerce') or 0
                
                if col == 'Director':
                    # Директор всегда 1
                    predictions[col] = 1
                elif col == 'Sales':
                    # Продавец всегда 1  
                    predictions[col] = 1
                elif col == 'Operation_manager':
                    # Для Operation_manager - зависимость от офисных операций
                    office_operations = ['Other_revenue', 'Reloading_Service', 'Goods_Storage', 'Additional_Service']
                    
                    # Базовые офисные операции (сентябрь)
                    baseline_office_ops = sum([pd.to_numeric(baseline_data[op_col], errors='coerce') or 0 for op_col in office_operations])
                    
                    # Прогнозируемые офисные операции
                    predicted_office_ops = sum([predictions.get(op_col, 0) for op_col in office_operations if op_col in predictions])
                    
                    # Коэффициент изменения операций
                    if baseline_office_ops > 0:
                        ops_change_ratio = predicted_office_ops / baseline_office_ops
                    else:
                        ops_change_ratio = 1.0
                    
                    # Корректируем количество менеджеров на основе изменения операций
                    if ops_change_ratio >= 1.2:  # Рост больше 20%
                        staff_change = 1 + (ops_change_ratio - 1) * 0.4  # 40% от роста операций
                    elif ops_change_ratio <= 0.8:  # Падение больше 20%
                        staff_change = 1 + (ops_change_ratio - 1) * 0.3  # 30% от падения операций
                    else:
                        staff_change = 1.0  # Нет значительных изменений
                    
                    predicted_managers = baseline_employees * staff_change
                    # Верхнее ограничение - уровень сентября (оптимизированные данные)
                    predictions[col] = max(2, min(baseline_employees, int(round(predicted_managers))))
                    
                elif col == 'Loader':
                    # Для Loader - зависимость от ручных операций
                    manual_operations = ['Direct_Overloading_20', 'Cross_Docking_20', 'Direct_Overloading_40', 'Cross_Docking_40']
                    
                    # Базовые ручные операции (сентябрь)
                    baseline_manual_ops = sum([pd.to_numeric(baseline_data[op_col], errors='coerce') or 0 for op_col in manual_operations])
                    
                    # Прогнозируемые ручные операции
                    predicted_manual_ops = sum([predictions.get(op_col, 0) for op_col in manual_operations if op_col in predictions])
                    
                    # Коэффициент изменения операций
                    if baseline_manual_ops > 0:
                        ops_change_ratio = predicted_manual_ops / baseline_manual_ops
                    else:
                        ops_change_ratio = 1.0
                    
                    # Корректируем количество грузчиков на основе изменения операций
                    if ops_change_ratio >= 1.15:  # Рост больше 15%
                        staff_change = 1 + (ops_change_ratio - 1) * 0.6  # 60% от роста операций
                    elif ops_change_ratio <= 0.85:  # Падение больше 15%
                        staff_change = 1 + (ops_change_ratio - 1) * 0.4  # 40% от падения операций
                    else:
                        staff_change = 1.0  # Нет значительных изменений
                    
                    predicted_loaders = baseline_employees * staff_change
                    # Верхнее ограничение - уровень сентября (оптимизированные данные)
                    predictions[col] = max(2, min(baseline_employees, int(round(predicted_loaders))))
                    
                elif col == 'Forklift_Operator':
                    # Для Forklift_Operator - зависимость от паллетных операций
                    pallet_operations = ['Pallet_Direct_Overloading', 'Pallet_Cross_Docking']
                    
                    # Базовые паллетные операции (сентябрь)
                    baseline_pallet_ops = sum([pd.to_numeric(baseline_data[op_col], errors='coerce') or 0 for op_col in pallet_operations])
                    
                    # Прогнозируемые паллетные операции
                    predicted_pallet_ops = sum([predictions.get(op_col, 0) for op_col in pallet_operations if op_col in predictions])
                    
                    # Если нет паллетных операций, то минимум операторов
                    if predicted_pallet_ops == 0:
                        predictions[col] = 1
                    else:
                        # Коэффициент изменения операций
                        if baseline_pallet_ops > 0:
                            ops_change_ratio = predicted_pallet_ops / baseline_pallet_ops
                        else:
                            ops_change_ratio = 1.0
                        
                        # Корректируем количество операторов на основе изменения операций
                        if ops_change_ratio >= 1.2:  # Рост больше 20%
                            staff_change = 1 + (ops_change_ratio - 1) * 0.7  # 70% от роста операций
                        elif ops_change_ratio <= 0.8:  # Падение больше 20%
                            staff_change = 1 + (ops_change_ratio - 1) * 0.5  # 50% от падения операций
                        else:
                            staff_change = 1.0  # Нет значительных изменений
                        
                        predicted_operators = baseline_employees * staff_change
                        # Верхнее ограничение - уровень сентября (оптимизированные данные)
                        predictions[col] = max(1, min(baseline_employees, int(round(predicted_operators))))
        
        # Создаем строку результата
        result_row = [target_month]
        for col in df.columns[1:]:
            result_row.append(predictions.get(col, 0))
        
        # Создаем DataFrame с прогнозом
        forecast_df = pd.DataFrame([result_row], columns=df.columns)
        
        return forecast_df, predictions
        
    except Exception as e:
        st.error(f"Error during forecasting: {str(e)}")
        return None, None

# Функция для создания сравнительного анализа
def create_comparison_analysis(original_df, optimized_df, forecast_df, forecast_month):
    """
    Создает сравнительную таблицу и графики для анализа различий между данными
    """
    try:
        # Создаем сводную таблицу для сравнения
        employee_columns = ['Operation_manager', 'Loader', 'Forklift_Operator']
        
        comparison_data = []
        
        # Данные на сентябрь (последний месяц)
        sep_original = original_df.iloc[-1]  # Последняя строка оригинальных данных
        sep_optimized = optimized_df.iloc[-1] if len(optimized_df) > 0 else sep_original
        forecast_data = forecast_df.iloc[0] if forecast_df is not None and len(forecast_df) > 0 else None
        
        for emp_type in employee_columns:
            original_val = pd.to_numeric(sep_original[emp_type], errors='coerce') or 0
            optimized_val = pd.to_numeric(sep_optimized[emp_type], errors='coerce') or 0
            forecast_val = pd.to_numeric(forecast_data[emp_type], errors='coerce') if forecast_data is not None else 0
            
            # Расчет процентных изменений
            opt_change = ((optimized_val - original_val) / original_val * 100) if original_val > 0 else 0
            forecast_change = ((forecast_val - optimized_val) / optimized_val * 100) if optimized_val > 0 else 0
            total_change = ((forecast_val - original_val) / original_val * 100) if original_val > 0 else 0
            
            comparison_data.append({
                'Employee Type': emp_type,
                'Original (Sep)': int(original_val),
                'Optimized (Sep)': int(optimized_val),
                f'Forecast ({forecast_month})': int(forecast_val),
                'Optimization Change (%)': round(opt_change, 1),
                f'Growth to {forecast_month} (%)': round(forecast_change, 1),
                'Total Change (%)': round(total_change, 1)
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Создаем график сравнения
        fig_comparison = go.Figure()
        
        x_labels = comparison_df['Employee Type']
        
        fig_comparison.add_trace(go.Bar(
            name='Original',
            x=x_labels,
            y=comparison_df['Original (Sep)'],
            marker_color='lightblue'
        ))
        
        fig_comparison.add_trace(go.Bar(
            name='Optimized',
            x=x_labels,
            y=comparison_df['Optimized (Sep)'],
            marker_color='orange'
        ))
        
        fig_comparison.add_trace(go.Bar(
            name=f'Forecast ({forecast_month})',
            x=x_labels,
            y=comparison_df[f'Forecast ({forecast_month})'],
            marker_color='green'
        ))
        
        fig_comparison.update_layout(
            title='Employee Numbers Comparison: Original vs Optimized vs Forecast',
            xaxis_title='Employee Type',
            yaxis_title='Number of Employees',
            barmode='group'
        )
        
        return comparison_df, fig_comparison
        
    except Exception as e:
        st.error(f"Error creating comparison analysis: {str(e)}")
        return None, None

def create_performance_metrics(original_df, optimized_df, forecast_df):
    """
    Создает метрики производительности и эффективности
    """
    try:
        operation_columns = ['Direct_Overloading_20', 'Cross_Docking_20', 'Direct_Overloading_40', 'Cross_Docking_40', 
                           'Pallet_Direct_Overloading', 'Pallet_Cross_Docking', 'Other_revenue', 'Reloading_Service', 
                           'Goods_Storage', 'Additional_Service']
        employee_columns = ['Operation_manager', 'Loader', 'Forklift_Operator']
        
        metrics = {}
        
        # Метрики для сентября
        sep_original = original_df.iloc[-1]
        sep_optimized = optimized_df.iloc[-1] if len(optimized_df) > 0 else sep_original
        
        # Общий объем операций
        original_ops = sum([pd.to_numeric(sep_original[col], errors='coerce') or 0 for col in operation_columns])
        optimized_ops = sum([pd.to_numeric(sep_optimized[col], errors='coerce') or 0 for col in operation_columns])
        
        # Общее количество сотрудников
        original_employees = sum([pd.to_numeric(sep_original[col], errors='coerce') or 0 for col in employee_columns])
        optimized_employees = sum([pd.to_numeric(sep_optimized[col], errors='coerce') or 0 for col in employee_columns])
        
        # Производительность (операций на сотрудника)
        original_productivity = original_ops / original_employees if original_employees > 0 else 0
        optimized_productivity = optimized_ops / optimized_employees if optimized_employees > 0 else 0
        
        metrics['Operations Efficiency'] = {
            'Original Operations': int(original_ops),
            'Optimized Operations': int(optimized_ops),
            'Operations Change (%)': round(((optimized_ops - original_ops) / original_ops * 100) if original_ops > 0 else 0, 1)
        }
        
        metrics['Staff Efficiency'] = {
            'Original Staff': int(original_employees),
            'Optimized Staff': int(optimized_employees),
            'Staff Change (%)': round(((optimized_employees - original_employees) / original_employees * 100) if original_employees > 0 else 0, 1)
        }
        
        metrics['Productivity'] = {
            'Original (ops/employee)': round(original_productivity, 1),
            'Optimized (ops/employee)': round(optimized_productivity, 1),
            'Productivity Gain (%)': round(((optimized_productivity - original_productivity) / original_productivity * 100) if original_productivity > 0 else 0, 1)
        }
        
        return metrics
        
    except Exception as e:
        st.error(f"Error creating performance metrics: {str(e)}")
        return None

def create_comprehensive_charts(df, optimized_df=None):
    """
    Создает расширенные графики для анализа данных
    """
    try:
        operation_columns = ['Direct_Overloading_20', 'Cross_Docking_20', 'Direct_Overloading_40', 'Cross_Docking_40', 
                           'Pallet_Direct_Overloading', 'Pallet_Cross_Docking', 'Other_revenue', 'Reloading_Service', 
                           'Goods_Storage', 'Additional_Service']
        
        # График 1: Распределение операций по типам
        ops_totals = {}
        for col in operation_columns:
            ops_totals[col] = pd.to_numeric(df[col], errors='coerce').sum()
        
        fig_pie = px.pie(values=list(ops_totals.values()), 
                        names=list(ops_totals.keys()),
                        title='Distribution of Operations by Type (May-September 2025)')
        
        # График 2: Тенденции операций по месяцам
        months = df[df.columns[0]].tolist()
        
        fig_trends = go.Figure()
        
        # Группируем операции по категориям
        direct_ops = pd.to_numeric(df['Direct_Overloading_20'], errors='coerce') + pd.to_numeric(df['Direct_Overloading_40'], errors='coerce')
        cross_ops = pd.to_numeric(df['Cross_Docking_20'], errors='coerce') + pd.to_numeric(df['Cross_Docking_40'], errors='coerce')
        pallet_ops = pd.to_numeric(df['Pallet_Direct_Overloading'], errors='coerce') + pd.to_numeric(df['Pallet_Cross_Docking'], errors='coerce')
        service_ops = pd.to_numeric(df['Other_revenue'], errors='coerce') + pd.to_numeric(df['Additional_Service'], errors='coerce')
        
        fig_trends.add_trace(go.Scatter(x=months, y=direct_ops, mode='lines+markers', name='Direct Overloading', line=dict(width=3)))
        fig_trends.add_trace(go.Scatter(x=months, y=cross_ops, mode='lines+markers', name='Cross Docking', line=dict(width=3)))
        fig_trends.add_trace(go.Scatter(x=months, y=pallet_ops, mode='lines+markers', name='Pallet Operations', line=dict(width=3)))
        fig_trends.add_trace(go.Scatter(x=months, y=service_ops, mode='lines+markers', name='Service Operations', line=dict(width=3)))
        
        fig_trends.update_layout(
            title='Operations Trends by Category (May-September 2025)',
            xaxis_title='Month',
            yaxis_title='Number of Operations',
            hovermode='x unified'
        )
        
        # График 3: Корреляционная матрица
        numeric_cols = operation_columns + ['Operation_manager', 'Loader', 'Forklift_Operator']
        correlation_data = df[numeric_cols].apply(pd.to_numeric, errors='coerce').corr()
        
        fig_corr = px.imshow(correlation_data, 
                           text_auto=True, 
                           aspect="auto",
                           title='Correlation Matrix: Operations vs Employees',
                           color_continuous_scale='RdBu_r')
        
        return fig_pie, fig_trends, fig_corr
        
    except Exception as e:
        st.error(f"Error creating comprehensive charts: {str(e)}")
        return None, None, None

# Функция для создания директорской аналитики
def create_executive_dashboard(original_df, combined_df, selected_operation, selected_month):
    """
    Создает яркую и простую визуализацию для директора
    Использует original_df для данных персонала (реальные показатели) и combined_df для операций
    """
    try:
        months_order = ["May", "June", "July", "August", "September", "October", "November", "December"]
        month_idx = months_order.index(selected_month)
        
        # Для исторических месяцев (May-September) берем данные из первой таблицы
        if month_idx < len(original_df):
            # Данные операций из combined_df (могут включать прогнозы)
            month_operations_data = combined_df.iloc[month_idx] if month_idx < len(combined_df) else combined_df.iloc[-1]
            # Данные персонала из original_df (реальные показатели)
            month_staff_data = original_df.iloc[month_idx] if month_idx < len(original_df) else original_df.iloc[-1]
        elif month_idx < len(combined_df):
            # Для прогнозных месяцев используем combined_df, но предупреждаем о прогнозных данных персонала
            month_operations_data = combined_df.iloc[month_idx]
            month_staff_data = combined_df.iloc[month_idx]
        else:
            return False
        
        # Получаем данные за месяц - операции из combined_df, персонал из original_df
        operation_value = pd.to_numeric(month_operations_data[selected_operation], errors='coerce') or 0
        
        # Ключевые метрики
        operation_columns = ['Direct_Overloading_20', 'Cross_Docking_20', 'Direct_Overloading_40', 'Cross_Docking_40', 
                           'Pallet_Direct_Overloading', 'Pallet_Cross_Docking', 'Other_revenue', 'Reloading_Service', 
                           'Goods_Storage', 'Additional_Service']
        employee_columns = ['Director', 'Sales', 'Operation_manager', 'Loader', 'Forklift_Operator']
        
        # Операции берем из combined_df (могут включать прогнозы)
        total_operations = sum([pd.to_numeric(month_operations_data[col], errors='coerce') or 0 for col in operation_columns])
        # Персонал берем из original_df (реальные показатели)
        total_employees = sum([pd.to_numeric(month_staff_data[col], errors='coerce') or 0 for col in employee_columns])
            
        # Метрики в карточках
        col1, col2, col3, col4 = st.columns(4)
            
        # Рассчитываем изменения по сравнению с предыдущим месяцем для всех метрик
        delta_operation = None
        delta_total_ops = None
        delta_staff = None 
        delta_productivity = None
        
        if month_idx > 0:
            # Получаем данные предыдущего месяца
            if month_idx - 1 < len(original_df):  
                prev_month_operations_data = original_df.iloc[month_idx - 1] if month_idx - 1 < len(original_df) else combined_df.iloc[month_idx - 1]
                prev_month_staff_data = original_df.iloc[month_idx - 1] if month_idx - 1 < len(original_df) else original_df.iloc[-1]
            else:  
                prev_month_operations_data = combined_df.iloc[month_idx - 1]
                prev_month_staff_data = combined_df.iloc[month_idx - 1]
            
            # Расчеты для выбранной операции
            prev_operation_value = pd.to_numeric(prev_month_operations_data[selected_operation], errors='coerce') or 0
            if prev_operation_value > 0:
                change_operation = ((operation_value - prev_operation_value) / prev_operation_value * 100)
                delta_operation = f"{change_operation:+.1f}%"
            
            # Расчеты для Total Operations
            prev_total_operations = sum([pd.to_numeric(prev_month_operations_data[col], errors='coerce') or 0 for col in operation_columns])
            if prev_total_operations > 0:
                change_total_ops = ((total_operations - prev_total_operations) / prev_total_operations * 100)
                delta_total_ops = f"{change_total_ops:+.1f}%"
            
            # Расчеты для Total Staff
            prev_total_employees = sum([pd.to_numeric(prev_month_staff_data[col], errors='coerce') or 0 for col in employee_columns])
            if prev_total_employees > 0:
                change_staff = ((total_employees - prev_total_employees) / prev_total_employees * 100)
                delta_staff = f"{change_staff:+.1f}%"
            
            # Расчеты для Productivity
            prev_productivity = prev_total_operations / prev_total_employees if prev_total_employees > 0 else 0
            productivity = total_operations / total_employees if total_employees > 0 else 0
            if prev_productivity > 0:
                change_productivity = ((productivity - prev_productivity) / prev_productivity * 100)
                delta_productivity = f"{change_productivity:+.1f}%"
        
        with col1:
            st.metric(
                label=f"{selected_operation}",
                value=f"{int(operation_value)}",
                delta=delta_operation
            )
        
        with col2:
            st.metric(
                label="Total Operations",
                value=f"{int(total_operations)}",
                delta=delta_total_ops
            )
        
        with col3:
            st.metric(
                label="Total Staff",
                value=f"{int(total_employees)}",
                delta=delta_staff
            )
        
        with col4:
            productivity = total_operations / total_employees if total_employees > 0 else 0
            st.metric(
                label="Productivity",
                value=f"{productivity:.1f}",
                delta=delta_productivity,
                help="Operations per employee - shows how many warehouse operations each employee handles on average per month. Higher values indicate better efficiency."
            )
            
        # График 1: Обзор всех операций за месяц
        operation_values = [pd.to_numeric(month_operations_data[col], errors='coerce') or 0 for col in operation_columns]
        operation_labels = [
            'Direct 20ft', 'Cross 20ft', 'Direct 40ft', 'Cross 40ft',
            'Pallet Direct', 'Pallet Cross', 'Revenue Ops', 'Reload Service', 
            'Storage', 'Additional'
        ]
            
        # Фильтруем операции с нулевыми значениями
        filtered_values = []
        filtered_labels = []
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3', '#54A0FF', '#5F27CD', '#00D2D3', '#FF9F43']
        filtered_colors = []
        
        for i, (value, label) in enumerate(zip(operation_values, operation_labels)):
            if value > 0:
                filtered_values.append(value)
                filtered_labels.append(f"{label}: {int(value)}")
                filtered_colors.append(colors[i % len(colors)])
        
        if filtered_values:
            fig_all_ops = go.Figure(data=[
                go.Pie(
                    labels=filtered_labels,
                    values=filtered_values,
                    hole=0.3,
                    marker=dict(
                        colors=filtered_colors,
                        line=dict(color='#FFFFFF', width=2)
                    ),
                    textinfo='label+percent',
                    textposition='auto',
                    hovertemplate='%{label}<br>%{value} operations<br>%{percent}<extra></extra>'
                )
            ])
            
            fig_all_ops.update_layout(
                title={
                    'text': f"All Operations Overview - {selected_month} 2025",
                    'x': 0.5,
                    'font': {'size': 20, 'color': '#2E86C1'}
                },
                font=dict(size=12),
                height=500,
                showlegend=True,
                legend=dict(
                    orientation="v",
                    yanchor="middle",
                    y=0.5,
                    xanchor="left",
                    x=1.05
                )
            )
            
            st.plotly_chart(fig_all_ops, use_container_width=True)
        
        # График 2: Соотношение выбранной операции к общему объему
        if operation_value > 0 and total_operations > operation_value:
            fig_pie = go.Figure(data=[
                go.Pie(
                    labels=[selected_operation, "Other Operations"],
                    values=[operation_value, total_operations - operation_value],
                    hole=0.4,
                    marker=dict(
                        colors=['#FF6B6B', '#4ECDC4'],
                        line=dict(color='#FFFFFF', width=3)
                    )
                )
            ])
            
            fig_pie.update_layout(
                title={
                    'text': f"{selected_operation} Share in Total Operations",
                    'x': 0.5,
                    'font': {'size': 20, 'color': '#2E86C1'}
                },
                font=dict(size=14),
                height=400,
                showlegend=True
            )
            
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # График 3: Распределение сотрудников (реальные данные)
        employee_values = [pd.to_numeric(month_staff_data[col], errors='coerce') or 0 for col in employee_columns]
        
        fig_bar = go.Figure(data=[
            go.Bar(
                x=employee_columns,
                y=employee_values,
                marker=dict(
                    color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57'],
                    line=dict(color='#FFFFFF', width=2)
                ),
                text=employee_values,
                textposition='auto'
            )
        ])
        
        fig_bar.update_layout(
            title={
                'text': f"Staff Distribution in {selected_month}",
                'x': 0.5,
                'font': {'size': 20, 'color': '#2E86C1'}
            },
            xaxis_title="Employee Type",
            yaxis_title="Number of Employees",
            font=dict(size=14),
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # График 4: Сравнение с предыдущим месяцем
        if month_idx > 0:
            prev_month_name = months_order[month_idx - 1]
            
            # Правильно определяем источник данных для предыдущего месяца
            if month_idx - 1 < len(original_df):  # Предыдущий месяц - исторические данные
                prev_month_operations_data = original_df.iloc[month_idx - 1] if month_idx - 1 < len(original_df) else combined_df.iloc[month_idx - 1]
            else:  # Предыдущий месяц - прогнозные данные
                prev_month_operations_data = combined_df.iloc[month_idx - 1]
            
            prev_operation_value = pd.to_numeric(prev_month_operations_data[selected_operation], errors='coerce') or 0
            
            change = ((operation_value - prev_operation_value) / prev_operation_value * 100) if prev_operation_value > 0 else 0
            
            # Отладочная информация убрана - после успешной отладки
            # st.write(f"📊 **Month Comparison for {selected_operation}:**")
            # st.write(f"   • {prev_month_name}: {int(prev_operation_value)} operations")
            # st.write(f"   • {selected_month}: {int(operation_value)} operations")
            # st.write(f"   • Formula: ({int(operation_value)} - {int(prev_operation_value)}) / {int(prev_operation_value)} × 100 = **{change:+.1f}%**")
            
            fig_comparison = go.Figure(data=[
                go.Bar(
                    x=[prev_month_name, selected_month],
                    y=[prev_operation_value, operation_value],
                    marker=dict(
                        color=[
                            '#95A5A6',  # Серый для предыдущего месяца
                            '#E74C3C' if change < 0 else '#27AE60'  # Красный при снижении, зеленый при росте
                        ],
                        line=dict(color='#FFFFFF', width=2)
                    ),
                    text=[int(prev_operation_value), int(operation_value)],
                    textposition='auto'
                )
            ])
            
            fig_comparison.update_layout(
                title={
                    'text': f"{selected_operation}: Month-to-Month Comparison",
                    'x': 0.5,
                    'font': {'size': 20, 'color': '#2E86C1'}
                },
                xaxis_title="Month",
                yaxis_title="Operations",
                font=dict(size=14),
                height=400,
                showlegend=False,
                annotations=[
                    dict(
                        x=1,
                        y=max(prev_operation_value, operation_value) * 1.1,
                        text=f"Change: {change:+.1f}%",
                        showarrow=True,
                        arrowhead=2,
                        arrowcolor='#E74C3C' if change < 0 else '#27AE60',
                        font=dict(size=16, color='#E74C3C' if change < 0 else '#27AE60')
                    )
                ]
            )
            
            st.plotly_chart(fig_comparison, use_container_width=True)
        
        return True
        
        return False
        
    except Exception as e:
        st.error(f"Error creating executive dashboard: {str(e)}")
        return False

# Функция для создания графиков зависимости
def create_dependency_charts(df):
    """
    Создает графики зависимости количества работников от объема операций
    """
    try:
        # Подготавливаем данные для визуализации - обновленные колонки
        operation_columns = ['Direct_Overloading_20', 'Cross_Docking_20', 'Direct_Overloading_40', 'Cross_Docking_40', 'Pallet_Direct_Overloading', 'Pallet_Cross_Docking', 'Other_revenue', 'Reloading_Service', 'Goods_Storage', 'Additional_Service']
        employee_columns = ['Operation_manager', 'Loader', 'Forklift_Operator']
        
        # Создаем суммарные операции
        total_operations = df[operation_columns].apply(pd.to_numeric, errors='coerce').sum(axis=1)
        
        # График 1: Зависимость Loader от общего объема операций
        fig1 = px.scatter(x=total_operations, y=pd.to_numeric(df['Loader'], errors='coerce'),
                         labels={'x': 'Total volume of operations', 'y': 'Number of Loaders'},
                         title='Dependence of the number of loaders on the volume of operations',
                         trendline='ols')
        fig1.update_traces(marker=dict(size=12, color='blue'))
        
        # График 2: Зависимость Operation_manager от Additional_Service - обновленное название
        fig2 = px.scatter(x=pd.to_numeric(df['Additional_Service'], errors='coerce'),
                         y=pd.to_numeric(df['Operation_manager'], errors='coerce'),
                         labels={'x': 'Additional Services', 'y': 'Number of Operation Managers'},
                         title='Operation Managers dependence on additional services',
                         trendline='ols')
        fig2.update_traces(marker=dict(size=12, color='green'))
        
        return fig1, fig2
        
    except Exception as e:
        st.error(f"Error during plot creation: {str(e)}")
        return None, None

# Создаем боковую панель (sidebar)
st.sidebar.header("Control Panel")

# Добавляем кнопку оптимизации с полной очисткой кеша
if st.sidebar.button("Employees number optimisation"):
    # Полная очистка всех кешей при повторном нажатии
    st.session_state.forecast_data = None
    st.session_state.show_forecast = False
    st.session_state.optimization_data = None  # Очищаем старые данные оптимизации
    st.session_state.show_optimization = False  # Сбрасываем флаг показа
    st.session_state.last_calculated_month = None  # Очищаем кеш прогноза
    
    # Показываем индикатор загрузки
    with st.spinner('Analyzing data with AI... Please wait for fresh optimization results.'):
        # Получаем НОВЫЕ оптимизированные данные от OpenAI
        optimized_data = optimize_employees_with_ai(df)
        
        # Сохраняем новые данные в session_state
        st.session_state.optimization_data = optimized_data
        st.session_state.show_optimization = True
        # Принудительное обновление страницы
        st.rerun()

# Добавляем секцию прогнозирования
st.sidebar.subheader("Forecasting")

# Проверяем, выполнена ли оптимизация
optimization_done = st.session_state.get('show_optimization', False)

if not optimization_done:
    st.sidebar.info("Please perform employee optimization first")

# Инициализируем session_state
if 'show_forecast' not in st.session_state:
    st.session_state.show_forecast = False
if 'forecast_month' not in st.session_state:
    st.session_state.forecast_month = None
if 'forecast_data' not in st.session_state:
    st.session_state.forecast_data = None
if 'show_optimization' not in st.session_state:
    st.session_state.show_optimization = False
if 'optimization_data' not in st.session_state:
    st.session_state.optimization_data = None

if st.sidebar.button("Create forecast", disabled=not optimization_done):
    st.session_state.show_forecast = True
    st.session_state.forecast_data = None  # Сбрасываем кэш

# Добавляем выбор одной операции
forecast_created = st.session_state.get('show_forecast', False)

if forecast_created:
    st.sidebar.subheader("Operation Analysis")
    
    operation_keys = [
        'Direct_Overloading_20', 'Cross_Docking_20', 'Direct_Overloading_40', 'Cross_Docking_40', 
        'Pallet_Direct_Overloading', 'Pallet_Cross_Docking', 'Other_revenue', 'Reloading_Service', 
        'Goods_Storage', 'Additional_Service'
    ]
    
    selected_operation = st.sidebar.selectbox(
        "Select operation for detailed analysis:",
        ["Select operation..."] + operation_keys,
        key="selected_operation"
    )
    
    # Добавляем выбор месяца для директорской аналитики
    if selected_operation != "Select operation...":
        selected_month_analysis = st.sidebar.selectbox(
            "Select month for executive dashboard:",
            ["May", "June", "July", "August", "September", "October", "November", "December"],
            key="selected_month_analysis"
        )


# Отображаем результаты оптимизации, если они есть
if st.session_state.show_optimization and st.session_state.optimization_data:
    optimized_data = st.session_state.optimization_data
    
    # Создаем секцию для результатов оптимизации
    st.subheader("🤖 AI-Powered Employee Optimization Results")
    
    # Выводим оптимизированную таблицу
    st.markdown("### Optimised number of employees:")
    
    try:
        # Пытаемся создать DataFrame из ответа AI
        import io
        # Парсим данные, полученные от AI
        lines = optimized_data.strip().split('\n')
        
        # Парсим данные как таблицу без заголовков
        # И присваиваем заголовки из исходной таблицы
        data_rows = []
        for line in lines:
            if line.strip():  # Пропускаем пустые строки
                # Разбиваем по пробелам/табуляциям
                parts = line.split()
                
                # Пропускаем строки заголовков и некорректные строки
                if (len(parts) >= 15 and 
                    not parts[0] == 'Month' and  # Пропускаем заголовок
                    not all(part.isalpha() for part in parts[:3]) and  # Пропускаем текстовые строки
                    parts[0] in ['May', 'June', 'July', 'August', 'September']):  # Проверяем, что месяц валиден
                    
                    if len(parts) >= 16:  # Полная таблица с месяцем
                        data_rows.append(parts[:len(df.columns)])
                    elif len(parts) == 15:  # Старый формат - добавляем месяц в начало
                        months = ['May', 'June', 'July', 'August', 'September']
                        month_idx = len(data_rows)
                        if month_idx < len(months):
                            parts = [months[month_idx]] + parts
                        data_rows.append(parts[:len(df.columns)])  # Берем только нужное количество колонок
        
        # Создаем DataFrame с правильными заголовками
        if data_rows:
            optimized_df = pd.DataFrame(data_rows, columns=df.columns)
        else:
            # Если не удалось распарсить, пробуем стандартный метод
            optimized_df = pd.read_csv(io.StringIO(optimized_data), sep='\s+', header=None)
            # Подгоняем количество колонок
            if len(optimized_df.columns) == len(df.columns):
                optimized_df.columns = df.columns
            else:
                # Если колонок не совпадает, берем первые N
                optimized_df = optimized_df.iloc[:, :len(df.columns)]
                optimized_df.columns = df.columns
            
        # Отображаем оптимизированную таблицу с правильным форматированием
        # Преобразуем числовые колонки
        optimized_df_display = optimized_df.copy()
        for col in optimized_df_display.columns[1:]:  # Пропускаем колонку Month
            optimized_df_display[col] = pd.to_numeric(optimized_df_display[col], errors='coerce').fillna(0).astype(int)
        
        # Конфигурация колонок
        column_config_opt = {
            "Month": st.column_config.TextColumn(
                "Month",
                width="medium",
                help="Месяц",
            )
        }
        
        for col in optimized_df_display.columns[1:]:
            column_config_opt[col] = st.column_config.NumberColumn(
                col,
                width="small",
                format="%d",
                help=f"Оптимизированное количество: {col}",
            )
        
        st.dataframe(
            optimized_df_display, 
            use_container_width=True, 
            column_config=column_config_opt,
            hide_index=True
        )
        
        # Анализируем различия
        st.markdown("### Differences analysis:")
        differences = analyze_differences(df, optimized_df)
        st.markdown(differences)
        
    except Exception as e:
        # Если не удалось распарсить как таблицу, выводим как текст
        st.code(optimized_data)
        st.markdown("### Analysis:")
        st.markdown(f"Error processing data from AI: {str(e)}")
    
    st.divider()
    
# Показываем прогноз октябрь-декабрь, если он был создан
if st.session_state.show_forecast:
    # Проверяем, нужно ли пересчитать прогноз
    if st.session_state.forecast_data is None:
        
        with st.spinner('Creating forecasts for October, November, December...'):
            # Получаем оптимизированные данные для прогноза
            base_df = df  # Исходные данные
            if st.session_state.get('optimization_data'):
                optimized_df = get_optimized_dataframe(df, st.session_state.optimization_data)
                if len(optimized_df) > 0 and len(optimized_df.columns) == len(df.columns):
                    base_df = optimized_df  # Используем оптимизированные данные
            
            # Прогнозируем все оставшиеся месяцы (каждый месяц на основе предыдущего)
            forecast_months = ["October", "November", "December"]
            forecast_data = []
            cumulative_df = base_df.copy()  # Начинаем с базовых данных
            
            for month in forecast_months:
                # Прогнозируем на основе обновленных данных
                month_forecast, _ = predict_future_operations(cumulative_df, month, use_optimized_data=False)
                if month_forecast is not None:
                    forecast_row = month_forecast.iloc[0].tolist()
                    forecast_data.append(forecast_row)
                    
                    # Добавляем прогноз к кумулятивным данным для следующего месяца
                    new_row_df = pd.DataFrame([forecast_row], columns=cumulative_df.columns)
                    cumulative_df = pd.concat([cumulative_df, new_row_df], ignore_index=True)
            
            if forecast_data and len(forecast_data) == 3:  # Убеждаемся, что у нас точно 3 месяца
                # Создаем общий DataFrame с прогнозами
                full_forecast_df = pd.DataFrame(forecast_data, columns=base_df.columns)
                
                # Проверяем, что таблица содержит точно 3 строки
                if len(full_forecast_df) == 3:
                    # Объединяем с историческими данными для графиков
                    combined_df = pd.concat([base_df, full_forecast_df], ignore_index=True)
                    
                    st.session_state.forecast_data = (full_forecast_df, combined_df)
    else:
        # Используем сохраненные данные
        full_forecast_df, combined_df = st.session_state.forecast_data
    
    if full_forecast_df is not None and len(full_forecast_df) > 0:
        # Создаем секцию для результатов прогноза
        st.subheader("📏 Forecast for October - December 2025")
        
        # Выводим общую таблицу с прогнозом на октябрь-декабрь
        st.markdown("### Forecasted operations and employee numbers (Oct-Dec 2025):")
        
        # Преобразуем числовые колонки и убеждаемся, что у нас только 3 строки
        forecast_df_display = full_forecast_df.copy()
        
        # Обрезаем до 3 строк, если больше
        if len(forecast_df_display) > 3:
            forecast_df_display = forecast_df_display.head(3)
        
        # Проверяем, что месяцы правильные
        expected_months = ["October", "November", "December"]
        if len(forecast_df_display) == 3:
            forecast_df_display[forecast_df_display.columns[0]] = expected_months
        
        for col in forecast_df_display.columns[1:]:  # Пропускаем колонку Month
            forecast_df_display[col] = pd.to_numeric(forecast_df_display[col], errors='coerce').fillna(0).astype(int)
        
        # Конфигурация колонок
        column_config_forecast = {
            "Month": st.column_config.TextColumn(
                "Month",
                width="medium",
                help="Месяц прогноза",
            )
        }
        
        for col in forecast_df_display.columns[1:]:
            column_config_forecast[col] = st.column_config.NumberColumn(
                col,
                width="small",
                format="%d",
                help=f"Прогнозное количество: {col}",
            )
        
        # Обрезаем до 3 строк окончательно
        forecast_df_final = forecast_df_display.head(3).copy()
        
        st.dataframe(
            forecast_df_final, 
            use_container_width=True, 
            column_config=column_config_forecast,
            hide_index=True
        )
        
        # Создаем основной график
        st.markdown("### Employee Numbers Trend (May - December 2025):")
        
        months_order = ["May", "June", "July", "August", "September", "October", "November", "December"]
        
        # График сотрудников (Loader, Forklift_Operator, Operation_manager)
        employee_columns = ['Operation_manager', 'Loader', 'Forklift_Operator']
        colors_emp = ['red', 'blue', 'green']
        
        fig_employees = go.Figure()
        
        for i, col in enumerate(employee_columns):
            if col in combined_df.columns:
                y_values = pd.to_numeric(combined_df[col], errors='coerce')
                
                # Разделяем на исторические и прогнозные данные
                historical_months = months_order[:len(combined_df) - len(full_forecast_df)]
                forecast_months = months_order[len(combined_df) - len(full_forecast_df):]
                
                historical_values = y_values[:len(combined_df) - len(full_forecast_df)]
                forecast_values = y_values[len(combined_df) - len(full_forecast_df):]
                
                # Исторические данные
                fig_employees.add_trace(go.Scatter(
                    x=historical_months,
                    y=historical_values,
                    mode='lines+markers',
                    name=f'{col} (historical)',
                    line=dict(color=colors_emp[i], width=3),
                    marker=dict(size=8)
                ))
                
                # Прогнозные данные
                if len(forecast_values) > 0:
                    # Соединительная линия
                    bridge_x = [historical_months[-1], forecast_months[0]]
                    bridge_y = [historical_values.iloc[-1], forecast_values.iloc[0]]
                    
                    fig_employees.add_trace(go.Scatter(
                        x=bridge_x,
                        y=bridge_y,
                        mode='lines',
                        line=dict(color=colors_emp[i], width=2, dash='dot'),
                        showlegend=False
                    ))
                    
                    fig_employees.add_trace(go.Scatter(
                        x=forecast_months,
                        y=forecast_values,
                        mode='lines+markers',
                        name=f'{col} (forecast)',
                        line=dict(color=colors_emp[i], width=3, dash='dot'),
                        marker=dict(size=8, symbol='diamond')
                    ))
        
        fig_employees.update_layout(
            title='Employee Numbers Trends (May-December 2025)',
            xaxis_title='Month',
            yaxis_title='Number of Employees',
            hovermode='x unified',
            height=500
        )
        
        st.plotly_chart(fig_employees, use_container_width=True)
        
        
        st.divider()

# Отображение анализа выбранной операции
if st.session_state.get('show_forecast') and 'selected_operation' in st.session_state:
    selected_op = st.session_state.get('selected_operation', "")
    
    if selected_op and selected_op != "Select operation..." and st.session_state.forecast_data:
        full_forecast_df, combined_df = st.session_state.forecast_data
        
        # Отображаем тренд выбранной операции
        st.header("Operation Trend Analysis")
        st.subheader(f"Trend for: {selected_op}")
        
        if selected_op in combined_df.columns:
            # Получаем данные для выбранной операции
            months_order = ["May", "June", "July", "August", "September", "October", "November", "December"]
            operation_values = pd.to_numeric(combined_df[selected_op], errors='coerce')
            
            # Разделяем на исторические и прогнозные
            hist_len = len(combined_df) - len(full_forecast_df)
            historical_months = months_order[:hist_len]
            forecast_months = months_order[hist_len:]
            historical_values = operation_values[:hist_len]
            forecast_values = operation_values[hist_len:]
            
            # График: Тренд выбранной операции по месяцам
            fig_trend = go.Figure()
            
            # Исторические данные
            fig_trend.add_trace(go.Scatter(
                x=historical_months,
                y=historical_values,
                mode='lines+markers',
                name=f'{selected_op} (Historical)',
                line=dict(color='blue', width=3),
                marker=dict(size=10)
            ))
            
            # Прогнозные данные
            if len(forecast_values) > 0:
                # Соединительная линия
                bridge_x = [historical_months[-1], forecast_months[0]]
                bridge_y = [historical_values.iloc[-1], forecast_values.iloc[0]]
                
                fig_trend.add_trace(go.Scatter(
                    x=bridge_x,
                    y=bridge_y,
                    mode='lines',
                    line=dict(color='blue', width=2, dash='dot'),
                    showlegend=False
                ))
                
                fig_trend.add_trace(go.Scatter(
                    x=forecast_months,
                    y=forecast_values,
                    mode='lines+markers',
                    name=f'{selected_op} (Forecast)',
                    line=dict(color='orange', width=3, dash='dot'),
                    marker=dict(size=10, symbol='diamond')
                ))
            
            fig_trend.update_layout(
                title=f'{selected_op} Trend (May-December 2025)',
                xaxis_title='Month',
                yaxis_title='Number of Operations',
                hovermode='x unified',
                height=500
            )
            
            st.plotly_chart(fig_trend, use_container_width=True)
        
        # Директорская аналитика по выбранному месяцу
        if 'selected_month_analysis' in st.session_state:
            selected_month = st.session_state.get('selected_month_analysis', 'May')
            
            st.divider()
            st.header("Executive Dashboard")
            st.subheader(f"Monthly Analysis: {selected_month} 2025")
            
            create_executive_dashboard(df, combined_df, selected_op, selected_month)
        
        st.divider()

