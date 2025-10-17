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
openai_client = OpenAI(api_key=env["OPENAI_API_KEY"])

# Настройка заголовка приложения
st.title("Analysis of the number of employees in the warehouse")

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

# Выводим DataFrame на главную страницу
st.subheader("Warehouse Operations and Employee Data")
st.dataframe(df)

# Модифицируем функцию для работы с OpenAI API
def optimize_employees_with_ai(df):
    """
    Функция отправляет данные в OpenAI для получения рекомендаций
    по оптимальному количеству сотрудников
    """
    # Подготавливаем данные для отправки в формате строки
    data_text = df.to_string(index=False)
    
    # Создаем улучшенный детальный промпт для OpenAI
    prompt = f"""
    Ты директор склада.
    Прими следующие исходные данные для расчета необходимого количества работников на складе.
    Ты выполняешь операции по перегрузке товаров, хранению и проведению других сопутствующих складских услуг.
    Склад - площадью 1500 квадратных метров.
    Используется в основе расчета 5-дневная рабочая неделя, 8-часовой рабочий день.
    
    Название рабочих мест:
    - Loader - грузчик для ручного труда
    - Forklift Operator - оператор погрузчика для паллетных перегрузок
    - Operation manager - работник офиса, обеспечивающий операционную поддержку всех складских процессов. ВАЖНО: Operation manager может выполнять одновременно несколько функций: Warehouse Logistics Specialist, Warehouse Shift Coordinator, Warehouse Logistics Assistant. При объемах менее 300 операций в месяц один Operation manager может совмещать все три функции. При 300-500 операциях нужно 2 Operation manager. При более 500 операциях нужно 3 Operation manager.
    - Sales_number - особа, ответственная за продажи складских услуг
    
    Все операции ты делишь на три части:
    
    1) Reloading Service - операции по ручной перегрузке товаров.
    Операции по ручной перегрузке (90%):
    - на одну операцию необходимо 4 Loader
    - время: 3 часа на одну операцию
    Операции по паллетной перегрузке (10%):
    - на одну операцию необходим 1 Forklift Operator + 1 Loader
    - время: 1 час на одну операцию
    При большом объеме работ можно выполнять 2-3 операции одновременно.
    
    2) Goods_Storage - операции по хранению груза и перемещения груза внутри склада. Этими операциями занимаются Loader и Forklift Operator.
    
    3) Additional_Services - операции по паллетизации, этикированию, стречиванию и подобным операциям. Этими операциями занимаются Operation manager.
    
    ОПТИМИЗАЦИЯ Operation manager (ОЧЕНЬ ВАЖНО):
    В исходных данных количество Operation manager завышено. При оптимизации руководствуйся следующими правилами:
    - Общие операции = Reloading_Service + Goods_Storage + Additional_Service
    - Менее 300 общих операций в месяц = 2 Operation manager
    - 300-500 общих операций в месяц = 3 Operation manager
    - Более 500 общих операций в месяц = 4 Operation manager
    Операционные работники могут совмещать функции, поэтому их количество не растет пропорционально объему операций.
    
    Твоим заданием есть предложить альтернативное оптимальное количество работников, основываясь на количестве проделанных операций в мае, июне, июле, августе и в сентябре. ОСОБОЕ ВНИМАНИЕ удели оптимизации Operation manager - уменьши их количество согласно вышеуказанным правилам.
    
    Исходные данные по складу:
    {data_text}
    
    ВАЖНО: Верни ТОЛЬКО данные таблицы в следующем формате (без заголовков):
    May 200 54 44 1 1 2 10 0
    June 237 138 76 1 1 3 9 0
    July 354 173 16 1 1 3 8 0
    August 494 252 35 1 1 7 7 1
    September 420 210 25 1 1 7 7 1
    
    Замени только числа в последних 5 колонках (количество сотрудников) на оптимизированные значения. НЕ добавляй заголовки или объяснения.
    """
    
    try:
        # Отправляем запрос к OpenAI API
        response = openai_client.chat.completions.create(
            model="gpt-4",  
            messages=[
                {"role": "system", "content": "Ты эксперт по оптимизации складских операций. Возвращай только числовые данные таблицы без дополнительных объяснений."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,  
            temperature=0.2   
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
        
        # Проверяем все колонки с числовыми данными (количество сотрудников)
        employee_columns = ['Director_number', 'Sales_number', 'Operation manager', 'Loader', 'Forklift Operator']
        
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
        for line in lines:
            if line.strip():  # Пропускаем пустые строки
                parts = line.split()
                if len(parts) >= len(original_df.columns) and not all(part.isalpha() for part in parts[:3]):
                    data_rows.append(parts[:len(original_df.columns)])
        
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
        # Используем оптимизированные данные, если они доступны
        if use_optimized_data and st.session_state.get('optimization_data'):
            df = get_optimized_dataframe(df, st.session_state.optimization_data)
        
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
        
        # Прогнозируем каждый тип операций
        predictions = {}
        operation_columns = ['Reloading_Service', 'Goods_Storage', 'Additional_Service']
        employee_columns = ['Director_number', 'Sales_number', 'Operation manager', 'Loader', 'Forklift Operator']
        
        target_month_num = month_mapping[target_month]
        X_pred = np.array([[target_month_num]])
        
        # Прогнозируем операции с учетом роста
        growth_factor = 1.0
        if target_month_num >= 10:  # Октябрь и далее
            # Постепенный рост операций: 5% в октябре, 8% в ноябре, 12% в декабре
            growth_rates = {10: 1.05, 11: 1.08, 12: 1.12}
            growth_factor = growth_rates.get(target_month_num, 1.0)
        
        for col in operation_columns:
            if col in df.columns:
                y = pd.to_numeric(df[col], errors='coerce')
                model = LinearRegression()
                model.fit(X, y)
                pred_value = model.predict(X_pred)[0]
                # Применяем коэффициент роста для операций
                pred_value_with_growth = pred_value * growth_factor
                predictions[col] = max(0, int(round(pred_value_with_growth)))
        
        # Прогнозируем количество сотрудников
        for col in employee_columns:
            if col in df.columns:
                if col == 'Operation manager':
                    # Особая логика для Operation manager
                    total_ops = sum([predictions.get(op_col, 0) for op_col in operation_columns if op_col in predictions])
                    if total_ops < 300:
                        predictions[col] = 2
                    elif total_ops <= 500:
                        predictions[col] = 3
                    else:
                        predictions[col] = 4
                else:
                    # Обычная линейная регрессия для остальных
                    y = pd.to_numeric(df[col], errors='coerce')
                    model = LinearRegression()
                    model.fit(X, y)
                    pred_value = model.predict(X_pred)[0]
                    predictions[col] = max(0, int(round(pred_value)))
        
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

# Функция для создания графиков зависимости
def create_dependency_charts(df):
    """
    Создает графики зависимости количества работников от объема операций
    """
    try:
        # Подготавливаем данные для визуализации
        operation_columns = ['Reloading_Service', 'Goods_Storage', 'Additional_Service']
        employee_columns = ['Operation manager', 'Loader', 'Forklift Operator']
        
        # Создаем суммарные операции
        total_operations = df[operation_columns].apply(pd.to_numeric, errors='coerce').sum(axis=1)
        
        # График 1: Зависимость Loader от общего объема операций
        fig1 = px.scatter(x=total_operations, y=pd.to_numeric(df['Loader'], errors='coerce'),
                         labels={'x': 'Total volume of operations', 'y': 'Number of Loaders'},
                         title='Dependence of the number of loaders on the volume of operations',
                         trendline='ols')
        fig1.update_traces(marker=dict(size=12, color='blue'))
        
        # График 2: Зависимость Operation manager от Additional_Service
        fig2 = px.scatter(x=pd.to_numeric(df['Additional_Service'], errors='coerce'),
                         y=pd.to_numeric(df['Operation manager'], errors='coerce'),
                         labels={'x': 'Additional Services', 'y': 'Number of Operation Managers'},
                         title='Operation Managers dependence on additional services',
                         trendline='ols')
        fig2.update_traces(marker=dict(size=12, color='green'))
        
        return fig1, fig2
        
    except Exception as e:
        st.error(f"Error during plot creation: {str(e)}")
        return None, None

# Создаем боковую панель (sidebar)
st.sidebar.header("Analytics Tools")

# Добавляем кнопку оптимизации
if st.sidebar.button("Employees number optimisation"):
    # Очищаем кэш прогнозов при повторном нажатии
    st.session_state.forecast_data = None
    st.session_state.show_forecast = False
    
    # Показываем индикатор загрузки
    with st.spinner('Analyzing data with AI...'):
        # Получаем оптимизированные данные от OpenAI
        optimized_data = optimize_employees_with_ai(df)
        # Сохраняем в session_state
        st.session_state.optimization_data = optimized_data
        st.session_state.show_optimization = True

# Добавляем секцию прогнозирования
st.sidebar.subheader("🔮 Forecasting")

# Проверяем, выполнена ли оптимизация
optimization_done = st.session_state.get('show_optimization', False)

if not optimization_done:
    st.sidebar.info("ℹ️ Please perform employee optimization first")

selected_month = st.sidebar.selectbox(
    "Select a month for forecasting:",
    ["October", "November", "December"],
    key="month_selector",
    disabled=not optimization_done
)

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

if st.sidebar.button("🔮 Create forecast", disabled=not optimization_done):
    st.session_state.show_forecast = True
    st.session_state.forecast_month = selected_month
    st.session_state.forecast_data = None  # Сбрасываем кэш

# Автоматически обновляем прогноз при смене месяца
if (st.session_state.show_forecast and 
    st.session_state.forecast_month != selected_month):
    st.session_state.forecast_month = selected_month
    st.session_state.forecast_data = None  # Сбрасываем кэш для пересчета

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
                # Если строка не похожа на заголовок, добавляем в данные
                if len(parts) >= len(df.columns) and not all(part.isalpha() for part in parts[:3]):
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
            
        st.dataframe(optimized_df, use_container_width=True)
        
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
    
# Показываем прогноз, если он был создан
if st.session_state.show_forecast and st.session_state.forecast_month:
    current_month = selected_month  # Используем текущий выбранный месяц
    
    # Проверяем, нужно ли пересчитать прогноз
    if (st.session_state.forecast_data is None or 
        current_month != st.session_state.get('last_calculated_month', None)):
        
        with st.spinner(f'Updating the forecast for {current_month}...'):
            # Получаем прогноз с помощью линейной регрессии
            forecast_df, predictions = predict_future_operations(df, current_month)
            st.session_state.forecast_data = (forecast_df, predictions)
            st.session_state.last_calculated_month = current_month
    else:
        # Используем сохраненные данные
        forecast_df, predictions = st.session_state.forecast_data
    
    if forecast_df is not None:
        # Создаем секцию для результатов прогноза
        st.subheader(f"📊 Forecast for {current_month} 2025")
        
        # Выводим таблицу с прогнозом
        st.markdown(f"### Forecasted operations and employee numbers for  {current_month}:")
        st.dataframe(forecast_df, use_container_width=True)
        
        # Создаем графики зависимости
        st.markdown("### Dependency charts:")
        fig1, fig2 = create_dependency_charts(df)
        
        if fig1 is not None and fig2 is not None:
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(fig1, use_container_width=True)
            with col2:
                st.plotly_chart(fig2, use_container_width=True)
        
        # Создаем трендовый график до конца 2024 года
        st.markdown("### Forecast for the rest of 2025:")
        
        # Прогнозируем все оставшиеся месяцы
        remaining_months = ["October", "November", "December"]
        forecast_data = []
        
        for month in remaining_months:
            month_forecast, _ = predict_future_operations(df, month)
            if month_forecast is not None:
                forecast_data.append(month_forecast.iloc[0].tolist())
        
        if forecast_data:
            # Объединяем исторические и прогнозные данные
            full_forecast_df = pd.DataFrame(forecast_data, columns=df.columns)
            combined_df = pd.concat([df, full_forecast_df], ignore_index=True)
            
            # Создаем линейный график тренда
            months_order = ["May", "June", "July", "August", "September", "October", "November", "December"]
            
            fig_trend = go.Figure()
            
            # Добавляем линии для ключевых показателей
            employee_columns = ['Operation manager', 'Loader', 'Forklift Operator']
            colors = ['red', 'blue', 'green']
            
            for i, col in enumerate(employee_columns):
                if col in combined_df.columns:
                    y_values = pd.to_numeric(combined_df[col], errors='coerce')
                    
                    # Разделяем на исторические и прогнозные данные
                    historical_months = months_order[:len(df)]
                    forecast_months = months_order[len(df):]
                    
                    historical_values = y_values[:len(df)]
                    forecast_values = y_values[len(df):]
                    
                    # Исторические данные (сплошная линия)
                    fig_trend.add_trace(go.Scatter(
                        x=historical_months,
                        y=historical_values,
                        mode='lines+markers',
                        name=f'{col} (факт)',
                        line=dict(color=colors[i], width=3),
                        marker=dict(size=8)
                    ))
                    
                    # Прогнозные данные (пунктирная линия)
                    if len(forecast_values) > 0:
                        # Добавляем соединительную точку
                        bridge_x = [historical_months[-1], forecast_months[0]]
                        bridge_y = [historical_values.iloc[-1], forecast_values.iloc[0]]
                        
                        fig_trend.add_trace(go.Scatter(
                            x=bridge_x,
                            y=bridge_y,
                            mode='lines',
                            line=dict(color=colors[i], width=2, dash='dot'),
                            showlegend=False
                        ))
                        
                        fig_trend.add_trace(go.Scatter(
                            x=forecast_months,
                            y=forecast_values,
                            mode='lines+markers',
                            name=f'{col} (прогноз)',
                            line=dict(color=colors[i], width=2, dash='dot'),
                            marker=dict(size=8, symbol='diamond')
                        ))
            
            fig_trend.update_layout(
                title='Trend in employee numbers until the end of 2025',
                xaxis_title='Month',
                yaxis_title='Number of employees',
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_trend, use_container_width=True)
        
        st.divider()
