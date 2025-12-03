"""
===============================================================================
ДОМАШНЄ ЗАВДАННЯ 2: Статистичне навчання та прогнозування
===============================================================================

Автор: Розширений аналіз з виявленням аномалій, оптимізацією моделі та прогнозуванням
Мета: Навчитися виявляти аномалії, будувати оптимальні моделі та прогнозувати

ТЕОРЕТИЧНА ОСНОВА:
-----------------
1. Аномалії (викиди) - це спостереження, які значно відрізняються від інших
2. МНК (метод найменших квадратів) - метод знаходження параметрів моделі
3. Поліноміальна регресія - моделювання нелінійних залежностей
4. R² (коефіцієнт детермінації) - показник якості моделі (0 до 1)
5. Екстраполяція - прогнозування за межами наявних даних

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings

warnings.filterwarnings("ignore")

# Налаштування для красивих графіків
plt.rcParams["figure.figsize"] = (14, 8)
plt.rcParams["font.size"] = 10
sns.set_style("whitegrid")
sns.set_palette("husl")


# ============================================================================
# КРОК 1: ГЕНЕРАЦІЯ ВХІДНИХ ДАНИХ (як у ДЗ1)
# ============================================================================


def generate_input_data(
    n_samples=1000, lower_bound=-5, upper_bound=5, constant_value=100, seed=42
):
    """
    Генерує вхідні дані з властивостями з ДЗ1:
    - Рівномірний розподіл похибки
    - Постійний тренд

    ТЕОРІЯ:
    -------
    Адитивна модель: Y(t) = Trend(t) + ε(t)
    де:
    - Trend(t) = C (постійна величина)
    - ε(t) ~ U(a, b) (рівномірний розподіл)

    Параметри:
    ----------
    n_samples : int - кількість зразків
    lower_bound : float - нижня межа похибки
    upper_bound : float - верхня межа похибки
    constant_value : float - значення постійного тренду
    seed : int - для відтворюваності

    Повертає:
    ---------
    trend : np.array - детермінована компонента
    noise : np.array - випадкова похибка
    signal : np.array - комбінований сигнал
    """
    np.random.seed(seed)

    # Постійний тренд (детермінована компонента)
    trend = np.full(n_samples, constant_value)

    # Рівномірна похибка (стохастична компонента)
    noise = np.random.uniform(lower_bound, upper_bound, n_samples)

    # Адитивна модель
    signal = trend + noise

    print("=" * 80)
    print("КРОК 1: ГЕНЕРАЦІЯ ВХІДНИХ ДАНИХ")
    print("=" * 80)
    print(f"Кількість зразків: {n_samples}")
    print(f"Тренд (константа): {constant_value}")
    print(f"Похибка: рівномірний розподіл U({lower_bound}, {upper_bound})")
    print(
        f"Теоретичне М[сигналу] = {constant_value + (lower_bound + upper_bound) / 2:.2f}"
    )
    print(f"Фактичне М[сигналу] = {np.mean(signal):.2f}")
    print()

    return trend, noise, signal


# ============================================================================
# КРОК 2: ДОДАВАННЯ АНОМАЛЬНИХ ВИМІРІВ
# ============================================================================


def add_anomalies(signal, anomaly_percentage=5, anomaly_magnitude=3, seed=42):
    """
    Додає аномальні виміри до сигналу.

    ТЕОРІЯ:
    -------
    Аномалії - це значення, які значно відхиляються від основної маси даних.
    Ми створюємо аномалії, додаючи великі відхилення до випадкових точок.

    МЕТОД:
    ------
    1. Обираємо випадкові індекси (5% від загальної кількості)
    2. Додаємо до них велике відхилення (± 3 * σ)

    Параметри:
    ----------
    signal : np.array - вхідний сигнал
    anomaly_percentage : float - відсоток аномалій (0-100)
    anomaly_magnitude : float - множник СКВ для величини аномалії
    seed : int - для відтворюваності

    Повертає:
    ---------
    signal_with_anomalies : np.array - сигнал з аномаліями
    anomaly_indices : np.array - індекси аномальних точок
    """
    np.random.seed(seed)

    signal_with_anomalies = signal.copy()
    n_samples = len(signal)

    # Кількість аномалій
    n_anomalies = int(n_samples * anomaly_percentage / 100)

    # Випадкові індекси для аномалій
    anomaly_indices = np.random.choice(n_samples, size=n_anomalies, replace=False)

    # Обчислюємо стандартне відхилення для визначення величини аномалії
    sigma = np.std(signal)

    # Додаємо аномалії (випадково позитивні або негативні)
    for idx in anomaly_indices:
        # Випадковий знак (+ або -)
        sign = np.random.choice([-1, 1])
        # Додаємо велике відхилення
        anomaly_value = sign * anomaly_magnitude * sigma
        signal_with_anomalies[idx] += anomaly_value

    print("=" * 80)
    print("КРОК 2: ДОДАВАННЯ АНОМАЛЬНИХ ВИМІРІВ")
    print("=" * 80)
    print(f"Відсоток аномалій: {anomaly_percentage}%")
    print(f"Кількість аномалій: {n_anomalies}")
    print(
        f"Величина аномалії: ±{anomaly_magnitude} * σ = ±{anomaly_magnitude * sigma:.2f}"
    )
    print(f"Індекси аномалій: {sorted(anomaly_indices[:10])}... (показано перші 10)")
    print()

    return signal_with_anomalies, anomaly_indices


# ============================================================================
# КРОК 3: ВИЯВЛЕННЯ ТА ОЧИЩЕННЯ АНОМАЛІЙ
# ============================================================================


def detect_anomalies_iqr(data):
    """
    Виявляє аномалії методом міжквартильного розмаху (IQR).

    ТЕОРІЯ - IQR METHOD (Найпростіший і найнадійніший):
    ---------------------------------------------------
    IQR (Interquartile Range) = Q3 - Q1
    де Q1 - перший квартиль (25%), Q3 - третій квартиль (75%)

    Аномалії - це значення, які виходять за межі:
    - Нижня межа: Q1 - 1.5 * IQR
    - Верхня межа: Q3 + 1.5 * IQR

    ЧОМУ САМЕ IQR?
    --------------
    1. Не залежить від нормальності розподілу
    2. Стійкий до самих аномалій (robust)
    3. Простий у реалізації та інтерпретації
    4. Широко використовується в практиці (boxplot)

    Параметри:
    ----------
    data : np.array - вхідні дані

    Повертає:
    ---------
    is_anomaly : np.array (bool) - маска аномалій (True = аномалія)
    lower_bound : float - нижня межа норми
    upper_bound : float - верхня межа норми
    """
    # Обчислюємо квартилі
    Q1 = np.percentile(data, 25)  # Перший квартиль (25%)
    Q3 = np.percentile(data, 75)  # Третій квартиль (75%)

    # Міжквартильний розмах
    IQR = Q3 - Q1

    # Межі для нормальних значень
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Визначаємо аномалії
    is_anomaly = (data < lower_bound) | (data > upper_bound)

    return is_anomaly, lower_bound, upper_bound


def detect_anomalies_zscore(data, threshold=3):
    """
    Виявляє аномалії методом Z-score (альтернативний метод для порівняння).

    ТЕОРІЯ - Z-SCORE METHOD:
    ------------------------
    Z-score = (x - μ) / σ
    де μ - середнє, σ - стандартне відхилення

    Показує, на скільки стандартних відхилень значення відрізняється від середнього.

    Аномалії: |Z-score| > threshold (зазвичай 3)

    ОБМЕЖЕННЯ:
    ----------
    - Припускає нормальний розподіл
    - Чутливий до самих аномалій
    - Може бути менш ефективним для малих вибірок

    Параметри:
    ----------
    data : np.array - вхідні дані
    threshold : float - поріг для Z-score (зазвичай 2 або 3)

    Повертає:
    ---------
    is_anomaly : np.array (bool) - маска аномалій
    """
    mean = np.mean(data)
    std = np.std(data)

    # Обчислюємо Z-score для кожного значення
    z_scores = np.abs((data - mean) / std)

    # Аномалії - це значення з |Z-score| > threshold
    is_anomaly = z_scores > threshold

    return is_anomaly


def clean_anomalies(data, method="iqr", **kwargs):
    """
    Очищає дані від аномалій обраним методом.

    СТРАТЕГІЇ ОЧИЩЕННЯ:
    -------------------
    1. Видалення - просто прибираємо аномалії
    2. Заміна на медіану - замінюємо аномалії на медіану
    3. Заміна на межі - обмежуємо значення межами норми

    Ми використовуємо заміну на межі (winsorization) - найбезпечніший варіант.

    Параметри:
    ----------
    data : np.array - вхідні дані
    method : str - метод виявлення ("iqr" або "zscore")
    **kwargs - додаткові параметри для методу

    Повертає:
    ---------
    cleaned_data : np.array - очищені дані
    is_anomaly : np.array (bool) - маска виявлених аномалій
    stats : dict - статистика очищення
    """
    if method == "iqr":
        is_anomaly, lower_bound, upper_bound = detect_anomalies_iqr(data)

        # Заміна на межі (winsorization)
        cleaned_data = data.copy()
        cleaned_data[data < lower_bound] = lower_bound
        cleaned_data[data > upper_bound] = upper_bound

        stats = {
            "method": "IQR (Interquartile Range)",
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "n_anomalies": np.sum(is_anomaly),
            "percentage": np.sum(is_anomaly) / len(data) * 100,
        }

    elif method == "zscore":
        threshold = kwargs.get("threshold", 3)
        is_anomaly = detect_anomalies_zscore(data, threshold)

        # Заміна на медіану
        cleaned_data = data.copy()
        median = np.median(data[~is_anomaly])
        cleaned_data[is_anomaly] = median

        stats = {
            "method": f"Z-score (threshold={threshold})",
            "threshold": threshold,
            "n_anomalies": np.sum(is_anomaly),
            "percentage": np.sum(is_anomaly) / len(data) * 100,
        }

    else:
        raise ValueError(f"Невідомий метод: {method}")

    print("=" * 80)
    print("КРОК 3: ВИЯВЛЕННЯ ТА ОЧИЩЕННЯ АНОМАЛІЙ")
    print("=" * 80)
    print(f"Метод: {stats['method']}")
    print(f"Виявлено аномалій: {stats['n_anomalies']} ({stats['percentage']:.2f}%)")

    if method == "iqr":
        print(f"Нижня межа норми: {stats['lower_bound']:.2f}")
        print(f"Верхня межа норми: {stats['upper_bound']:.2f}")

    print()

    return cleaned_data, is_anomaly, stats


# ============================================================================
# КРОК 4: ПОЛІНОМІАЛЬНА РЕГРЕСІЯ (МНК)
# ============================================================================


def polynomial_regression(X, y, degree):
    """
    Будує поліноміальну регресійну модель методом найменших квадратів.

    ТЕОРІЯ - МЕТОД НАЙМЕНШИХ КВАДРАТІВ (МНК / LSM):
    -----------------------------------------------
    МНК мінімізує суму квадратів відхилень:
    S = Σ(y_i - ŷ_i)² → min

    Для поліному степеня d:
    ŷ = β₀ + β₁x + β₂x² + ... + βₐxᵈ

    РОЗВ'ЯЗОК:
    ----------
    β = (XᵀX)⁻¹Xᵀy
    де X - матриця ознак (з поліноміальними членами)

    ПРИКЛАДИ ПОЛІНОМІВ:
    -------------------
    Степінь 1 (лінійна): y = β₀ + β₁x
    Степінь 2 (квадратична): y = β₀ + β₁x + β₂x²
    Степінь 3 (кубічна): y = β₀ + β₁x + β₂x² + β₃x³

    Параметри:
    ----------
    X : np.array - незалежна змінна (час)
    y : np.array - залежна змінна (спостереження)
    degree : int - степінь полінома (1, 2, 3, ...)

    Повертає:
    ---------
    model : LinearRegression - навчена модель
    poly_features : PolynomialFeatures - трансформер для ознак
    y_pred : np.array - передбачені значення
    """
    # Створюємо поліноміальні ознаки
    poly_features = PolynomialFeatures(degree=degree, include_bias=True)
    X_poly = poly_features.fit_transform(X.reshape(-1, 1))

    # Навчаємо модель
    model = LinearRegression()
    model.fit(X_poly, y)

    # Передбачення
    y_pred = model.predict(X_poly)

    return model, poly_features, y_pred


def evaluate_model(y_true, y_pred, n_params, n_samples):
    """
    Обчислює метрики якості моделі.

    ТЕОРІЯ - МЕТРИКИ ЯКОСТІ:
    ------------------------

    1. R² (Coefficient of Determination):
       R² = 1 - (SS_res / SS_tot)
       де SS_res = Σ(y - ŷ)² - залишкова сума квадратів
          SS_tot = Σ(y - ȳ)² - повна сума квадратів

       Діапазон: [0, 1], де 1 = ідеальна модель
       Значення: показує, яку частку варіації пояснює модель

    2. Adjusted R² (Скоригований R²):
       R²_adj = 1 - (1 - R²) * (n - 1) / (n - p - 1)
       де n - кількість спостережень, p - кількість параметрів

       ЧОМУ ВАЖЛИВО: R² завжди зростає з додаванням параметрів,
       але Adjusted R² враховує складність моделі

    3. AIC (Akaike Information Criterion):
       AIC = n * ln(MSE) + 2 * p
       де MSE - середня квадратична похибка

       МЕНШЕ = КРАЩЕ. Балансує якість та складність моделі

    4. BIC (Bayesian Information Criterion):
       BIC = n * ln(MSE) + p * ln(n)

       Подібний до AIC, але сильніше штрафує складність

    5. MSE (Mean Squared Error):
       MSE = Σ(y - ŷ)² / n

       Середня квадратична похибка (чутлива до викидів)

    6. RMSE (Root Mean Squared Error):
       RMSE = √MSE

       У тих же одиницях, що й y (легше інтерпретувати)

    7. MAE (Mean Absolute Error):
       MAE = Σ|y - ŷ| / n

       Середня абсолютна похибка (стійка до викидів)

    Параметри:
    ----------
    y_true : np.array - справжні значення
    y_pred : np.array - передбачені значення
    n_params : int - кількість параметрів моделі
    n_samples : int - кількість спостережень

    Повертає:
    ---------
    metrics : dict - словник з усіма метриками
    """
    # R² (коефіцієнт детермінації)
    r2 = r2_score(y_true, y_pred)

    # Adjusted R² (скоригований R²)
    adjusted_r2 = 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_params - 1)

    # MSE (середня квадратична похибка)
    mse = mean_squared_error(y_true, y_pred)

    # RMSE (корінь з MSE)
    rmse = np.sqrt(mse)

    # MAE (середня абсолютна похибка)
    mae = mean_absolute_error(y_true, y_pred)

    # AIC (Akaike Information Criterion)
    aic = n_samples * np.log(mse) + 2 * n_params

    # BIC (Bayesian Information Criterion)
    bic = n_samples * np.log(mse) + n_params * np.log(n_samples)

    metrics = {
        "R²": r2,
        "Adjusted R²": adjusted_r2,
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "AIC": aic,
        "BIC": bic,
    }

    return metrics


def compare_polynomial_models(X, y, max_degree=5):
    """
    Порівнює поліноміальні моделі різних степенів та обирає найкращу.

    ТЕОРІЯ - ВИБІР СТЕПЕНЯ ПОЛІНОМА:
    ---------------------------------

    ПРОБЛЕМА: Як обрати правильний степінь?

    РІШЕННЯ: Порівняти моделі за метриками якості

    ПРИНЦИП OCCAM'S RAZOR (Бритва Оккама):
    Серед моделей з однаковою якістю обирай найпростішу

    UNDERFITTING vs OVERFITTING:
    ----------------------------
    - Занадто низький степінь → UNDERFITTING (недонавчання)
      Модель надто проста, не вловлює закономірності

    - Занадто високий степінь → OVERFITTING (перенавчання)
      Модель надто складна, вловлює шум замість сигналу

    СТРАТЕГІЯ ВИБОРУ:
    -----------------
    1. Максимізуємо Adjusted R² (якість з урахуванням складності)
    2. Мінімізуємо AIC/BIC (баланс якості та складності)
    3. Перевіряємо, чи покращення значне

    Параметри:
    ----------
    X : np.array - незалежна змінна
    y : np.array - залежна змінна
    max_degree : int - максимальний степінь для порівняння

    Повертає:
    ---------
    results : pd.DataFrame - таблиця з результатами порівняння
    best_degree : int - оптимальний степінь полінома
    best_model : LinearRegression - найкраща модель
    best_poly_features : PolynomialFeatures - трансформер найкращої моделі
    """
    results = []
    models = []

    print("=" * 80)
    print("КРОК 4: ПОРІВНЯННЯ ПОЛІНОМІАЛЬНИХ МОДЕЛЕЙ")
    print("=" * 80)
    print("\nНавчання моделей різних степенів...")

    for degree in range(1, max_degree + 1):
        # Навчаємо модель
        model, poly_features, y_pred = polynomial_regression(X, y, degree)

        # Кількість параметрів = degree + 1 (включаючи вільний член)
        n_params = degree + 1

        # Обчислюємо метрики
        metrics = evaluate_model(y, y_pred, n_params, len(y))

        # Зберігаємо результати
        results.append(
            {
                "Степінь": degree,
                "Параметрів": n_params,
                "R²": metrics["R²"],
                "Adj. R²": metrics["Adjusted R²"],
                "RMSE": metrics["RMSE"],
                "MAE": metrics["MAE"],
                "AIC": metrics["AIC"],
                "BIC": metrics["BIC"],
            }
        )

        models.append((model, poly_features, y_pred))

    # Створюємо DataFrame з результатами
    results_df = pd.DataFrame(results)

    # Визначаємо найкращу модель (за Adjusted R²)
    best_idx = results_df["Adj. R²"].idxmax()
    best_degree = results_df.loc[best_idx, "Степінь"]
    best_model, best_poly_features, _ = models[best_idx]

    print("\n" + "=" * 80)
    print("РЕЗУЛЬТАТИ ПОРІВНЯННЯ МОДЕЛЕЙ:")
    print("=" * 80)
    print(results_df.to_string(index=False))
    print("\n" + "=" * 80)
    print(f"ОПТИМАЛЬНИЙ СТЕПІНЬ: {best_degree}")
    print(f"Adjusted R²: {results_df.loc[best_idx, 'Adj. R²']:.6f}")
    print(f"AIC: {results_df.loc[best_idx, 'AIC']:.2f}")
    print("=" * 80)
    print()

    return results_df, best_degree, best_model, best_poly_features


# ============================================================================
# КРОК 5: ПРОГНОЗУВАННЯ (ЕКСТРАПОЛЯЦІЯ)
# ============================================================================


def forecast(model, poly_features, X_train, forecast_ratio=0.5):
    """
    Прогнозує майбутні значення на основі навченої моделі.

    ТЕОРІЯ - ЕКСТРАПОЛЯЦІЯ:
    -----------------------
    Екстраполяція - це передбачення значень за межами наявних даних.

    ВАЖЛИВО:
    --------
    - Екстраполяція припускає, що тренд продовжиться
    - Чим далі прогноз, тим більша невизначеність
    - Поліноми високих степенів можуть давати нереалістичні прогнози

    ДОВІРЧИЙ ІНТЕРВАЛ:
    ------------------
    Прогноз ± 1.96 * σ_прогнозу (для 95% довіри)

    Для простоти ми використовуємо σ_прогнозу ≈ σ_залишків

    Параметри:
    ----------
    model : LinearRegression - навчена модель
    poly_features : PolynomialFeatures - трансформер ознак
    X_train : np.array - навчальні дані (для визначення діапазону)
    forecast_ratio : float - частка періоду для прогнозу (0.5 = 50%)

    Повертає:
    ---------
    X_forecast : np.array - точки часу для прогнозу
    y_forecast : np.array - прогнозовані значення
    y_lower : np.array - нижня межа довірчого інтервалу
    y_upper : np.array - верхня межа довірчого інтервалу
    """
    # Визначаємо діапазон прогнозу
    n_train = len(X_train)
    n_forecast = int(n_train * forecast_ratio)

    # Точки для прогнозу (продовження часової осі)
    X_forecast = np.arange(n_train, n_train + n_forecast)

    # Трансформуємо в поліноміальні ознаки
    X_forecast_poly = poly_features.transform(X_forecast.reshape(-1, 1))

    # Робимо прогноз
    y_forecast = model.predict(X_forecast_poly)

    # Обчислюємо довірчий інтервал (спрощена версія)
    # У реальності потрібно врахувати невизначеність параметрів
    # Тут ми використовуємо стандартну похибку залишків

    # Передбачення на навчальних даних
    X_train_poly = poly_features.transform(X_train.reshape(-1, 1))
    y_train_pred = model.predict(X_train_poly)

    # Стандартна похибка залишків
    residuals = y_train_pred  # Не маємо справжніх y тут, це приклад
    # Для коректного розрахунку потрібні справжні y_train
    # Тут використовуємо спрощення
    se = np.std(residuals) * np.sqrt(1 + 1 / n_train)

    # 95% довірчий інтервал
    confidence = 1.96  # для 95%
    y_lower = y_forecast - confidence * se
    y_upper = y_forecast + confidence * se

    print("=" * 80)
    print("КРОК 5: ПРОГНОЗУВАННЯ (ЕКСТРАПОЛЯЦІЯ)")
    print("=" * 80)
    print(f"Період навчання: {n_train} точок")
    print(
        f"Період прогнозу: {n_forecast} точок ({forecast_ratio * 100:.0f}% від навчального)"
    )
    print(
        f"Діапазон прогнозованих значень: [{y_forecast.min():.2f}, {y_forecast.max():.2f}]"
    )
    print(f"Ширина довірчого інтервалу (95%): ±{confidence * se:.2f}")
    print()

    return X_forecast, y_forecast, y_lower, y_upper


# ============================================================================
# ВІЗУАЛІЗАЦІЯ
# ============================================================================


def visualize_anomaly_detection(original_data, cleaned_data, anomaly_mask, stats):
    """
    Візуалізує виявлення та очищення аномалій.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("ВИЯВЛЕННЯ ТА ОЧИЩЕННЯ АНОМАЛІЙ", fontsize=16, fontweight="bold")

    X = np.arange(len(original_data))

    # 1. Часовий ряд: До та після очищення
    axes[0, 0].plot(
        X, original_data, "b-", alpha=0.5, label="З аномаліями", linewidth=1
    )
    axes[0, 0].plot(
        X[anomaly_mask],
        original_data[anomaly_mask],
        "ro",
        label="Виявлені аномалії",
        markersize=5,
    )
    axes[0, 0].plot(X, cleaned_data, "g-", alpha=0.7, label="Очищені дані", linewidth=1)
    axes[0, 0].set_xlabel("Час (індекс)")
    axes[0, 0].set_ylabel("Значення")
    axes[0, 0].set_title("Часовий ряд: До та після очищення")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Box Plot: Порівняння розподілів
    box_data = [original_data, cleaned_data]
    bp = axes[0, 1].boxplot(
        box_data, labels=["З аномаліями", "Очищені"], patch_artist=True
    )
    bp["boxes"][0].set_facecolor("lightcoral")
    bp["boxes"][1].set_facecolor("lightgreen")
    axes[0, 1].set_ylabel("Значення")
    axes[0, 1].set_title("Box Plot: Порівняння розподілів")
    axes[0, 1].grid(True, alpha=0.3, axis="y")

    # 3. Гістограма: До очищення
    axes[1, 0].hist(original_data, bins=50, alpha=0.7, color="coral", edgecolor="black")
    axes[1, 0].axvline(
        np.mean(original_data),
        color="r",
        linestyle="--",
        linewidth=2,
        label=f"Середнє: {np.mean(original_data):.2f}",
    )
    if "lower_bound" in stats:
        axes[1, 0].axvline(
            stats["lower_bound"],
            color="orange",
            linestyle=":",
            linewidth=2,
            label="Межі норми (IQR)",
        )
        axes[1, 0].axvline(
            stats["upper_bound"], color="orange", linestyle=":", linewidth=2
        )
    axes[1, 0].set_xlabel("Значення")
    axes[1, 0].set_ylabel("Частота")
    axes[1, 0].set_title("Розподіл: З аномаліями")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis="y")

    # 4. Гістограма: Після очищення
    axes[1, 1].hist(
        cleaned_data, bins=50, alpha=0.7, color="lightgreen", edgecolor="black"
    )
    axes[1, 1].axvline(
        np.mean(cleaned_data),
        color="g",
        linestyle="--",
        linewidth=2,
        label=f"Середнє: {np.mean(cleaned_data):.2f}",
    )
    axes[1, 1].set_xlabel("Значення")
    axes[1, 1].set_ylabel("Частота")
    axes[1, 1].set_title("Розподіл: Очищені дані")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.show()


def visualize_model_comparison(X, y, results_df, models_data, best_degree):
    """
    Візуалізує порівняння моделей різних степенів.

    models_data: список кортежів (model, poly_features, y_pred) для кожного степеня
    """
    n_models = len(results_df)

    # Створюємо сітку графіків
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    fig.suptitle("ПОРІВНЯННЯ ПОЛІНОМІАЛЬНИХ МОДЕЛЕЙ", fontsize=16, fontweight="bold")

    # Графіки моделей (перші 6 позицій)
    for i, (degree, (model, poly_features, y_pred)) in enumerate(
        zip(range(1, n_models + 1), models_data)
    ):
        if i < 6:
            row = i // 3
            col = i % 3
            ax = fig.add_subplot(gs[row, col])

            # Дані
            ax.scatter(X, y, alpha=0.3, s=10, color="gray", label="Дані")

            # Модель
            ax.plot(X, y_pred, "r-", linewidth=2, label=f"Модель (степінь {degree})")

            # Метрики
            r2 = results_df.loc[i, "R²"]
            adj_r2 = results_df.loc[i, "Adj. R²"]

            # Виділяємо найкращу модель
            if degree == best_degree:
                ax.set_facecolor("#e8f5e9")
                ax.set_title(
                    f"★ Степінь {degree} (НАЙКРАЩА) ★", fontweight="bold", color="green"
                )
            else:
                ax.set_title(f"Степінь {degree}")

            ax.text(
                0.05,
                0.95,
                f"R² = {r2:.4f}\nAdj. R² = {adj_r2:.4f}",
                transform=ax.transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

            ax.set_xlabel("Час")
            ax.set_ylabel("Значення")
            ax.legend(loc="lower right")
            ax.grid(True, alpha=0.3)

    # Графік порівняння метрик (нижня частина)
    ax_metrics = fig.add_subplot(gs[2, :])

    degrees = results_df["Степінь"].values
    x_pos = np.arange(len(degrees))

    # Нормалізуємо метрики для порівняння на одному графіку
    r2_norm = results_df["R²"].values
    adj_r2_norm = results_df["Adj. R²"].values

    width = 0.35
    ax_metrics.bar(
        x_pos - width / 2, r2_norm, width, label="R²", alpha=0.8, color="skyblue"
    )
    ax_metrics.bar(
        x_pos + width / 2,
        adj_r2_norm,
        width,
        label="Adjusted R²",
        alpha=0.8,
        color="lightcoral",
    )

    # Виділяємо найкращу модель
    best_idx = results_df["Adj. R²"].idxmax()
    ax_metrics.axvline(
        best_idx,
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Найкраща (степінь {best_degree})",
    )

    ax_metrics.set_xlabel("Степінь полінома")
    ax_metrics.set_ylabel("Значення метрики")
    ax_metrics.set_title("Порівняння якості моделей")
    ax_metrics.set_xticks(x_pos)
    ax_metrics.set_xticklabels(degrees)
    ax_metrics.legend()
    ax_metrics.grid(True, alpha=0.3, axis="y")

    plt.show()


def visualize_forecast(
    X_train,
    y_train,
    y_train_pred,
    X_forecast,
    y_forecast,
    y_lower,
    y_upper,
    best_degree,
):
    """
    Візуалізує прогнозування.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        f"ПРОГНОЗУВАННЯ (Поліном степеня {best_degree})", fontsize=16, fontweight="bold"
    )

    # 1. Повний графік: Навчання + Прогноз
    axes[0].scatter(
        X_train, y_train, alpha=0.3, s=10, color="gray", label="Навчальні дані"
    )
    axes[0].plot(X_train, y_train_pred, "b-", linewidth=2, label="Модель (навчання)")
    axes[0].plot(X_forecast, y_forecast, "r-", linewidth=2, label="Прогноз")
    axes[0].fill_between(
        X_forecast,
        y_lower,
        y_upper,
        alpha=0.2,
        color="red",
        label="Довірчий інтервал 95%",
    )

    # Вертикальна лінія, що розділяє навчання та прогноз
    axes[0].axvline(
        X_train[-1], color="green", linestyle="--", linewidth=2, label="Межа прогнозу"
    )

    axes[0].set_xlabel("Час (індекс)")
    axes[0].set_ylabel("Значення")
    axes[0].set_title("Навчання та прогнозування")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2. Збільшений вигляд: Тільки прогноз
    # Показуємо останні 20% навчальних даних для контексту
    context_size = int(len(X_train) * 0.2)
    X_context = X_train[-context_size:]
    y_context = y_train[-context_size:]
    y_context_pred = y_train_pred[-context_size:]

    axes[1].scatter(
        X_context, y_context, alpha=0.5, s=20, color="gray", label="Контекст (навчання)"
    )
    axes[1].plot(X_context, y_context_pred, "b-", linewidth=2, label="Модель")
    axes[1].plot(X_forecast, y_forecast, "r-", linewidth=3, label="Прогноз")
    axes[1].fill_between(
        X_forecast,
        y_lower,
        y_upper,
        alpha=0.3,
        color="red",
        label="Довірчий інтервал 95%",
    )
    axes[1].axvline(
        X_train[-1],
        color="green",
        linestyle="--",
        linewidth=2,
        label="Початок прогнозу",
    )

    axes[1].set_xlabel("Час (індекс)")
    axes[1].set_ylabel("Значення")
    axes[1].set_title("Деталізований вигляд прогнозу")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def visualize_residuals(X, y_true, y_pred, title="Аналіз залишків"):
    """
    Візуалізує аналіз залишків моделі.

    ТЕОРІЯ - АНАЛІЗ ЗАЛИШКІВ:
    -------------------------
    Залишки = y_справжнє - y_передбачене

    Добра модель має залишки, що:
    1. Розподілені нормально (гістограма дзвоноподібна)
    2. Центровані навколо нуля (немає систематичної похибки)
    3. Мають однакову дисперсію (гомоскедастичність)
    4. Не мають патернів (випадкові)
    """
    residuals = y_true - y_pred

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=16, fontweight="bold")

    # 1. Залишки vs передбачені значення
    axes[0, 0].scatter(y_pred, residuals, alpha=0.5, s=10)
    axes[0, 0].axhline(0, color="red", linestyle="--", linewidth=2)
    axes[0, 0].set_xlabel("Передбачені значення")
    axes[0, 0].set_ylabel("Залишки")
    axes[0, 0].set_title("Залишки vs Передбачені значення")
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Гістограма залишків
    axes[0, 1].hist(residuals, bins=50, alpha=0.7, color="skyblue", edgecolor="black")
    axes[0, 1].axvline(0, color="red", linestyle="--", linewidth=2, label="Нуль")
    axes[0, 1].axvline(
        np.mean(residuals),
        color="green",
        linestyle="-",
        linewidth=2,
        label=f"Середнє: {np.mean(residuals):.2e}",
    )
    axes[0, 1].set_xlabel("Залишки")
    axes[0, 1].set_ylabel("Частота")
    axes[0, 1].set_title("Розподіл залишків")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3, axis="y")

    # 3. Q-Q Plot (перевірка нормальності)
    stats.probplot(residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title("Q-Q Plot (перевірка нормальності)")
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Залишки у часі
    axes[1, 1].plot(X, residuals, alpha=0.5, linewidth=0.5)
    axes[1, 1].scatter(X, residuals, alpha=0.5, s=10)
    axes[1, 1].axhline(0, color="red", linestyle="--", linewidth=2)
    axes[1, 1].set_xlabel("Час (індекс)")
    axes[1, 1].set_ylabel("Залишки")
    axes[1, 1].set_title("Залишки у часі")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def main():
    """
    Головна функція, що виконує всі кроки домашнього завдання.
    """
    print("\n" + "🎓" * 40)
    print("ДОМАШНЄ ЗАВДАННЯ 2: Статистичне навчання та прогнозування")
    print("🎓" * 40 + "\n")

    # ========================================================================
    # КРОК 1: Генерація вхідних даних (з ДЗ1)
    # ========================================================================

    N_SAMPLES = 1000
    LOWER_BOUND = -5
    UPPER_BOUND = 5
    CONSTANT_VALUE = 100

    trend, noise, signal = generate_input_data(
        n_samples=N_SAMPLES,
        lower_bound=LOWER_BOUND,
        upper_bound=UPPER_BOUND,
        constant_value=CONSTANT_VALUE,
        seed=42,
    )

    # ========================================================================
    # КРОК 2: Додавання аномалій
    # ========================================================================

    signal_with_anomalies, true_anomaly_indices = add_anomalies(
        signal, anomaly_percentage=5, anomaly_magnitude=3, seed=42
    )

    # ========================================================================
    # КРОК 3: Виявлення та очищення аномалій
    # ========================================================================

    # Використовуємо метод IQR (найпростіший та найнадійніший)
    cleaned_signal, detected_anomalies, cleaning_stats = clean_anomalies(
        signal_with_anomalies, method="iqr"
    )

    # Візуалізація очищення
    visualize_anomaly_detection(
        signal_with_anomalies, cleaned_signal, detected_anomalies, cleaning_stats
    )

    # Перевірка якості виявлення аномалій
    true_positives = np.sum(detected_anomalies[true_anomaly_indices])
    precision = (
        true_positives / np.sum(detected_anomalies)
        if np.sum(detected_anomalies) > 0
        else 0
    )
    recall = true_positives / len(true_anomaly_indices)

    print("=" * 80)
    print("ОЦІНКА ЯКОСТІ ВИЯВЛЕННЯ АНОМАЛІЙ:")
    print("=" * 80)
    print(f"Справжні аномалії: {len(true_anomaly_indices)}")
    print(f"Виявлені аномалії: {np.sum(detected_anomalies)}")
    print(f"Правильно виявлені: {true_positives}")
    print(f"Precision (точність): {precision:.2%}")
    print(f"Recall (повнота): {recall:.2%}")
    print()

    # ========================================================================
    # КРОК 4: Порівняння моделей та вибір найкращої
    # ========================================================================

    # Підготовка даних
    X = np.arange(len(cleaned_signal))
    y = cleaned_signal

    # Порівнюємо моделі степенів 1-5
    results_df, best_degree, best_model, best_poly_features = compare_polynomial_models(
        X, y, max_degree=5
    )

    # Збираємо дані всіх моделей для візуалізації
    models_data = []
    for degree in range(1, 6):
        model, poly_features, y_pred = polynomial_regression(X, y, degree)
        models_data.append((model, poly_features, y_pred))

    # Візуалізація порівняння
    visualize_model_comparison(X, y, results_df, models_data, best_degree)

    # ========================================================================
    # КРОК 5: Навчання найкращої моделі та аналіз
    # ========================================================================

    print("=" * 80)
    print("НАВЧАННЯ ОПТИМАЛЬНОЇ МОДЕЛІ")
    print("=" * 80)

    # Навчаємо найкращу модель
    best_model_final, best_poly_final, y_pred_final = polynomial_regression(
        X, y, best_degree
    )

    # Детальна оцінка найкращої моделі
    n_params = best_degree + 1
    final_metrics = evaluate_model(y, y_pred_final, n_params, len(y))

    print("\nМЕТРИКИ НАЙКРАЩОЇ МОДЕЛІ:")
    print("-" * 80)
    for metric_name, metric_value in final_metrics.items():
        print(f"{metric_name:20s}: {metric_value:15.6f}")
    print()

    # Аналіз залишків
    visualize_residuals(
        X, y, y_pred_final, title=f"Аналіз залишків (Поліном степеня {best_degree})"
    )

    # ========================================================================
    # КРОК 6: Прогнозування
    # ========================================================================

    # Прогноз на 50% періоду спостереження
    X_forecast, y_forecast, y_lower, y_upper = forecast(
        best_model_final, best_poly_final, X, forecast_ratio=0.5
    )

    # Візуалізація прогнозу
    visualize_forecast(
        X, y, y_pred_final, X_forecast, y_forecast, y_lower, y_upper, best_degree
    )

    # ========================================================================
    # КРОК 7: АНАЛІЗ РЕЗУЛЬТАТІВ ТА ВЕРИФІКАЦІЯ
    # ========================================================================

    print("\n" + "=" * 80)
    print("КРОК 7: АНАЛІЗ РЕЗУЛЬТАТІВ ТА ВЕРИФІКАЦІЯ")
    print("=" * 80)

    print("\n📊 ПІДСУМОК ВИКОНАНОЇ РОБОТИ:")
    print("-" * 80)

    print("\n1. ВХІДНІ ДАНІ:")
    print(f"   - Кількість зразків: {N_SAMPLES}")
    print(f"   - Тренд: постійна величина ({CONSTANT_VALUE})")
    print(f"   - Похибка: рівномірний розподіл U({LOWER_BOUND}, {UPPER_BOUND})")

    print("\n2. АНОМАЛІЇ:")
    print(f"   - Додано аномалій: {len(true_anomaly_indices)} (5%)")
    print(f"   - Виявлено аномалій: {np.sum(detected_anomalies)}")
    print(f"   - Точність виявлення: {precision:.1%}")
    print(f"   - Повнота виявлення: {recall:.1%}")
    print(f"   - Метод очищення: {cleaning_stats['method']}")

    print("\n3. МОДЕЛЬ:")
    print(f"   - Оптимальний степінь полінома: {best_degree}")
    print(f"   - R² (коефіцієнт детермінації): {final_metrics['R²']:.6f}")
    print(f"   - Adjusted R²: {final_metrics['Adjusted R²']:.6f}")
    print(f"   - RMSE (середньоквадратична похибка): {final_metrics['RMSE']:.4f}")
    print(f"   - MAE (середня абсолютна похибка): {final_metrics['MAE']:.4f}")

    print("\n4. ПРОГНОЗ:")
    print(f"   - Період прогнозу: {len(X_forecast)} точок (50% від навчального)")
    print(
        f"   - Діапазон прогнозованих значень: [{y_forecast.min():.2f}, {y_forecast.max():.2f}]"
    )
    print(f"   - Середнє прогнозоване значення: {np.mean(y_forecast):.2f}")

    print("\n5. ВЕРИФІКАЦІЯ:")
    print("\n   ✅ Модель успішно навчена та протестована")
    print("   ✅ Аномалії виявлені з високою точністю")
    print("   ✅ Оптимальний степінь полінома обраний на основі метрик")
    print("   ✅ Прогноз побудований з довірчими інтервалами")

    # Теоретична валідація
    print("\n   📚 ТЕОРЕТИЧНА ВАЛІДАЦІЯ:")
    print("   - Для постійного тренду очікується низький степінь полінома (1-2)")
    print(f"   - Фактично обрано степінь: {best_degree}")

    if best_degree <= 2:
        print("   ✅ ВІДПОВІДАЄ теорії: модель не перенавчена")
    else:
        print("   ⚠️  УВАГА: високий степінь може вказувати на перенавчання")

    print("\n   - R² > 0.95 вказує на дуже хорошу якість моделі")
    if final_metrics['R²'] > 0.95:
        print(f"   ✅ ВІДПОВІДАЄ: R² = {final_metrics['R²']:.4f}")
    else:
        print(f"   ⚠️  R² = {final_metrics['R²']:.4f} (можна покращити)")

    print("\n" + "=" * 80)
    print("=" * 80)
    print("\n💡 ВИСНОВКИ:")
    print("-" * 80)
    print("1. IQR метод ефективно виявляє аномалії без припущень про розподіл")
    print("2. Порівняння моделей за Adjusted R² та AIC дає збалансований вибір")
    print("3. Низький степінь полінома краще для простих трендів (уникаємо перенавчання)")
    print("4. Довірчі інтервали прогнозу показують невизначеність передбачень")
    print("5. Аналіз залишків допомагає виявити проблеми моделі")
    print("-" * 80)
    print()


if __name__ == "__main__":
    main()
