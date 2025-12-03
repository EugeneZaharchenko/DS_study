"""
Домашнє завдання 2: Статистичне навчання та прогнозування
Виконав: Євген Захарченко
Варіант 1: Рівномірний розподіл похибки, постійний тренд
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

plt.rcParams["figure.figsize"] = (14, 8)
plt.rcParams["font.size"] = 10
sns.set_style("whitegrid")


# ============================================================================
# 1. Генерація вхідних даних
# ============================================================================


def generate_input_data(
    n_samples=1000, lower_bound=-5, upper_bound=5, constant_value=100, seed=42
):
    np.random.seed(seed)
    trend = np.full(n_samples, constant_value)
    noise = np.random.uniform(lower_bound, upper_bound, n_samples)
    signal = trend + noise

    print("=" * 80)
    print("КРОК 1: ГЕНЕРАЦІЯ ВХІДНИХ ДАНИХ")
    print("=" * 80)
    print(f"Кількість зразків: {n_samples}")
    print(f"Тренд: {constant_value}, Похибка: U({lower_bound}, {upper_bound})")
    print(f"Середнє значення сигналу: {np.mean(signal):.2f}\n")

    return trend, noise, signal


# ============================================================================
# 2. Додавання аномальних вимірів
# ============================================================================


def add_anomalies(signal, anomaly_percentage=5, anomaly_magnitude=3, seed=42):
    np.random.seed(seed)
    signal_with_anomalies = signal.copy()
    n_samples = len(signal)
    n_anomalies = int(n_samples * anomaly_percentage / 100)
    anomaly_indices = np.random.choice(n_samples, size=n_anomalies, replace=False)
    sigma = np.std(signal)

    for idx in anomaly_indices:
        sign = np.random.choice([-1, 1])
        anomaly_value = sign * anomaly_magnitude * sigma
        signal_with_anomalies[idx] += anomaly_value

    print("=" * 80)
    print("КРОК 2: ДОДАВАННЯ АНОМАЛЬНИХ ВИМІРІВ")
    print("=" * 80)
    print(f"Додано аномалій: {n_anomalies} ({anomaly_percentage}%)")
    print(f"Величина аномалії: ±{anomaly_magnitude * sigma:.2f}\n")

    return signal_with_anomalies, anomaly_indices


# ============================================================================
# 3. Виявлення та очищення аномалій (метод IQR)
# ============================================================================


def detect_anomalies_iqr(data):
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    is_anomaly = (data < lower_bound) | (data > upper_bound)

    return is_anomaly, lower_bound, upper_bound


def clean_anomalies(data):
    is_anomaly, lower_bound, upper_bound = detect_anomalies_iqr(data)
    cleaned_data = data.copy()
    cleaned_data[data < lower_bound] = lower_bound
    cleaned_data[data > upper_bound] = upper_bound

    n_anomalies = np.sum(is_anomaly)
    percentage = n_anomalies / len(data) * 100

    print("=" * 80)
    print("КРОК 3: ОЧИЩЕННЯ АНОМАЛІЙ (МЕТОД IQR)")
    print("=" * 80)
    print(f"Виявлено аномалій: {n_anomalies} ({percentage:.2f}%)")
    print(f"Межі норми: [{lower_bound:.2f}, {upper_bound:.2f}]\n")

    return cleaned_data, is_anomaly


# ============================================================================
# 4. Поліноміальна регресія (МНК)
# ============================================================================


def polynomial_regression(X, y, degree):
    poly_features = PolynomialFeatures(degree=degree, include_bias=True)
    X_poly = poly_features.fit_transform(X.reshape(-1, 1))
    model = LinearRegression()
    model.fit(X_poly, y)
    y_pred = model.predict(X_poly)

    return model, poly_features, y_pred


def evaluate_model(y_true, y_pred, n_params, n_samples):
    r2 = r2_score(y_true, y_pred)
    adjusted_r2 = 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_params - 1)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    aic = n_samples * np.log(mse) + 2 * n_params
    bic = n_samples * np.log(mse) + n_params * np.log(n_samples)

    return {
        "R²": r2,
        "Adjusted R²": adjusted_r2,
        "RMSE": rmse,
        "MAE": mae,
        "AIC": aic,
        "BIC": bic,
    }


def compare_polynomial_models(X, y, max_degree=5):
    results = []
    models = []

    print("=" * 80)
    print("КРОК 4: ПОРІВНЯННЯ ПОЛІНОМІАЛЬНИХ МОДЕЛЕЙ")
    print("=" * 80)

    for degree in range(1, max_degree + 1):
        model, poly_features, y_pred = polynomial_regression(X, y, degree)
        n_params = degree + 1
        metrics = evaluate_model(y, y_pred, n_params, len(y))

        results.append(
            {
                "Степінь": degree,
                "R²": metrics["R²"],
                "Adj. R²": metrics["Adjusted R²"],
                "RMSE": metrics["RMSE"],
                "AIC": metrics["AIC"],
                "BIC": metrics["BIC"],
            }
        )

        models.append((model, poly_features, y_pred))

    results_df = pd.DataFrame(results)
    best_idx = results_df["Adj. R²"].idxmax()
    best_degree = results_df.loc[best_idx, "Степінь"]
    best_model, best_poly_features, _ = models[best_idx]

    print("\n" + results_df.to_string(index=False))
    print(f"\nОптимальний степінь: {best_degree}")
    print(f"Adjusted R²: {results_df.loc[best_idx, 'Adj. R²']:.6f}\n")

    return results_df, best_degree, best_model, best_poly_features, models


# ============================================================================
# 5. Прогнозування (екстраполяція)
# ============================================================================


def forecast(model, poly_features, X_train, y_train, forecast_ratio=0.5):
    n_train = len(X_train)
    n_forecast = int(n_train * forecast_ratio)
    X_forecast = np.arange(n_train, n_train + n_forecast)
    X_forecast_poly = poly_features.transform(X_forecast.reshape(-1, 1))
    y_forecast = model.predict(X_forecast_poly)

    X_train_poly = poly_features.transform(X_train.reshape(-1, 1))
    y_train_pred = model.predict(X_train_poly)
    residuals = y_train - y_train_pred
    se = np.std(residuals) * np.sqrt(1 + 1 / n_train)

    confidence = 1.96
    y_lower = y_forecast - confidence * se
    y_upper = y_forecast + confidence * se

    print("=" * 80)
    print("КРОК 5: ПРОГНОЗУВАННЯ")
    print("=" * 80)
    print(f"Період прогнозу: {n_forecast} точок ({forecast_ratio * 100:.0f}%)")
    print(f"Діапазон прогнозу: [{y_forecast.min():.2f}, {y_forecast.max():.2f}]")
    print(f"Довірчий інтервал (95%): ±{confidence * se:.2f}\n")

    return X_forecast, y_forecast, y_lower, y_upper


# ============================================================================
# Візуалізація
# ============================================================================


def visualize_anomaly_detection(original, cleaned, anomaly_mask):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Виявлення та очищення аномалій", fontsize=14, fontweight="bold")

    X = np.arange(len(original))

    axes[0, 0].plot(X, original, "b-", alpha=0.5, linewidth=1)
    axes[0, 0].plot(X[anomaly_mask], original[anomaly_mask], "ro", markersize=5)
    axes[0, 0].plot(X, cleaned, "g-", alpha=0.7, linewidth=1)
    axes[0, 0].set_title("Часовий ряд")
    axes[0, 0].legend(["З аномаліями", "Виявлені", "Очищені"])
    axes[0, 0].grid(True, alpha=0.3)

    bp = axes[0, 1].boxplot(
        [original, cleaned], labels=["З аномаліями", "Очищені"], patch_artist=True
    )
    bp["boxes"][0].set_facecolor("lightcoral")
    bp["boxes"][1].set_facecolor("lightgreen")
    axes[0, 1].set_title("Box Plot")
    axes[0, 1].grid(True, alpha=0.3, axis="y")

    axes[1, 0].hist(original, bins=50, alpha=0.7, color="coral", edgecolor="black")
    axes[1, 0].set_title("Розподіл (з аномаліями)")
    axes[1, 0].grid(True, alpha=0.3, axis="y")

    axes[1, 1].hist(cleaned, bins=50, alpha=0.7, color="lightgreen", edgecolor="black")
    axes[1, 1].set_title("Розподіл (очищені)")
    axes[1, 1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.show()


def visualize_models(X, y, results_df, models_data, best_degree):
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    fig.suptitle("Порівняння поліноміальних моделей", fontsize=14, fontweight="bold")

    for i, (degree, (model, poly_features, y_pred)) in enumerate(
        zip(range(1, 6), models_data)
    ):
        if i < 6:
            ax = fig.add_subplot(gs[i // 3, i % 3])
            ax.scatter(X, y, alpha=0.3, s=10, color="gray")
            ax.plot(X, y_pred, "r-", linewidth=2)

            if degree == best_degree:
                ax.set_facecolor("#e8f5e9")
                ax.set_title(f"★ Степінь {degree} ★", fontweight="bold", color="green")
            else:
                ax.set_title(f"Степінь {degree}")

            r2 = results_df.loc[i, "R²"]
            ax.text(
                0.05,
                0.95,
                f"R² = {r2:.4f}",
                transform=ax.transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", alpha=0.5),
            )
            ax.grid(True, alpha=0.3)

    ax_metrics = fig.add_subplot(gs[2, :])
    degrees = results_df["Степінь"].values
    x_pos = np.arange(len(degrees))
    width = 0.35

    ax_metrics.bar(
        x_pos - width / 2, results_df["R²"].values, width, label="R²", alpha=0.8
    )
    ax_metrics.bar(
        x_pos + width / 2,
        results_df["Adj. R²"].values,
        width,
        label="Adjusted R²",
        alpha=0.8,
    )
    ax_metrics.axvline(
        results_df["Adj. R²"].idxmax(),
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Найкраща (степінь {best_degree})",
    )
    ax_metrics.set_xlabel("Степінь полінома")
    ax_metrics.set_ylabel("Значення метрики")
    ax_metrics.set_title("Порівняння якості")
    ax_metrics.set_xticks(x_pos)
    ax_metrics.set_xticklabels(degrees)
    ax_metrics.legend()
    ax_metrics.grid(True, alpha=0.3, axis="y")

    plt.show()


def visualize_forecast(
    X_train, y_train, y_pred, X_forecast, y_forecast, y_lower, y_upper, degree
):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        f"Прогнозування (поліном степеня {degree})", fontsize=14, fontweight="bold"
    )

    axes[0].scatter(X_train, y_train, alpha=0.3, s=10, color="gray", label="Дані")
    axes[0].plot(X_train, y_pred, "b-", linewidth=2, label="Модель")
    axes[0].plot(X_forecast, y_forecast, "r-", linewidth=2, label="Прогноз")
    axes[0].fill_between(
        X_forecast,
        y_lower,
        y_upper,
        alpha=0.2,
        color="red",
        label="95% довірчий інтервал",
    )
    axes[0].axvline(X_train[-1], color="green", linestyle="--", linewidth=2)
    axes[0].set_title("Повний вигляд")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    context_size = int(len(X_train) * 0.2)
    X_context = X_train[-context_size:]
    y_context = y_train[-context_size:]
    y_context_pred = y_pred[-context_size:]

    axes[1].scatter(X_context, y_context, alpha=0.5, s=20, color="gray")
    axes[1].plot(X_context, y_context_pred, "b-", linewidth=2, label="Модель")
    axes[1].plot(X_forecast, y_forecast, "r-", linewidth=3, label="Прогноз")
    axes[1].fill_between(X_forecast, y_lower, y_upper, alpha=0.3, color="red")
    axes[1].axvline(X_train[-1], color="green", linestyle="--", linewidth=2)
    axes[1].set_title("Деталізований вигляд")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def visualize_residuals(X, y_true, y_pred):
    residuals = y_true - y_pred

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Аналіз залишків", fontsize=14, fontweight="bold")

    axes[0, 0].scatter(y_pred, residuals, alpha=0.5, s=10)
    axes[0, 0].axhline(0, color="red", linestyle="--", linewidth=2)
    axes[0, 0].set_xlabel("Передбачені значення")
    axes[0, 0].set_ylabel("Залишки")
    axes[0, 0].set_title("Залишки vs передбачення")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].hist(residuals, bins=50, alpha=0.7, color="skyblue", edgecolor="black")
    axes[0, 1].axvline(0, color="red", linestyle="--", linewidth=2)
    axes[0, 1].set_xlabel("Залишки")
    axes[0, 1].set_ylabel("Частота")
    axes[0, 1].set_title("Розподіл залишків")
    axes[0, 1].grid(True, alpha=0.3, axis="y")

    stats.probplot(residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title("Q-Q Plot")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(X, residuals, alpha=0.5, linewidth=0.5)
    axes[1, 1].scatter(X, residuals, alpha=0.5, s=10)
    axes[1, 1].axhline(0, color="red", linestyle="--", linewidth=2)
    axes[1, 1].set_xlabel("Час")
    axes[1, 1].set_ylabel("Залишки")
    axes[1, 1].set_title("Залишки у часі")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ============================================================================
# Головна функція
# ============================================================================


def main():
    print("\n" + "=" * 80)
    print("ДОМАШНЄ ЗАВДАННЯ 2: Статистичне навчання та прогнозування")
    print("Виконав: Євген Захарченко")
    print("=" * 80 + "\n")

    # Параметри
    N_SAMPLES = 1000
    LOWER_BOUND = -5
    UPPER_BOUND = 5
    CONSTANT_VALUE = 100

    # 1. Генерація вхідних даних
    trend, noise, signal = generate_input_data(
        N_SAMPLES, LOWER_BOUND, UPPER_BOUND, CONSTANT_VALUE, seed=42
    )

    # 2. Додавання аномалій
    signal_with_anomalies, true_anomaly_idx = add_anomalies(
        signal, anomaly_percentage=7, anomaly_magnitude=3, seed=42
    )

    # 3. Очищення від аномалій
    cleaned_signal, detected_anomalies = clean_anomalies(signal_with_anomalies)

    visualize_anomaly_detection(
        signal_with_anomalies, cleaned_signal, detected_anomalies
    )

    # Оцінка якості виявлення
    true_positives = np.sum(detected_anomalies[true_anomaly_idx])
    precision = (
        true_positives / np.sum(detected_anomalies)
        if np.sum(detected_anomalies) > 0
        else 0
    )
    recall = true_positives / len(true_anomaly_idx)

    print("Оцінка виявлення аномалій:")
    print(f"  Precision: {precision:.2%}, Recall: {recall:.2%}\n")

    # 4. Порівняння моделей
    X = np.arange(len(cleaned_signal))
    y = cleaned_signal

    results_df, best_degree, best_model, best_poly, models_data = (
        compare_polynomial_models(X, y, max_degree=5)
    )

    visualize_models(X, y, results_df, models_data, best_degree)

    # 5. Навчання оптимальної моделі
    model, poly_features, y_pred = polynomial_regression(X, y, best_degree)
    metrics = evaluate_model(y, y_pred, best_degree + 1, len(y))

    print("Метрики оптимальної моделі:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.6f}")
    print()

    visualize_residuals(X, y, y_pred)

    # 6. Прогнозування
    X_forecast, y_forecast, y_lower, y_upper = forecast(
        model, poly_features, X, y, forecast_ratio=0.5
    )

    visualize_forecast(
        X, y, y_pred, X_forecast, y_forecast, y_lower, y_upper, best_degree
    )

    # 7. Підсумок
    print("=" * 80)
    print("ПІДСУМОК")
    print("=" * 80)
    print(f"Вхідні дані: {N_SAMPLES} зразків, тренд = {CONSTANT_VALUE}")
    print(f"Аномалії: додано {len(true_anomaly_idx)}, виявлено {np.sum(detected_anomalies)}")
    print(f"Оптимальна модель: поліном степеня {best_degree}")
    print(f"Якість моделі: R² = {metrics['R²']:.4f}, RMSE = {metrics['RMSE']:.4f}")
    print(f"Прогноз: {len(X_forecast)} точок, діапазон [{y_forecast.min():.2f}, {y_forecast.max():.2f}]")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
