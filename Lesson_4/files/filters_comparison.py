"""
Порівняння Alpha-Beta та Kalman фільтрів
=========================================

Цей скрипт демонструє:
1. Реалізацію Alpha-Beta фільтра (фіксовані коефіцієнти)
2. Реалізацію фільтра Калмана (адаптивні коефіцієнти)
3. Порівняння їх роботи на реальних даних курсу USD/UAH

Автор: Навчальний приклад для Data Science
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Optional
from dataclasses import dataclass


# =============================================================================
# ЧАСТИНА 1: ALPHA-BETA ФІЛЬТР
# =============================================================================

@dataclass
class AlphaBetaResult:
    """Результати роботи Alpha-Beta фільтра"""
    positions: np.ndarray      # Оцінки позиції (згладжені значення)
    velocities: np.ndarray     # Оцінки швидкості (тренд)
    residuals: np.ndarray      # Нев'язки (різниця між виміром і прогнозом)


class AlphaBetaFilter:
    """
    Alpha-Beta фільтр для рекурентного згладжування.
    
    Модель стану:
        x(k) = x(k-1) + v(k-1) * dt    # позиція
        v(k) = v(k-1)                   # швидкість (константна)
    
    Рівняння оновлення:
        x̂(k) = x_pred(k) + α * residual
        v̂(k) = v_pred(k) + (β/dt) * residual
    
    де residual = z(k) - x_pred(k)
    
    Параметри:
        α (alpha): коефіцієнт корекції позиції, 0 < α < 1
            - Більше α → швидша реакція на зміни, але більше шуму
            - Менше α → краще згладжування, але більше відставання
        
        β (beta): коефіцієнт корекції швидкості, 0 < β < 4-2α
            - Більше β → швидша адаптація тренду
            - Менше β → стабільніший тренд
    """
    
    def __init__(self, alpha: float, beta: float, dt: float = 1.0):
        """
        Ініціалізація фільтра.
        
        Args:
            alpha: коефіцієнт корекції позиції (0 < α < 1)
            beta: коефіцієнт корекції швидкості (0 < β < 4-2α)
            dt: крок часу між вимірами
        """
        # Валідація параметрів
        if not 0 < alpha < 1:
            raise ValueError(f"Alpha має бути в межах (0, 1), отримано: {alpha}")
        if not 0 < beta < (4 - 2*alpha):
            raise ValueError(f"Beta має бути в межах (0, {4-2*alpha}), отримано: {beta}")
        
        self.alpha = alpha
        self.beta = beta
        self.dt = dt
        
        # Стан фільтра
        self.x_est = 0.0  # оцінка позиції
        self.v_est = 0.0  # оцінка швидкості
        self.initialized = False
    
    def reset(self):
        """Скидання стану фільтра"""
        self.x_est = 0.0
        self.v_est = 0.0
        self.initialized = False
    
    def update(self, measurement: float) -> Tuple[float, float, float]:
        """
        Один крок фільтрації.
        
        Args:
            measurement: нове виміряне значення
        
        Returns:
            (x_est, v_est, residual): оцінка позиції, швидкості та нев'язка
        """
        if not self.initialized:
            # Перший вимір — ініціалізуємо стан
            self.x_est = measurement
            self.v_est = 0.0
            self.initialized = True
            return self.x_est, self.v_est, 0.0
        
        # === КРОК 1: ПРОГНОЗ (Predict) ===
        # Екстраполюємо стан на основі моделі руху
        x_pred = self.x_est + self.v_est * self.dt
        v_pred = self.v_est  # швидкість вважаємо константною
        
        # === КРОК 2: НЕВ'ЯЗКА (Residual) ===
        # Різниця між тим, що виміряли, і тим, що очікували
        residual = measurement - x_pred
        
        # === КРОК 3: КОРЕКЦІЯ (Update) ===
        # Коригуємо прогноз на основі нев'язки
        self.x_est = x_pred + self.alpha * residual
        self.v_est = v_pred + (self.beta / self.dt) * residual
        
        return self.x_est, self.v_est, residual
    
    def filter(self, measurements: np.ndarray) -> AlphaBetaResult:
        """
        Фільтрація всього масиву даних.
        
        Args:
            measurements: масив вимірів
        
        Returns:
            AlphaBetaResult з оцінками позиції, швидкості та нев'язками
        """
        self.reset()
        n = len(measurements)
        
        positions = np.zeros(n)
        velocities = np.zeros(n)
        residuals = np.zeros(n)
        
        for i, z in enumerate(measurements):
            x, v, r = self.update(z)
            positions[i] = x
            velocities[i] = v
            residuals[i] = r
        
        return AlphaBetaResult(positions, velocities, residuals)
    
    @staticmethod
    def benedict_bordner_beta(alpha: float) -> float:
        """
        Оптимальне β за критерієм Бенедикта-Борднера.
        
        Мінімізує комбіновану похибку для цілей з постійною швидкістю.
        
        Формула: β = α² / (2 - α)
        """
        return alpha**2 / (2 - alpha)
    
    def noise_reduction_ratio(self) -> float:
        """
        Коефіцієнт придушення шуму (ρ²).
        
        Показує, яка частка дисперсії шуму проходить через фільтр.
        Менше значення = краще згладжування.
        """
        a, b = self.alpha, self.beta
        numerator = 2*a**2 + a*b + 2*b
        denominator = a * (4 - b - 2*a)
        return numerator / denominator
    
    def steady_state_lag(self, acceleration: float) -> float:
        """
        Стаціонарна динамічна похибка при постійному прискоренні.
        
        Формула: e = a * T² / β
        
        Args:
            acceleration: прискорення сигналу
        
        Returns:
            Величина відставання фільтра
        """
        return acceleration * self.dt**2 / self.beta


# =============================================================================
# ЧАСТИНА 2: ФІЛЬТР КАЛМАНА
# =============================================================================

@dataclass
class KalmanResult:
    """Результати роботи фільтра Калмана"""
    positions: np.ndarray      # Оцінки позиції
    velocities: np.ndarray     # Оцінки швидкості
    residuals: np.ndarray      # Нев'язки
    kalman_gains: np.ndarray   # Історія коефіцієнтів Калмана (α, β)
    uncertainties: np.ndarray  # Історія невизначеності оцінки


class KalmanFilter:
    """
    Фільтр Калмана для рекурентного згладжування.
    
    Ключова відмінність від Alpha-Beta:
    - Коефіцієнти корекції (Kalman gain) обчислюються АВТОМАТИЧНО
    - На кожному кроці враховується поточна невизначеність
    - Адаптується до змін у процесі
    
    Модель стану (та сама, що й Alpha-Beta):
        x(k) = x(k-1) + v(k-1) * dt + w_x    # позиція + шум процесу
        v(k) = v(k-1) + w_v                   # швидкість + шум процесу
    
    Модель виміру:
        z(k) = x(k) + v_k                     # вимір = позиція + шум виміру
    
    Параметри налаштування:
        Q: коваріація шуму процесу (наскільки процес "блукає")
        R: коваріація шуму виміру (наскільки неточні виміри)
        
        Співвідношення Q/R визначає поведінку:
        - Велике Q/R → більше довіри вимірам (як великі α, β)
        - Мале Q/R → більше довіри моделі (як малі α, β)
    """
    
    def __init__(self, 
                 process_noise: float = 0.1, 
                 measurement_noise: float = 1.0,
                 dt: float = 1.0):
        """
        Ініціалізація фільтра Калмана.
        
        Args:
            process_noise: стандартне відхилення шуму процесу (σ_w)
            measurement_noise: стандартне відхилення шуму виміру (σ_v)
            dt: крок часу між вимірами
        """
        self.dt = dt
        
        # === МАТРИЦІ МОДЕЛІ ===
        
        # Матриця переходу стану F (State Transition Matrix)
        # [x_new]   [1  dt] [x_old]
        # [v_new] = [0   1] [v_old]
        self.F = np.array([
            [1, dt],
            [0, 1]
        ])
        
        # Матриця виміру H (Observation Matrix)
        # z = [1  0] * [x, v]^T = x (вимірюємо тільки позицію)
        self.H = np.array([[1, 0]])
        
        # Коваріація шуму процесу Q (Process Noise Covariance)
        # Модель: шум впливає на швидкість, яка інтегрується в позицію
        q = process_noise**2
        self.Q = np.array([
            [q * dt**3 / 3, q * dt**2 / 2],
            [q * dt**2 / 2, q * dt]
        ])
        
        # Коваріація шуму виміру R (Measurement Noise Covariance)
        self.R = np.array([[measurement_noise**2]])
        
        # === СТАН ФІЛЬТРА ===
        
        # Вектор стану [позиція, швидкість]
        self.x = np.zeros(2)
        
        # Коваріаційна матриця невизначеності P (Error Covariance)
        # Початкова невизначеність — велика
        self.P = np.eye(2) * 1000
        
        self.initialized = False
    
    def reset(self):
        """Скидання стану фільтра"""
        self.x = np.zeros(2)
        self.P = np.eye(2) * 1000
        self.initialized = False
    
    def update(self, measurement: float) -> Tuple[float, float, float, np.ndarray, float]:
        """
        Один крок фільтрації Калмана.
        
        Args:
            measurement: нове виміряне значення
        
        Returns:
            (x_est, v_est, residual, kalman_gain, uncertainty)
        """
        if not self.initialized:
            self.x[0] = measurement
            self.x[1] = 0.0
            self.initialized = True
            return self.x[0], self.x[1], 0.0, np.array([0, 0]), np.sqrt(self.P[0, 0])
        
        # === КРОК 1: ПРОГНОЗ (Predict) ===
        
        # Прогноз стану: x_pred = F * x
        x_pred = self.F @ self.x
        
        # Прогноз невизначеності: P_pred = F * P * F^T + Q
        P_pred = self.F @ self.P @ self.F.T + self.Q
        
        # === КРОК 2: НЕВ'ЯЗКА (Innovation) ===
        
        # Прогнозований вимір
        z_pred = self.H @ x_pred
        
        # Нев'язка (innovation)
        residual = measurement - z_pred[0]
        
        # Коваріація нев'язки: S = H * P_pred * H^T + R
        S = self.H @ P_pred @ self.H.T + self.R
        
        # === КРОК 3: КОЕФІЦІЄНТ КАЛМАНА (Kalman Gain) ===
        
        # K = P_pred * H^T * S^(-1)
        # Це і є ті самі α, β, але обчислені оптимально!
        K = P_pred @ self.H.T @ np.linalg.inv(S)
        
        # === КРОК 4: КОРЕКЦІЯ (Update) ===
        
        # Оновлення стану: x = x_pred + K * residual
        self.x = x_pred + K.flatten() * residual
        
        # Оновлення невизначеності: P = (I - K*H) * P_pred
        I = np.eye(2)
        self.P = (I - K @ self.H) @ P_pred
        
        # Витягуємо еквівалентні α, β для порівняння
        alpha_equivalent = K[0, 0]
        beta_equivalent = K[1, 0] * self.dt
        
        uncertainty = np.sqrt(self.P[0, 0])
        
        return (self.x[0], self.x[1], residual, 
                np.array([alpha_equivalent, beta_equivalent]), uncertainty)
    
    def filter(self, measurements: np.ndarray) -> KalmanResult:
        """
        Фільтрація всього масиву даних.
        
        Args:
            measurements: масив вимірів
        
        Returns:
            KalmanResult з оцінками та історією параметрів
        """
        self.reset()
        n = len(measurements)
        
        positions = np.zeros(n)
        velocities = np.zeros(n)
        residuals = np.zeros(n)
        kalman_gains = np.zeros((n, 2))
        uncertainties = np.zeros(n)
        
        for i, z in enumerate(measurements):
            x, v, r, K, u = self.update(z)
            positions[i] = x
            velocities[i] = v
            residuals[i] = r
            kalman_gains[i] = K
            uncertainties[i] = u
        
        return KalmanResult(positions, velocities, residuals, 
                           kalman_gains, uncertainties)


# =============================================================================
# ЧАСТИНА 3: ФУНКЦІЇ ВІЗУАЛІЗАЦІЇ ТА АНАЛІЗУ
# =============================================================================

def load_oschadbank_data(filepath: str) -> pd.DataFrame:
    """Завантаження даних Oschadbank"""
    try:
        df = pd.read_excel(filepath)
        print(f"Завантажено {len(df)} записів")
        print(f"Колонки: {list(df.columns)}")
        return df
    except Exception as e:
        print(f"Помилка завантаження: {e}")
        return None


def compare_filters(measurements: np.ndarray, 
                    title: str = "Порівняння фільтрів",
                    alpha: float = 0.3,
                    process_noise: float = 0.5,
                    measurement_noise: float = 2.0) -> dict:
    """
    Порівняння Alpha-Beta та Kalman фільтрів на одних даних.
    
    Args:
        measurements: масив вимірів
        title: заголовок графіка
        alpha: параметр α для Alpha-Beta фільтра
        process_noise: σ_w для Калмана
        measurement_noise: σ_v для Калмана
    
    Returns:
        Словник з результатами обох фільтрів
    """
    # Налаштування Alpha-Beta
    beta = AlphaBetaFilter.benedict_bordner_beta(alpha)
    ab_filter = AlphaBetaFilter(alpha=alpha, beta=beta)
    
    # Налаштування Калмана
    kalman = KalmanFilter(
        process_noise=process_noise,
        measurement_noise=measurement_noise
    )
    
    # Фільтрація
    ab_result = ab_filter.filter(measurements)
    kalman_result = kalman.filter(measurements)
    
    # === ВІЗУАЛІЗАЦІЯ ===
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    time = np.arange(len(measurements))
    
    # --- Графік 1: Порівняння згладжених сигналів ---
    ax1 = axes[0, 0]
    ax1.plot(time, measurements, 'b.', alpha=0.3, markersize=2, label='Виміри')
    ax1.plot(time, ab_result.positions, 'r-', linewidth=1.5, 
             label=f'Alpha-Beta (α={alpha:.2f}, β={beta:.3f})')
    ax1.plot(time, kalman_result.positions, 'g-', linewidth=1.5, 
             label='Kalman')
    ax1.set_xlabel('Час')
    ax1.set_ylabel('Значення')
    ax1.set_title('Згладжені сигнали')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # --- Графік 2: Оцінки швидкості (тренду) ---
    ax2 = axes[0, 1]
    ax2.plot(time, ab_result.velocities, 'r-', linewidth=1, 
             label='Alpha-Beta', alpha=0.8)
    ax2.plot(time, kalman_result.velocities, 'g-', linewidth=1, 
             label='Kalman', alpha=0.8)
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Час')
    ax2.set_ylabel('Швидкість')
    ax2.set_title('Оцінка швидкості (тренду)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # --- Графік 3: Нев'язки Alpha-Beta ---
    ax3 = axes[1, 0]
    ax3.plot(time, ab_result.residuals, 'r-', linewidth=0.5, alpha=0.7)
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax3.fill_between(time, ab_result.residuals, alpha=0.3, color='red')
    ax3.set_xlabel('Час')
    ax3.set_ylabel('Нев\'язка')
    ax3.set_title(f'Нев\'язки Alpha-Beta (σ={np.std(ab_result.residuals):.2f})')
    ax3.grid(True, alpha=0.3)
    
    # --- Графік 4: Нев'язки Kalman ---
    ax4 = axes[1, 1]
    ax4.plot(time, kalman_result.residuals, 'g-', linewidth=0.5, alpha=0.7)
    ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax4.fill_between(time, kalman_result.residuals, alpha=0.3, color='green')
    ax4.set_xlabel('Час')
    ax4.set_ylabel('Нев\'язка')
    ax4.set_title(f'Нев\'язки Kalman (σ={np.std(kalman_result.residuals):.2f})')
    ax4.grid(True, alpha=0.3)
    
    # --- Графік 5: Еквівалентні α, β Калмана ---
    ax5 = axes[2, 0]
    ax5.plot(time, kalman_result.kalman_gains[:, 0], 'b-', 
             linewidth=1, label='α (позиція)')
    ax5.plot(time, kalman_result.kalman_gains[:, 1], 'orange', 
             linewidth=1, label='β (швидкість)')
    ax5.axhline(y=alpha, color='r', linestyle='--', alpha=0.7, 
                label=f'α Alpha-Beta = {alpha}')
    ax5.axhline(y=beta, color='darkred', linestyle=':', alpha=0.7,
                label=f'β Alpha-Beta = {beta:.3f}')
    ax5.set_xlabel('Час')
    ax5.set_ylabel('Значення коефіцієнта')
    ax5.set_title('Адаптивні коефіцієнти Калмана vs фіксовані Alpha-Beta')
    ax5.legend(loc='upper right')
    ax5.grid(True, alpha=0.3)
    
    # --- Графік 6: Невизначеність оцінки Калмана ---
    ax6 = axes[2, 1]
    ax6.plot(time, kalman_result.uncertainties, 'purple', linewidth=1)
    ax6.set_xlabel('Час')
    ax6.set_ylabel('σ оцінки')
    ax6.set_title('Невизначеність оцінки Калмана (зменшується з часом)')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('filters_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # === СТАТИСТИКА ===
    print("\n" + "="*60)
    print("СТАТИСТИКА ПОРІВНЯННЯ")
    print("="*60)
    
    print(f"\n--- Alpha-Beta (α={alpha:.2f}, β={beta:.3f}) ---")
    print(f"СКВ нев'язок: {np.std(ab_result.residuals):.4f}")
    print(f"Коефіцієнт придушення шуму ρ²: {ab_filter.noise_reduction_ratio():.4f}")
    
    print(f"\n--- Kalman ---")
    print(f"СКВ нев'язок: {np.std(kalman_result.residuals):.4f}")
    print(f"Фінальні еквівалентні коефіцієнти:")
    print(f"  α ≈ {kalman_result.kalman_gains[-1, 0]:.4f}")
    print(f"  β ≈ {kalman_result.kalman_gains[-1, 1]:.4f}")
    print(f"Фінальна невизначеність: {kalman_result.uncertainties[-1]:.4f}")
    
    return {
        'alpha_beta': ab_result,
        'kalman': kalman_result,
        'ab_filter': ab_filter
    }


def demonstrate_tradeoff(n_points: int = 500):
    """
    Демонстрація компромісу між динамічною та стохастичною точністю.
    """
    np.random.seed(42)
    
    # Генеруємо сигнал: постійна швидкість, потім стрибок, потім прискорення
    true_signal = np.zeros(n_points)
    
    for i in range(1, n_points):
        if i < 150:
            # Постійна швидкість
            true_signal[i] = true_signal[i-1] + 0.5
        elif i < 200:
            # Різкий стрибок (імітація шоку)
            true_signal[i] = true_signal[i-1] + 3.0
        elif i < 350:
            # Прискорення
            acceleration = 0.02 * (i - 200)
            true_signal[i] = true_signal[i-1] + 0.5 + acceleration
        else:
            # Знову постійна швидкість
            true_signal[i] = true_signal[i-1] + 1.5
    
    # Додаємо шум
    noise_std = 5.0
    measurements = true_signal + np.random.normal(0, noise_std, n_points)
    
    # Тестуємо різні α
    alphas = [0.05, 0.2, 0.5, 0.9]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Компроміс: різні значення α', fontsize=14, fontweight='bold')
    
    for ax, alpha in zip(axes.flat, alphas):
        beta = AlphaBetaFilter.benedict_bordner_beta(alpha)
        ab = AlphaBetaFilter(alpha=alpha, beta=beta)
        result = ab.filter(measurements)
        
        rmse = np.sqrt(np.mean((result.positions - true_signal)**2))
        noise_ratio = ab.noise_reduction_ratio()
        
        ax.plot(true_signal, 'g-', linewidth=2, label='Істинний сигнал', alpha=0.7)
        ax.plot(measurements, 'b.', alpha=0.15, markersize=1, label='Виміри')
        ax.plot(result.positions, 'r-', linewidth=1.5, label='Фільтр')
        
        # Позначки режимів
        ax.axvline(150, color='gray', linestyle='--', alpha=0.3)
        ax.axvline(200, color='gray', linestyle='--', alpha=0.3)
        ax.axvline(350, color='gray', linestyle='--', alpha=0.3)
        
        ax.set_title(f'α={alpha}, β={beta:.3f}\n'
                    f'RMSE={rmse:.2f}, ρ²={noise_ratio:.3f}')
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('tradeoff_demonstration.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n" + "="*60)
    print("АНАЛІЗ КОМПРОМІСУ")
    print("="*60)
    print("\nМале α (0.05): відмінне згладжування, але ВЕЛИЧЕЗНЕ відставання")
    print("             при різких змінах — динамічна точність низька")
    print("\nВелике α (0.9): миттєва реакція, але майже нульове згладжування")
    print("              — стохастична точність низька")
    print("\nСереднє α (0.2-0.5): компроміс — обидві точності 'відносно невисокі'")


# =============================================================================
# ГОЛОВНИЙ БЛОК
# =============================================================================

if __name__ == "__main__":
    
    print("="*60)
    print("ДЕМОНСТРАЦІЯ ALPHA-BETA ТА KALMAN ФІЛЬТРІВ")
    print("="*60)
    
    # Спроба завантажити реальні дані
    try:
        df = load_oschadbank_data('../Oschadbank (USD).xls')
        
        if df is not None:
            # Вибираємо колонку з курсом (зазвичай 'Продаж' або 'КурсНбу')
            # Спробуємо знайти потрібну колонку
            possible_columns = ['Продаж', 'Купівля', 'КурсНбу', 'Sale', 'Buy']
            data_column = None
            
            for col in possible_columns:
                if col in df.columns:
                    data_column = col
                    break
            
            if data_column is None:
                # Беремо першу числову колонку
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    data_column = numeric_cols[0]
            
            if data_column:
                print(f"\nВикористовуємо колонку: '{data_column}'")
                measurements = df[data_column].dropna().values
                
                # Порівняння фільтрів
                results = compare_filters(
                    measurements,
                    title=f'Курс USD/UAH (Oschadbank): {data_column}',
                    alpha=0.3,
                    process_noise=0.5,
                    measurement_noise=2.0
                )
    except Exception as e:
        print(f"Не вдалося завантажити реальні дані: {e}")
        print("Використовуємо синтетичні дані...")
    
    # Демонстрація компромісу
    print("\n" + "="*60)
    print("ДЕМОНСТРАЦІЯ КОМПРОМІСУ ДИНАМІЧНОЇ/СТОХАСТИЧНОЇ ТОЧНОСТІ")
    print("="*60)
    
    demonstrate_tradeoff()
    
    print("\n✅ Готово! Графіки збережено.")
