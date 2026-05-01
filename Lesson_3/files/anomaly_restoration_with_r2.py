"""
Скрипт для відновлення аномальних вимірів з аналізом R²
Автор: Eugene Pysarchuk
Завдання: Відновлення АВ за МНК з оцінкою достовірності апроксимації
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# Налаштування для українських шрифтів (якщо потрібно)
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 10


class AnomalyRestoration:
    """
    Клас для відновлення аномальних вимірів (АВ) з повною оцінкою достовірності.
    
    Основні можливості:
    - Відновлення АВ за методом найменших квадратів (МНК)
    - Розрахунок метрик якості (R², Adjusted R², MSE, RMSE, MAE, MAPE)
    - Візуалізація результатів
    - Аналіз залишків
    """
    
    def __init__(self, time, data, anomaly_indices):
        """
        Ініціалізація класу.
        
        Параметри:
        -----------
        time : array-like
            Часова вісь (індекси вимірів)
        data : array-like
            Значення вимірів (включаючи АВ)
        anomaly_indices : list or array
            Індекси аномальних вимірів
        """
        self.time = np.array(time)
        self.data = np.array(data)
        self.anomaly_indices = np.array(anomaly_indices)
        self.model = None
        self.restored_data = None
        self.metrics = {}
        
    def restore(self, model_type='linear'):
        """
        Відновлення АВ за вибраною моделлю.
        
        Алгоритм:
        1. Виділення нормальних точок (без АВ)
        2. Побудова регресійної моделі на нормальних точках
        3. Прогнозування значень для АВ
        4. Обчислення метрик якості
        
        Параметри:
        -----------
        model_type : str
            Тип моделі ('linear' - лінійна регресія)
            
        Повертає:
        ----------
        restored_data : array
            Відновлені дані (з замінами АВ)
        """
        print("\n" + "="*70)
        print("ЕТАП 1: ВІДНОВЛЕННЯ АНОМАЛЬНИХ ВИМІРІВ")
        print("="*70)
        
        # Маска нормальних точок
        normal_mask = np.ones(len(self.data), dtype=bool)
        normal_mask[self.anomaly_indices] = False
        
        print(f"Загальна кількість точок: {len(self.data)}")
        print(f"Кількість АВ: {len(self.anomaly_indices)}")
        print(f"Кількість нормальних точок: {np.sum(normal_mask)}")
        print(f"Індекси АВ: {self.anomaly_indices}")
        
        # Нормальні дані для побудови моделі
        X_normal = self.time[normal_mask].reshape(-1, 1)
        y_normal = self.data[normal_mask]
        
        # Будуємо модель
        if model_type == 'linear':
            self.model = LinearRegression()
        else:
            raise ValueError(f"Невідомий тип моделі: {model_type}")
        
        self.model.fit(X_normal, y_normal)
        
        print(f"\nПараметри моделі:")
        print(f"  Нахил (коефіцієнт): {self.model.coef_[0]:.4f}")
        print(f"  Зсув (intercept): {self.model.intercept_:.4f}")
        print(f"  Рівняння: Y = {self.model.intercept_:.4f} + {self.model.coef_[0]:.4f} * X")
        
        # Відновлюємо всі дані
        X_all = self.time.reshape(-1, 1)
        predicted_all = self.model.predict(X_all)
        
        self.restored_data = self.data.copy()
        self.restored_data[self.anomaly_indices] = predicted_all[self.anomaly_indices]
        
        print(f"\nВідновлені значення АВ:")
        for idx in self.anomaly_indices:
            print(f"  t={self.time[idx]:3d}: {self.data[idx]:8.2f} → {self.restored_data[idx]:8.2f}")
        
        # Обчислюємо метрики
        self._calculate_metrics(X_normal, y_normal)
        
        return self.restored_data
    
    def _calculate_metrics(self, X_normal, y_normal):
        """
        Розрахунок метрик якості апроксимації.
        
        Метрики:
        --------
        R² (R-squared) - коефіцієнт детермінації
        Adjusted R² - скоригований R² (враховує кількість параметрів)
        MSE - Mean Squared Error (середньоквадратична похибка)
        RMSE - Root Mean Squared Error (корінь з MSE)
        MAE - Mean Absolute Error (середня абсолютна похибка)
        MAPE - Mean Absolute Percentage Error (відсоткова похибка)
        """
        y_pred_normal = self.model.predict(X_normal)
        
        # Базові метрики
        r2 = r2_score(y_normal, y_pred_normal)
        mse = mean_squared_error(y_normal, y_pred_normal)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_normal, y_pred_normal)
        
        # MAPE (з захистом від ділення на нуль)
        mask = y_normal != 0
        if np.any(mask):
            mape = np.mean(np.abs((y_normal[mask] - y_pred_normal[mask]) / y_normal[mask])) * 100
        else:
            mape = np.nan
        
        # Adjusted R²
        n = len(y_normal)
        p = X_normal.shape[1]  # кількість предикторів
        if n > p + 1:
            adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        else:
            adj_r2 = np.nan
        
        self.metrics = {
            'R²': r2,
            'Adjusted R²': adj_r2,
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape
        }
    
    def print_metrics(self):
        """Виведення метрик у зручному форматі."""
        print("\n" + "="*70)
        print("ЕТАП 2: МЕТРИКИ ДОСТОВІРНОСТІ АПРОКСИМАЦІЇ")
        print("="*70)
        
        print("\n📊 Коефіцієнт детермінації (R²):")
        print("-" * 70)
        r2 = self.metrics['R²']
        interpretation = self._interpret_r2(r2)
        print(f"  R² = {r2:.6f} ({r2*100:.4f}%)")
        print(f"  Інтерпретація: {interpretation}")
        print(f"  Пояснення: Модель пояснює {r2*100:.2f}% варіації даних")
        
        if not np.isnan(self.metrics['Adjusted R²']):
            adj_r2 = self.metrics['Adjusted R²']
            print(f"\n  Adjusted R² = {adj_r2:.6f} ({adj_r2*100:.4f}%)")
            print(f"  (скоригований на кількість параметрів)")
        
        print("\n📏 Метрики похибок:")
        print("-" * 70)
        print(f"  MSE  (Mean Squared Error)     = {self.metrics['MSE']:.6f}")
        print(f"  RMSE (Root MSE)               = {self.metrics['RMSE']:.6f}")
        print(f"  MAE  (Mean Absolute Error)    = {self.metrics['MAE']:.6f}")
        
        if not np.isnan(self.metrics['MAPE']):
            print(f"  MAPE (Mean Abs. % Error)      = {self.metrics['MAPE']:.4f}%")
        
        print("\n💡 Висновок:")
        print("-" * 70)
        if r2 >= 0.90:
            print("  ✓ ВІДМІННА модель! Відновлення надійне.")
        elif r2 >= 0.75:
            print("  ✓ ДОБРА модель. Відновлення прийнятне.")
        elif r2 >= 0.50:
            print("  ⚠ ЗАДОВІЛЬНА модель. Потрібна обережність.")
        else:
            print("  ✗ ПОГАНА модель. Розгляньте інший підхід.")
        
        print("="*70)
    
    def _interpret_r2(self, r2):
        """Інтерпретація значення R²."""
        if r2 >= 0.95:
            return "ВІДМІННО (≥95%)"
        elif r2 >= 0.85:
            return "ДУЖЕ ДОБРЕ (85-95%)"
        elif r2 >= 0.70:
            return "ДОБРЕ (70-85%)"
        elif r2 >= 0.50:
            return "ЗАДОВІЛЬНО (50-70%)"
        else:
            return "ПОГАНО (<50%)"
    
    def visualize(self, save_path='restoration_analysis.png'):
        """
        Комплексна візуалізація результатів відновлення.
        
        Створює 4 графіки:
        1. Відновлення АВ (вихідні vs відновлені дані)
        2. Графік залишків (residual plot)
        3. Q-Q plot (перевірка нормальності залишків)
        4. Фактичні vs Передбачені значення
        
        Параметри:
        -----------
        save_path : str
            Шлях для збереження графіка
        """
        print(f"\n{'='*70}")
        print("ЕТАП 3: ВІЗУАЛІЗАЦІЯ РЕЗУЛЬТАТІВ")
        print("="*70)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Маска нормальних точок
        normal_mask = np.ones(len(self.data), dtype=bool)
        normal_mask[self.anomaly_indices] = False
        
        # ==================== ГРАФІК 1: Відновлення АВ ====================
        ax1 = axes[0, 0]
        
        # Вихідні дані
        ax1.plot(self.time, self.data, 'o-', color='#3498db', 
                label='Вихідні дані (з АВ)', linewidth=2, markersize=6, alpha=0.7)
        
        # Відновлені дані
        ax1.plot(self.time, self.restored_data, '^-', color='#2ecc71', 
                label='Відновлені дані', linewidth=2, markersize=8, alpha=0.8)
        
        # Виділяємо АВ
        ax1.scatter(self.time[self.anomaly_indices], 
                   self.data[self.anomaly_indices],
                   s=300, c='red', marker='x', linewidths=4, 
                   label='Аномальні виміри', zorder=5)
        
        # Модель (лінія регресії)
        ax1.plot(self.time, self.model.predict(self.time.reshape(-1, 1)),
                '--', color='#e74c3c', linewidth=2, alpha=0.7, label='Модель МНК')
        
        ax1.set_xlabel('Час (індекс)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Значення', fontsize=12, fontweight='bold')
        ax1.set_title(f'Відновлення АВ (R² = {self.metrics["R²"]:.4f})', 
                     fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10, loc='best')
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        # ==================== ГРАФІК 2: Залишки ====================
        ax2 = axes[0, 1]
        
        residuals = self.data[normal_mask] - self.model.predict(
            self.time[normal_mask].reshape(-1, 1)
        )
        
        ax2.scatter(self.time[normal_mask], residuals, 
                   alpha=0.6, s=50, color='#3498db')
        ax2.axhline(y=0, color='red', linestyle='--', linewidth=2, label='y=0')
        ax2.axhline(y=np.std(residuals), color='orange', linestyle=':', 
                   linewidth=1.5, label='+1σ')
        ax2.axhline(y=-np.std(residuals), color='orange', linestyle=':', 
                   linewidth=1.5, label='-1σ')
        
        ax2.set_xlabel('Час (індекс)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Залишки (Residuals)', fontsize=12, fontweight='bold')
        ax2.set_title('Графік залишків', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3, linestyle='--')
        
        # Додаємо статистику залишків
        textstr = f'μ = {np.mean(residuals):.4f}\nσ = {np.std(residuals):.4f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        # ==================== ГРАФІК 3: Q-Q Plot ====================
        ax3 = axes[1, 0]
        
        from scipy import stats as sp_stats
        sp_stats.probplot(residuals, dist="norm", plot=ax3)
        ax3.get_lines()[0].set_markerfacecolor('#3498db')
        ax3.get_lines()[0].set_markeredgecolor('#3498db')
        ax3.get_lines()[0].set_markersize(6)
        ax3.get_lines()[1].set_color('#e74c3c')
        ax3.get_lines()[1].set_linewidth(2)
        
        ax3.set_title('Q-Q Plot (перевірка нормальності залишків)', 
                     fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3, linestyle='--')
        
        # ==================== ГРАФІК 4: Фактичні vs Передбачені ====================
        ax4 = axes[1, 1]
        
        y_pred = self.model.predict(self.time[normal_mask].reshape(-1, 1))
        
        ax4.scatter(self.data[normal_mask], y_pred, 
                   alpha=0.6, s=60, color='#3498db', edgecolors='black', linewidth=0.5)
        
        # Лінія ідеального прогнозу (y = x)
        min_val = min(self.data[normal_mask].min(), y_pred.min())
        max_val = max(self.data[normal_mask].max(), y_pred.max())
        ax4.plot([min_val, max_val], [min_val, max_val], 
                'r--', linewidth=2, label='Ідеальний прогноз (y=x)')
        
        ax4.set_xlabel('Фактичні значення', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Передбачені значення', fontsize=12, fontweight='bold')
        ax4.set_title('Фактичні vs Передбачені значення', 
                     fontsize=14, fontweight='bold')
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3, linestyle='--')
        
        # Додаємо R²
        textstr = f'R² = {self.metrics["R²"]:.4f}'
        props = dict(boxstyle='round', facecolor='lightgreen', alpha=0.5)
        ax4.text(0.05, 0.95, textstr, transform=ax4.transAxes, fontsize=12,
                verticalalignment='top', bbox=props, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Графік збережено: {save_path}")
        plt.close()


def compare_r2_scenarios():
    """
    Демонстрація різних сценаріїв R² (високий, середній, низький).
    """
    print("\n" + "🔬"*35)
    print("ДЕМОНСТРАЦІЯ: Порівняння різних рівнів R²")
    print("🔬"*35)
    
    np.random.seed(42)
    X = np.linspace(0, 10, 50).reshape(-1, 1)
    
    # Випадок 1: R² ≈ 0.98 (майже ідеальна модель)
    y_high = 2 * X.flatten() + 3 + np.random.normal(0, 1, 50)
    
    # Випадок 2: R² ≈ 0.70 (добра модель)
    y_medium = 2 * X.flatten() + 3 + np.random.normal(0, 5, 50)
    
    # Випадок 3: R² ≈ 0.30 (слабка модель)
    y_low = 2 * X.flatten() + 3 + np.random.normal(0, 10, 50)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    datasets = [
        (y_high, "Високий R²", '#2ecc71'),
        (y_medium, "Середній R²", '#f39c12'),
        (y_low, "Низький R²", '#e74c3c')
    ]
    
    for idx, (y, title, color) in enumerate(datasets):
        # Тренуємо модель
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        
        # Обчислюємо R²
        r2 = r2_score(y, y_pred)
        
        # Візуалізація
        axes[idx].scatter(X, y, alpha=0.6, s=40, color=color, 
                         edgecolors='black', linewidth=0.5, label='Дані')
        axes[idx].plot(X, y_pred, 'r-', linewidth=3, label='Модель')
        axes[idx].set_title(f"{title}\nR² = {r2:.4f}", 
                           fontsize=14, fontweight='bold')
        axes[idx].set_xlabel('X', fontsize=12)
        axes[idx].set_ylabel('Y', fontsize=12)
        axes[idx].legend(fontsize=10)
        axes[idx].grid(True, alpha=0.3, linestyle='--')
        
        # Додаємо інтерпретацію
        if r2 >= 0.90:
            interpretation = "ВІДМІННО"
        elif r2 >= 0.70:
            interpretation = "ДОБРЕ"
        elif r2 >= 0.50:
            interpretation = "ЗАДОВІЛЬНО"
        else:
            interpretation = "ПОГАНО"
        
        textstr = f'{interpretation}\n{r2*100:.1f}% варіації\nпояснено'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.7)
        axes[idx].text(0.05, 0.95, textstr, transform=axes[idx].transAxes,
                      fontsize=10, verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig('r2_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Графік порівняння R² збережено: r2_comparison.png\n")
    plt.close()


def main():
    """
    Головна функція - демонстрація роботи системи відновлення АВ.
    """
    print("\n" + "🎯"*35)
    print("ПРОГРАМА: ВІДНОВЛЕННЯ АНОМАЛЬНИХ ВИМІРІВ З АНАЛІЗОМ R²")
    print("🎯"*35)
    
    # ==================== ЧАСТИНА 1: Порівняння R² ====================
    compare_r2_scenarios()
    
    # ==================== ЧАСТИНА 2: Відновлення АВ ====================
    print("\n" + "="*70)
    print("ЧАСТИНА 2: ПРИКЛАД ВІДНОВЛЕННЯ АВ (Симуляція курсу валюти)")
    print("="*70)
    
    # Генеруємо реалістичні дані (наприклад, курс валюти)
    np.random.seed(42)
    time = np.arange(1, 51)
    
    # Базовий тренд: зростання курсу
    true_signal = 27.0 + 0.05 * time + 0.001 * time**2
    
    # Додаємо невеликий шум
    noise = np.random.normal(0, 0.3, len(time))
    data = true_signal + noise
    
    # Симулюємо АВ (різкі стрибки курсу)
    anomaly_indices = [15, 16, 17, 30, 31, 32]
    data[anomaly_indices] = [35.0, 36.5, 34.0, 24.0, 25.0, 23.5]
    
    print(f"\nГенеровано {len(time)} точок даних")
    print(f"Додано {len(anomaly_indices)} аномальних вимірів")
    print(f"Діапазон значень: [{data.min():.2f}, {data.max():.2f}]")
    
    # Створюємо об'єкт для відновлення
    restorer = AnomalyRestoration(time, data, anomaly_indices)
    
    # Виконуємо відновлення
    restored = restorer.restore(model_type='linear')
    
    # Виводимо метрики
    restorer.print_metrics()
    
    # Візуалізація
    restorer.visualize(save_path='restoration_analysis.png')
    
    print("\n" + "✅"*35)
    print("ПРОГРАМА ЗАВЕРШЕНА УСПІШНО!")
    print("✅"*35)
    print("\nЗгенеровані файли:")
    print("  1. r2_comparison.png - порівняння різних рівнів R²")
    print("  2. restoration_analysis.png - аналіз відновлення АВ")
    print("\nВи можете відкрити ці файли у поточній директорії.")


if __name__ == "__main__":
    main()
