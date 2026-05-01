import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


class PolynomialOrderEstimator:
    """
    Клас для визначення оптимального порядку поліноміальної моделі
    за алгоритмом з лекції.
    """

    def __init__(self, y, dt=1.0, max_order=5):
        """
        Параметри:
        -----------
        y : array-like
            Часовий ряд даних
        dt : float
            Інтервал між вимірами
        max_order : int
            Максимальний порядок для перевірки
        """
        self.y = np.array(y)
        self.dt = dt
        self.max_order = max_order
        self.n = len(y)

        # Результати
        self.derivatives = {}
        self.sigma_T = {}
        self.sigma_E = {}
        self.delta = {}
        self.optimal_order = None

    def compute_derivative(self, y_prev, p):
        """
        Обчислення похідної p-го порядку.

        Формула: y_j^(p) = (y_{j+1}^(p-1) - y_j^(p-1)) / Δt
        """
        if p == 0:
            return y_prev

        derivative = np.diff(y_prev) / self.dt
        return derivative

    def compute_theoretical_variance(self, p):
        """
        Обчислення теоретичної дисперсії похідної p-го порядку.

        Формула: σ²_{y^(p)_T} = (2σ_y²) / (Δt^{2p})
        """
        sigma_y_squared = np.var(self.y, ddof=1)
        sigma_T_squared = (2 * sigma_y_squared) / (self.dt ** (2 * p))
        return sigma_T_squared

    def compute_experimental_variance(self, derivative):
        """
        Обчислення експериментальної дисперсії.

        Формула: σ²_{y^(p)_E} = (1/(m-p-1)) Σ(y_j^(p) - m_{y^(p)})²
        """
        if len(derivative) <= 1:
            return 0.0

        mean_derivative = np.mean(derivative)
        sigma_E_squared = np.sum((derivative - mean_derivative) ** 2) / (len(derivative) - 1)
        return sigma_E_squared

    def find_optimal_order(self):
        """
        Головна функція: знаходження оптимального порядку полінома.
        """
        print("=" * 70)
        print("ВИЗНАЧЕННЯ ОПТИМАЛЬНОГО ПОРЯДКУ ПОЛІНОМІАЛЬНОЇ МОДЕЛІ")
        print("=" * 70)
        print(f"\nВхідні дані:")
        print(f"  Кількість точок: {self.n}")
        print(f"  Інтервал Δt: {self.dt}")
        print(f"  Діапазон даних: [{self.y.min():.4f}, {self.y.max():.4f}]")

        # Обчислюємо похідні та метрики для кожного порядку
        y_current = self.y.copy()

        for p in range(1, self.max_order + 1):
            # Крок 1: Обчислення похідної
            y_current = self.compute_derivative(y_current, 1)
            self.derivatives[p] = y_current.copy()

            # Перевірка: чи достатньо точок?
            if len(y_current) < 2:
                print(f"\n⚠️ Недостатньо точок для порядку p={p}")
                break

            # Крок 2: Теоретична дисперсія
            self.sigma_T[p] = self.compute_theoretical_variance(p)

            # Крок 3: Експериментальна дисперсія
            self.sigma_E[p] = self.compute_experimental_variance(y_current)

            # Крок 4: Різниця дисперсій
            self.delta[p] = abs(self.sigma_E[p] - self.sigma_T[p])

        # Знаходимо мінімальну Δ
        if self.delta:
            self.optimal_order = min(self.delta.keys(), key=lambda p: self.delta[p])

        # Виводимо результати
        self._print_results()

        return self.optimal_order

    def _print_results(self):
        """Виведення результатів у таблиці."""
        print("\n" + "=" * 70)
        print("РЕЗУЛЬТАТИ ОБЧИСЛЕНЬ")
        print("=" * 70)
        print(f"\n{'p':<5} {'σ²_T':<15} {'σ²_E':<15} {'Δ^(p)':<15} {'Статус':<15}")
        print("-" * 70)

        for p in sorted(self.delta.keys()):
            status = "✓ ОПТИМУМ" if p == self.optimal_order else ""
            print(f"{p:<5} {self.sigma_T[p]:<15.6f} {self.sigma_E[p]:<15.6f} "
                  f"{self.delta[p]:<15.6f} {status:<15}")

        print("\n" + "=" * 70)
        print(f"🎯 ОПТИМАЛЬНИЙ ПОРЯДОК: p = {self.optimal_order}")
        print("=" * 70)

        # Інтерпретація
        if self.optimal_order == 1:
            print("\n💡 Інтерпретація: Дані мають ЛІНІЙНИЙ тренд")
            print("   Рекомендована модель: y = a₀ + a₁·x")
        elif self.optimal_order == 2:
            print("\n💡 Інтерпретація: Дані мають КВАДРАТИЧНИЙ тренд")
            print("   Рекомендована модель: y = a₀ + a₁·x + a₂·x²")
        elif self.optimal_order == 3:
            print("\n💡 Інтерпретація: Дані мають КУБІЧНИЙ тренд")
            print("   Рекомендована модель: y = a₀ + a₁·x + a₂·x² + a₃·x³")
        else:
            print(f"\n💡 Інтерпретація: Дані мають складний тренд порядку {self.optimal_order}")

    def visualize(self, save_path='polynomial_order_analysis.png'):
        """Візуалізація результатів."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))

        # Графік 1: Вихідні дані
        ax1 = axes[0, 0]
        ax1.plot(self.y, 'o-', linewidth=2, markersize=6, color='#3498db')
        ax1.set_title('Вихідний часовий ряд', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Індекс t', fontsize=12)
        ax1.set_ylabel('Значення y', fontsize=12)
        ax1.grid(True, alpha=0.3)

        # Графік 2: Похідні різних порядків
        ax2 = axes[0, 1]
        colors = ['#e74c3c', '#f39c12', '#2ecc71', '#9b59b6', '#1abc9c']
        for p, derivative in self.derivatives.items():
            if p <= len(colors):
                t = np.arange(len(derivative))
                ax2.plot(t, derivative, 'o-', label=f'p={p}',
                         color=colors[p - 1], linewidth=2, markersize=4)
        ax2.set_title('Похідні різних порядків', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Індекс', fontsize=12)
        ax2.set_ylabel('Значення похідної', fontsize=12)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

        # Графік 3: Порівняння дисперсій
        ax3 = axes[1, 0]
        orders = sorted(self.sigma_T.keys())
        sigma_T_vals = [self.sigma_T[p] for p in orders]
        sigma_E_vals = [self.sigma_E[p] for p in orders]

        x = np.arange(len(orders))
        width = 0.35
        ax3.bar(x - width / 2, sigma_T_vals, width, label='σ²_T (теоретична)',
                color='#3498db', alpha=0.7)
        ax3.bar(x + width / 2, sigma_E_vals, width, label='σ²_E (експериментальна)',
                color='#e74c3c', alpha=0.7)
        ax3.set_xlabel('Порядок p', fontsize=12)
        ax3.set_ylabel('Дисперсія', fontsize=12)
        ax3.set_title('Порівняння дисперсій', fontsize=14, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(orders)
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3, axis='y')

        # Графік 4: Δ^(p) для різних порядків
        ax4 = axes[1, 1]
        orders = sorted(self.delta.keys())
        delta_vals = [self.delta[p] for p in orders]

        ax4.plot(orders, delta_vals, 'o-', linewidth=3, markersize=10,
                 color='#2ecc71')
        ax4.axvline(x=self.optimal_order, color='red', linestyle='--',
                    linewidth=2, label=f'Оптимум: p={self.optimal_order}')
        ax4.scatter([self.optimal_order], [self.delta[self.optimal_order]],
                    s=300, color='red', marker='*', zorder=5,
                    label='Мінімум Δ')
        ax4.set_xlabel('Порядок p', fontsize=12)
        ax4.set_ylabel('Δ^(p) = |σ²_E - σ²_T|', fontsize=12)
        ax4.set_title('Критерій вибору порядку', fontsize=14, fontweight='bold')
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Графік збережено: {save_path}")
        plt.close()


# ============================================================================
# ДЕМОНСТРАЦІЯ НА ПРИКЛАДАХ
# ============================================================================

def demo_examples():
    """Демонстрація алгоритму на різних типах даних."""

    print("\n" + "🔬" * 35)
    print("ДЕМОНСТРАЦІЯ АЛГОРИТМУ ВИЗНАЧЕННЯ ПОРЯДКУ МОДЕЛІ")
    print("🔬" * 35)

    # Приклад 1: Лінійний тренд + шум
    print("\n" + "=" * 70)
    print("ПРИКЛАД 1: ЛІНІЙНИЙ ТРЕНД")
    print("=" * 70)

    np.random.seed(42)
    t = np.linspace(0, 10, 50)
    y_linear = 5 + 2 * t + np.random.normal(0, 0.5, 50)

    estimator1 = PolynomialOrderEstimator(y_linear, dt=t[1] - t[0], max_order=5)
    optimal1 = estimator1.find_optimal_order()
    estimator1.visualize('example1_linear.png')

    # Приклад 2: Квадратичний тренд + шум
    print("\n" + "=" * 70)
    print("ПРИКЛАД 2: КВАДРАТИЧНИЙ ТРЕНД")
    print("=" * 70)

    y_quadratic = 10 + 1 * t + 0.5 * t ** 2 + np.random.normal(0, 1, 50)

    estimator2 = PolynomialOrderEstimator(y_quadratic, dt=t[1] - t[0], max_order=5)
    optimal2 = estimator2.find_optimal_order()
    estimator2.visualize('example2_quadratic.png')

    # Приклад 3: Кубічний тренд + шум
    print("\n" + "=" * 70)
    print("ПРИКЛАД 3: КУБІЧНИЙ ТРЕНД")
    print("=" * 70)

    y_cubic = 5 + 0.5 * t + 0.2 * t ** 2 - 0.01 * t ** 3 + np.random.normal(0, 0.8, 50)

    estimator3 = PolynomialOrderEstimator(y_cubic, dt=t[1] - t[0], max_order=5)
    optimal3 = estimator3.find_optimal_order()
    estimator3.visualize('example3_cubic.png')

    # Підсумок
    print("\n" + "=" * 70)
    print("ПІДСУМОК РЕЗУЛЬТАТІВ")
    print("=" * 70)
    print(f"Приклад 1 (лінійний):      оптимальний порядок = {optimal1}")
    print(f"Приклад 2 (квадратичний):  оптимальний порядок = {optimal2}")
    print(f"Приклад 3 (кубічний):      оптимальний порядок = {optimal3}")
    print("=" * 70)


if __name__ == "__main__":
    demo_examples()
