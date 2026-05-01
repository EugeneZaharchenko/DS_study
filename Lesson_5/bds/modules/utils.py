import numpy as np
import matplotlib.pyplot as plt


def multi_plot(
    *args: tuple[np.ndarray, str], ylims: tuple[float, float] = None
) -> None:
    for i in range(len(args)):
        plt.plot(args[i][0], label=args[i][1])
    plt.legend(loc="lower right")
    if ylims:
        plt.ylim(ylims)
    plt.show()


def statistics(
    data: np.ndarray,
    fitted: np.ndarray,
) -> None:
    med = np.median(fitted)
    var = np.var(fitted)
    std = np.std(fitted)
    lin = np.sum([np.abs(data[i] - e) for i, e in enumerate(fitted)])
    cov = np.cov(data, fitted, bias=True)[0][1]
    con = 2 * cov / (np.var(data) + var + np.power((np.median(data) - med), 2))

    print(
        f"""
Median      = {med}
Variance    = {var}
St. Div.    = {std}
Lin. Div.   = {lin}
Covariance  = {cov}
Concordance = {con}
"""
    )
