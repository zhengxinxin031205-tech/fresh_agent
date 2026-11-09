"""
标准正态分布工具(纯 math 实现,替代 scipy.stats.norm)
=====================================================
仅提供项目所需的 pdf / cdf / ppf 三个函数,
避免引入 scipy(约 66MB 打包体积)。
"""
import math
import numpy as np

_SQRT_2PI = math.sqrt(2.0 * math.pi)
_SQRT_2 = math.sqrt(2.0)


def norm_pdf(x: float) -> float:
    """标准正态分布概率密度函数"""
    return math.exp(-0.5 * x * x) / _SQRT_2PI


def norm_cdf(x: float) -> float:
    """标准正态分布累积分布函数"""
    return 0.5 * math.erfc(-x / _SQRT_2)


def norm_ppf(p: float) -> float:
    """
    标准正态分布分位数函数(逆CDF)
    采用 Acklam 有理逼近算法,相对误差 < 1.15e-9
    """
    if p <= 0.0:
        return -math.inf
    if p >= 1.0:
        return math.inf

    # Acklam 算法系数
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
          4.374664140464868e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]

    p_low = 0.02425
    p_high = 1.0 - p_low

    if p < p_low:
        # 左尾逼近
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
               ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)
    elif p <= p_high:
        # 中央区域逼近
        q = p - 0.5
        r = q * q
        return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / \
               (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1.0)
    else:
        # 右尾逼近
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)


class SimpleLinearRegression:
    """
    最小二乘线性回归(替代 sklearn.linear_model.LinearRegression)
    ============================================================
    仅实现项目所需的 fit / predict / score / coef_,
    基于 numpy.linalg.lstsq,避免引入 sklearn(约 12MB 打包体积)。
    """
    def __init__(self):
        self.coef_ = None
        self.intercept_ = 0.0

    def fit(self, X, y):
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        y_arr = np.asarray(y, dtype=float)
        # 追加截距列
        X_biased = np.column_stack([X_arr, np.ones(len(X_arr))])
        coef, _, _, _ = np.linalg.lstsq(X_biased, y_arr, rcond=None)
        self.coef_ = coef[:-1]
        self.intercept_ = float(coef[-1])
        return self

    def predict(self, X):
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        return X_arr @ self.coef_ + self.intercept_

    def score(self, X, y):
        """R² 决定系数"""
        y_arr = np.asarray(y, dtype=float)
        y_pred = self.predict(X)
        ss_res = float(np.sum((y_arr - y_pred) ** 2))
        ss_tot = float(np.sum((y_arr - y_arr.mean()) ** 2))
        return 1.0 - ss_res / max(ss_tot, 1e-10)