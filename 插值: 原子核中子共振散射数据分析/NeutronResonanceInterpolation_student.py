import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# 实验数据
energy = np.array([0, 25, 50, 75, 100, 125, 150, 175, 200])  # MeV
cross_section = np.array([10.6, 16.0, 45.0, 83.5, 52.8, 19.9, 10.8, 8.25, 4.7])  # mb
error = np.array([9.34, 17.9, 41.5, 85.5, 51.5, 21.5, 10.8, 6.29, 4.14])  # mb

def lagrange_interpolation(x, x_data, y_data):
    """拉格朗日多项式插值实现"""
    n = len(x_data)
    result = np.zeros_like(x, dtype=float)
    for k in range(len(x)):  # 对每个插值点单独计算
        px = x[k]
        total = 0.0
        for i in range(n):
            term = y_data[i]
            for j in range(n):
                if i != j:
                    if x_data[i] == x_data[j]:
                        continue  # 跳过相同点避免除以零
                    term *= (px - x_data[j]) / (x_data[i] - x_data[j])
            total += term
        result[k] = total
    return result

def cubic_spline_interpolation(x, x_data, y_data):
    """三次样条插值实现（使用自然边界条件）"""
    f = interp1d(x_data, y_data, kind='cubic', fill_value='extrapolate')
    return f(x)
    
def find_peak(x, y):
    """共振峰位置和FWHM计算（修复版）"""
    peak_idx = np.argmax(y)
    peak_x = x[peak_idx]
    half_max = y[peak_idx] / 2
    
    # 左半高搜索：找到最后一个低于半高的点
    left_mask = y[:peak_idx] <= half_max
    if not np.any(left_mask):
        x_left = x[0]
    else:
        left_start = np.where(left_mask)[0][-1]  # 最后一个满足条件的索引
        # 截取从left_start到峰值的片段（确保升序）
        y_left_segment = y[left_start:peak_idx+1]
        x_left_segment = x[left_start:peak_idx+1]
        # 反向插值（因为片段是降序）
        x_left = np.interp(half_max, y_left_segment[::-1], x_left_segment[::-1])
    
    # 右半高搜索：找到第一个低于半高的点
    right_mask = y[peak_idx:] <= half_max
    if not np.any(right_mask):
        x_right = x[-1]
    else:
        right_end = np.where(right_mask)[0][0] + peak_idx  # 第一个满足条件的索引
        # 截取从峰值到right_end的片段
        y_right_segment = y[peak_idx:right_end+1]
        x_right_segment = x[peak_idx:right_end+1]
        x_right = np.interp(half_max, y_right_segment, x_right_segment)
    
    fwhm = x_right - x_left
    return peak_x, fwhm

   
def plot_results():
    """结果可视化"""
    x_interp = np.linspace(0, 200, 500)
    
    # 计算插值结果
    lagrange_result = lagrange_interpolation(x_interp, energy, cross_section)
    spline_result = cubic_spline_interpolation(x_interp, energy, cross_section)
    
    # 计算共振峰参数
    lagrange_peak, lagrange_fwhm = find_peak(x_interp, lagrange_result)
    spline_peak, spline_fwhm = find_peak(x_interp, spline_result)
    
    # 绘图设置
    plt.figure(figsize=(12, 6))
    plt.errorbar(energy, cross_section, yerr=error, fmt='o', color='black',
                label='Original Data', capsize=5, zorder=3)
    plt.plot(x_interp, lagrange_result, '-', label=f'Lagrange (Peak: {lagrange_peak:.1f} MeV, FWHM: {lagrange_fwhm:.1f} MeV)')
    plt.plot(x_interp, spline_result, '--', label=f'Cubic Spline (Peak: {spline_peak:.1f} MeV, FWHM: {spline_fwhm:.1f} MeV)')
    
    # 标注峰值线
    plt.axvline(lagrange_peak, color='blue', linestyle=':', alpha=0.5)
    plt.axvline(spline_peak, color='orange', linestyle=':', alpha=0.5)
    
    # 图表装饰
    plt.xlabel('Energy (MeV)', fontsize=12)
    plt.ylabel('Cross Section (mb)', fontsize=12)
    plt.title('Neutron Resonance Scattering Cross Section Analysis', fontsize=14)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim(0, 200)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_results()
