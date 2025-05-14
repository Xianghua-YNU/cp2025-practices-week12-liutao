import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def breit_wigner(E, Er, Gamma, fr):
    """
    Breit-Wigner共振公式
    
    参数:
        E (float or numpy.ndarray): 能量(MeV)
        Er (float): 共振能量(MeV)
        Gamma (float): 共振宽度(MeV)
        fr (float): 共振强度(mb)
        
    返回:
        float or numpy.ndarray: 共振截面(mb)
    """
    return fr / ((E - Er)**2 + Gamma**2 / 4)

def fit_without_errors(energy, cross_section):
    """
    不考虑误差的Breit-Wigner拟合
    
    参数:
        energy (numpy.ndarray): 能量数据(MeV)
        cross_section (numpy.ndarray): 截面数据(mb)
        
    返回:
        tuple: 包含以下元素的元组
            - popt (array): 拟合参数 [Er, Gamma, fr]
            - pcov (2D array): 参数的协方差矩阵
    """
    # 初始猜测值
    Er_guess = 75.0
    Gamma_guess = 50.0
    fr_guess = 10000.0
    
    popt, pcov = curve_fit(breit_wigner, energy, cross_section, 
                          p0=[Er_guess, Gamma_guess, fr_guess])
    return popt, pcov

def fit_with_errors(energy, cross_section, errors):
    """
    考虑误差的Breit-Wigner拟合
    
    参数:
        energy (numpy.ndarray): 能量数据(MeV)
        cross_section (numpy.ndarray): 截面数据(mb)
        errors (numpy.ndarray): 误差数据(mb)
        
    返回:
        tuple: 包含以下元素的元组
            - popt (array): 拟合参数 [Er, Gamma, fr]
            - pcov (2D array): 参数的协方差矩阵
    """
    # 初始猜测值
    Er_guess = 75.0
    Gamma_guess = 50.0
    fr_guess = 10000.0
    
    popt, pcov = curve_fit(breit_wigner, energy, cross_section, 
                          p0=[Er_guess, Gamma_guess, fr_guess],
                          sigma=errors, absolute_sigma=True)
    return popt, pcov

def plot_fit_results(energy, cross_section, errors, popt, pcov, title):
    """
    绘制拟合结果
    
    参数:
        energy (numpy.ndarray): 能量数据
        cross_section (numpy.ndarray): 截面数据
        errors (numpy.ndarray): 误差数据
        popt (array): 拟合参数
        pcov (2D array): 协方差矩阵
        title (str): 图表标题
    """
    plt.figure(figsize=(10, 6))
    
    # 绘制数据点
    plt.errorbar(energy, cross_section, yerr=errors, fmt='o', 
                color='blue', markersize=5, ecolor='gray',
                elinewidth=1, capsize=2, label='Experimental Data')
    
    # 绘制拟合曲线
    E_fit = np.linspace(min(energy), max(energy), 500)
    cross_section_fit = breit_wigner(E_fit, *popt)
    plt.plot(E_fit, cross_section_fit, '-', color='red', 
             linewidth=2, label='Fitted Curve')
    
    # 添加参数信息
    Er, Gamma, fr = popt
    Er_std = np.sqrt(pcov[0, 0])
    Gamma_std = np.sqrt(pcov[1, 1])
    fr_std = np.sqrt(pcov[2, 2])
    
    plt.text(0.05, 0.95, 
             f'$E_r$ = {Er:.1f} ± {1.96*Er_std:.1f} MeV (95% CI)\n'
             f'$\Gamma$ = {Gamma:.1f} ± {1.96*Gamma_std:.1f} MeV (95% CI)\n'
             f'$f_r$ = {fr:.0f} ± {1.96*fr_std:.0f} (95% CI)',
             transform=plt.gca().transAxes, 
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.xlabel('Energy (MeV)')
    plt.ylabel('Cross Section (mb)')
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    return plt.gcf()

def main():
    # 实验数据
    energy = np.array([0, 25, 50, 75, 100, 125, 150, 175, 200])
    cross_section = np.array([10.6, 16.0, 45.0, 83.5, 52.8, 19.9, 10.8, 8.25, 4.7])
    errors = np.array([9.34, 17.9, 41.5, 85.5, 51.5, 21.5, 10.8, 6.29, 4.14])
    
    # 任务1：不考虑误差的拟合
    popt1, pcov1 = fit_without_errors(energy, cross_section)
    fig1 = plot_fit_results(energy, cross_section, errors, popt1, pcov1,
                          'Breit-Wigner Fit (Without Errors)')
    
    # 任务2：考虑误差的拟合
    popt2, pcov2 = fit_with_errors(energy, cross_section, errors)
    fig2 = plot_fit_results(energy, cross_section, errors, popt2, pcov2,
                          'Breit-Wigner Fit (With Errors)')
    
    plt.show()
    
    # 任务3：结果比较
    print("\n拟合结果比较:")
    print(f"不考虑误差: Er={popt1[0]:.1f}±{1.96*np.sqrt(pcov1[0,0]):.1f} MeV (95% CI), "
          f"Γ={popt1[1]:.1f}±{1.96*np.sqrt(pcov1[1,1]):.1f} MeV (95% CI), "
          f"fr={popt1[2]:.0f}±{1.96*np.sqrt(pcov1[2,2]):.0f} (95% CI)")
    print(f"考虑误差:   Er={popt2[0]:.1f}±{1.96*np.sqrt(pcov2[0,0]):.1f} MeV (95% CI), "
          f"Γ={popt2[1]:.1f}±{1.96*np.sqrt(pcov2[1,1]):.1f} MeV (95% CI), "
          f"fr={popt2[2]:.0f}±{1.96*np.sqrt(pcov2[2,2]):.0f} (95% CI)")

if __name__ == "__main__":
    main()
