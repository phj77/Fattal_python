import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Windows 환경에서 한글 깨짐 방지 및 마이너스 기호 처리
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

def phi_function(x, a, beta, noise, y_0):
    """
    calculate_scaling_factor 함수와 수학적으로 동일한 식을 계산합니다.
    x: ||∇H|| (경사도 크기)
    a: alfa * avgGrad (상수 임계값)
    y_0: x=0일 때의 scaling factor 값 (y_0 > 1.0)
    """
    cond_less = x < a
    cond_ge = x >= a
    
    phi = np.zeros_like(x, dtype=float)
    
    # x < a 구간: 새로운 제안 식 적용
    exponent = (1.0 - beta) / (y_0 - 1.0)
    # y_0 - (y_0 - 1) * (x/a)^exponent
    phi[cond_less] = y_0 - (y_0 - 1.0) * ((x[cond_less] / a) ** exponent)
    
    # x >= a 구간: 기존 Fattal 식 적용
    phi[cond_ge] = ((x[cond_ge] + noise) / a) ** (beta - 1.0)
    
    return phi

def create_plot(beta, a, noise, y0_values, title, save_path):
    x = np.linspace(0, 2.5 * a, 1000)
    
    plt.figure(figsize=(9, 6), dpi=150)
    
    # 제안된 modified 식 (세 가지 y_0 케이스) - 모두 실선으로 표시
    colors = ['blue','orange', 'green']
    for y_0, color in zip(y0_values, colors):
        y = phi_function(x, a, beta, noise, y_0)
        plt.plot(x, y, color=color, linestyle='-', linewidth=2.0, alpha=0.75, zorder=2)
        
    # 기존 식 (x >= a 구간 식을 전 구간으로 확장)을 빨간색 실선으로 렌더링 (가장 위층에 렌더링되도록 zorder=3 설정)
    y_original = ((x + noise) / a) ** (beta - 1.0)
    plt.plot(x, y_original, color='red', linestyle='-', linewidth=2.5, zorder=3)
    
    # x = a 임계값 (시각화에서는 \alpha로 표시) 및 y = 1 기준선 점선 표시
    plt.axvline(x=a, color='#868E96', linestyle=':', linewidth=1.5)
    plt.text(a + 0.05, 0.1, r'$\alpha$', color='#495057', fontsize=10, fontweight='bold')
    
    plt.axhline(y=1.0, color='#868E96', linestyle=':', linewidth=1.5)
    plt.text(0.05, 1.05, r'$\Phi = 1$', color='#495057', fontsize=10, fontweight='bold')
    
    # 기존 식 전구간 확장선 안내 텍스트 표시
    max_y = max(y0_values)
    
    # 축 설정
    plt.xlabel('$||\\nabla H||$ (Gradient Magnitude)', fontsize=12, labelpad=10)
    plt.ylabel('$\\Phi$ (Scaling Factor)', fontsize=12, labelpad=10)
    
    # 0,0 눈금만 남김
    plt.xticks([0])
    plt.yticks([0])
    
    plt.xlim(0, 2.5 * a)
    # y축 범위 조절
    plt.ylim(0, max_y * 1.2)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Successfully saved plot to: {save_path}")

def main():
    # 출력 경로 설정
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 공통 설정
    a = 1.0
    noise = 0.001
    
    # 실험 1: beta = 0.8 일 때 (2 - beta = 1.2)
    # y_0 > 1.0 조건 만족
    beta1 = 0.8
    y0_cases_1 = [
        2.0 - beta1,  # 1.2 (직선)
        1.5,          # > 1.2 (아래로 볼록)
        1.05          # < 1.2 이면서 > 1.0 (위로 볼록)
    ]
    path1 = os.path.join(current_dir, 'scaling_factor_beta_0.8.png')
    create_plot(
        beta=beta1,
        a=a,
        noise=noise,
        y0_values=y0_cases_1,
        title=f'Scaling Factor $\\Phi$ vs $||\\nabla H||$ ($\\beta = {beta1}$)',
        save_path=path1
    )
    
    # 실험 2: beta = 0.5 일 때 (2 - beta = 1.5)
    # y_0 > 1.0 조건 만족
    beta2 = 0.5
    y0_cases_2 = [
        2.0 - beta2,  # 1.5 (직선)
        1.9,          # > 1.5 (아래로 볼록)
        1.4           # < 1.5 이면서 > 1.0 (위로 볼록)
    ]
    path2 = os.path.join(current_dir, 'scaling_factor_beta_0.5.png')
    create_plot(
        beta=beta2,
        a=a,
        noise=noise,
        y0_values=y0_cases_2,
        title=f'Scaling Factor $\\Phi$ vs $||\\nabla H||$ ($\\beta = {beta2}$)',
        save_path=path2
    )

if __name__ == '__main__':
    main()
