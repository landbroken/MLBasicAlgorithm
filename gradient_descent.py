import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-1, 6, 141)
y = (x - 2.5) ** 2 - 1


# 每点梯度
def dj(theta):
    return 2 * (theta - 2.5)


# 每点的Y值
def J(theta):
    try:
        return (theta - 2.5) ** 2 - 1
        # 防止J越来越大
    except:
        return float('inf')
        # 梯度下降，将theta的值记录下来，定义最大迭代次数和允许的最小误差


def gradient_descent(initial_theta, eta, n_iters=1e4, error=1e-8):
    theta = initial_theta
    theta_hist.append(initial_theta)
    i_iter = 0
    while i_iter < n_iters:
        gradient = dj(theta)
        last_theta = theta
        theta = theta - eta * gradient
        theta_hist.append(theta)
        if abs(J(theta) - J(last_theta)) < error:
            break
        i_iter += 1
        # 绘制原始曲线和梯度下降过程


def plot_thetahist():
    plt.plot(x, J(x))
    plt.plot(np.array(theta_hist), J(np.array(theta_hist)), color='r', marker='+')
    plt.show()

theta_hist = []
# eta:学习率,步长
gradient_descent(0, eta=0.1, n_iters=10)
plot_thetahist()
