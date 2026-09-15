# 连续时间傅里叶变换（周期、非周期信号）
将连续时间信号 **（周期、非周期）** 分解为复指数信号分量的线性组合

### 非周期信号的连续时间傅里叶变换

1. 傅里叶变换对：
- 傅里叶变换
$$
x(t)=\frac{1}{2\pi}\int_{-\infty}^{+\infty}X(jw)e^{jwt}dw
$$
- 傅里叶逆变换$\to $频谱
$$
X(jw)=\int_{-\infty}^{+\infty}x(t)e^{-jwt}dt
$$

2. 周期信号的傅里叶级数系数
$a_{k}=\frac{1}{T}X(jw)|_{w=kw_{0}}$

3. 傅里叶变换的收敛条件
- 平方可积
- 绝对可积

### 周期信号的傅里叶变换
注：统一周期信号与非周期信号

1. ==周期信号的傅里叶变换表示==：

$x(t)=\sum_{k=-\infty}^{\infty}a_k e^{jkw_{0}t}$

$X(jw)=2\pi \sum_{k=-\infty}^{\infty}a_k \delta(w-kw_0)$

### 连续时间傅里叶变换的性质

1. 线性
2. 时移
$x(t-t_0)\leftrightarrow X(jw)e^{-jwt_0}$
3. 共轭及共轭对称性
$x^*(t)\leftrightarrow X^*(-jw)$

==时间反转==
$x(-t)\leftrightarrow X(-jw)$

4. 时域微分与积分（作用：转代数运算）

    $\frac{dx(t)}{dt}\leftrightarrow jwX(jw)$

    $\int_{-\infty}^tx(\tau)d\tau \leftrightarrow \frac{1}{jw}X(jw)+\pi X(0)\delta (w)$
5. 时域和频域的尺度变换：
$x(at)\leftrightarrow \frac{1}{|a|}X(j\frac{w}{a})$
6. **对偶性**：
$X(jt)\leftrightarrow 2\pi x(-w)$

==频域特性==

- 移频特性
$x(t)e^{jw_0t}\leftrightarrow X[j(w-w_0)]$
- 频域微分特性
$-jtx(t)\leftrightarrow \frac{d}{dw}X(jw)$
- 频域积分特性
$\frac{x(t)}{-jt}+\pi x(0)\delta(t)\leftrightarrow \int_{-\infty}^wX(j\tau)d\tau$

*时域与频域*
| 时域特性 | 频域对偶特性 |
| :--- | :--- |
| **时移特性**<br>$x(t-t_0)\leftrightarrow X(j\omega)e^{-j\omega t_0}$ | **移频特性**<br>$x(t)e^{j\omega_0 t}\leftrightarrow X[j(\omega-\omega_0)]$ |
| **时域微分**<br>$\displaystyle\frac{dx(t)}{dt}\leftrightarrow j\omega X(j\omega)$ | **频域微分**<br>$\displaystyle -jtx(t)\leftrightarrow \frac{d}{d\omega}X(j\omega)$ |
| **时域积分**<br>$\displaystyle\int_{-\infty}^t x(\tau)d\tau \leftrightarrow \frac{1}{j\omega}X(j\omega)+\pi X(0)\delta(\omega)$ | **频域积分**<br>$\displaystyle\frac{x(t)}{-jt}+\pi x(0)\delta(t)\leftrightarrow \int_{-\infty}^\omega X(j\tau)d\tau$ |

7. Parseval定理
$\int_{-\infty}^{\infty}|x(t)|^2dt=\frac{1}{2\pi}\int_{-\infty}^{\infty}|X(jw)|^2dw$

### 卷积性质
系统频域分析方法的理论基础

$x(t)*h(t)\leftrightarrow X(jw)H(jw)$

### 相乘性质
通信和信号传输邻域各种调制解调技术的理论基础

$s(t)p(t)\leftrightarrow \frac{1}{2\pi}S(jw)*P(jw)$

### 由线性常系数微分方程表征的系统

微分方程：
$\sum_{k=0}^Na_k\frac{d^ky(t)}{dt^k}=\sum_{k=0}^{M}b_k\frac{d^kx(t)}{dt^k}$
$\downarrow$
傅里叶变换：
$\sum_{k=0}^{N}a_k(jw)^kY(jw)=\sum_{k=0}^{M}b_k(jw)^kX(jw)$
$\downarrow$
卷积性质：
$H(jw)=\frac{\sum_{k=0}^{M}b_k(jw)^k}{\sum_{k=0}^{M}a_k(jw)^k}$
$\downarrow$
反变换求$h(t)$:
- 部分分式展开
- 常用变换对