# 离散时间傅里叶变换

### 非周期信号的表示——离散时间傅里叶变换

1. 离散时间傅里叶变换对：
- 傅里叶变换/分析公式——频谱函数连续且**以$2\pi$为周期**
$X(e^{jw})=\sum_{n=-\infty}^{\infty}x[n]e^{-jwn}$
- 傅里叶反变换/综合公式——离散且非周期
$x[n]=\frac{1}{2\pi}\int_{2\pi}X(e^{jw})e^{jwn}dw$

2. 离散时间周期信号的傅里叶级数系数：一个周期内傅里叶变换等间隔采样
$a_k=\frac{1}{N}X(e^{jw})|_{w=kw_0}$

3. 离散时间傅里叶变换的收敛
- 平方可和
- 绝对可和

4. ==**常见傅里叶变换**==
- 指数信号
$x[n]=a^nu[n]\leftrightarrow X(e^{jw})=\frac{1}{1-ae^{-jw}}$
- 带绝对值指数函数
$x[n]=a^{|n|}\leftrightarrow X(e^{jw})=\frac{1-a^2}{1+a^2-2acosw}$
- 单位冲激函数
$x[n]=\delta [n]\leftrightarrow X(e^{jw})=1$
- 矩形脉冲
$$
x[n]=
\begin{cases}
1, & |n|\leq N_1 \\
0, & |n|\ge N_1 
\end{cases}
\leftrightarrow X(e^{jw})=\frac{sin[(2N_1+1)\frac{w}{2}]}{sin\frac{w}{2}}$$
- 矩形脉冲的对偶
$$x[n]=\frac{W}{\pi}sinc(\frac{Wn}{\pi})\leftrightarrow X(e^{jw})=\begin{cases}
1, & |w|< W \\
0, & W<|w|\leq \pi 
\end{cases}$$
- **常数1**
$x[n]=1\leftrightarrow X(e^{jw})=2\pi\sum_{l=-\infty}^{\infty}\delta(w-2\pi l)$

### 周期信号的傅里叶变换

1. $x[n]=e^{jkw_0n}\leftrightarrow X(e^{jw})=\sum_{l=-\infty}^{\infty}2\pi \delta (w-kw_0-2\pi l)$
注：由频移性质得到

2. 表示为傅里叶级数的*周期信号*
$x[n]=\sum_{k=<N>}a_ke^{jkw_0n}\leftrightarrow X(e^{jw})=2\pi\sum_{k=-\infty}^{\infty}a_k\delta(w-kw_0)=2\pi \sum^{\infty}_{k=-\infty}a_k\delta(w-\frac{2\pi}{N}k)$

3. 常见周期信号的傅里叶变换
- 余弦信号
$x[n]=cosw_0n\leftrightarrow X(e^{jw})=\pi \sum^\infty_{k=-\infty}[\delta(w-w_0-2\pi k)+\delta(w+w_0-2\pi k)$

- 均匀脉冲串
$x[n]=\sum^\infty_{k=-\infty}\delta(n-kN)\leftrightarrow X(e^{jw})=\frac{2\pi}{N}\sum^{\infty}_{k=-\infty}\delta(w-\frac{2\pi}{N}k)$

### 离散时间傅里叶变换的性质

- 周期性
$x[n]$（离散非周期时域）$\leftrightarrow$ $X(e^{jw})$（连续周期频域）
- 线性
- 时移：$x[n-n_0]\leftrightarrow X(e^{jw})e^{-jwn_0}$
  频移：$x[n]e^{jw_0n}\leftrightarrow X(e^{j(w-w_0)})$
- **时域反转**
$x[-n]\leftrightarrow X(e^{-jw})$
- 共轭及共轭对称性
$x*[n]\leftrightarrow X^*(e^{-jw})$

*实信号*：
- 时域：$x[n]=x^*[n]$
- 频谱为共轭偶函数：$X(e^{jw})=X^*(e^{-jw})$

    ==复数坐标表示：==
- 实部偶函数：$Re[X(e^{jw})]=Re[X(e^{-jw})]$
- 虚部是虚函数：$Im[X(e^{jw})]=-Im[X(e^{-jw})]$

    ==极坐标表示：==
- 幅度偶函数：$|X(e^{jw})|=|X(e^{-jw})|$
- 相位是奇函数：$\measuredangle X(e^{jw})=-\measuredangle X(e^{-jw})$

    ==奇偶部分表示==
$x[n]=x_e[n]+x_o[n]\leftrightarrow X(e^{jw})=X_e(e^{jw})+jX_o(e^{jw})$
其中，$x_e(t)\leftrightarrow X_e(e^{jw}),x_o(t)\leftrightarrow jX_o(e^{jw})$

*奇/偶信号*：傅里叶变换也是奇/偶函数

- 时域内插：
$$
x_k[n]=\begin{cases}
x[n/k] & n为k的整数倍\\
0 & 其他n
\end{cases},
x_k[n]\leftrightarrow X(e^{jkw})$$
- 频域微分：
$nx[n]\leftrightarrow j\frac{dX(e^{jw})}{dw}$
- Parseval定理：
$\sum_{n=-\infty}^\infty |x[n]|^2=\frac{1}{2\pi}\int_{2\pi}|X(e^{jw})|^2dw$

### 卷积性质
> 频域分析得理论基础

1. $x[n]*h[n]\leftrightarrow X(e^{jw})H(e^{jw})$
2. **频域分析法**
- $x[n]\rightarrow X(e^{jw})$: $X(e^{jw})=\sum_{n=-\infty}^{\infty}x[n]e^{-jwn}$
- 求$H(e^{jw})$（频率响应）: $H(e^{jw})=\sum_{n=-\infty}^{\infty}h[n]e^{-jwn}$
- $Y(e^{jw})=X(e^{jw})H(e^{jw})$
- $y[n]=\mathcal{F}^{-1}[Y(e^{jw})]$: $y[n]=\frac{1}{2\pi}\int_{2\pi}Y(e^{jw})e^{jwn}dw$

### 相乘性质
> 为离散时间信号传输提供了理论基础

1. $y[n]=x_1[n]x_2[n]\leftrightarrow Y(e^{jw})=\frac{1}{2\pi}\int_{2\pi}X_1(e^{j\theta})X_2(e^{j(w-\theta)})d\theta=\frac{1}{2\pi}X_1(e^{jw})\otimes X_2(e^{jw})$(周期卷积)
2. 举例
$c[n]=(-1)^n=e^{j\pi n}\leftrightarrow C(e^{jw})=2\pi \sum_{k=-\infty}^{\infty}\delta (w-\pi-2k\pi)$(冲激串的周期延拓)
$\therefore Y(e^{jw})=\int_{2\pi}X(e^{j\theta})2\pi \sum_{k=-\infty}^{\infty}\delta (w-\pi-2k\pi)d\theta=\int_{w-2\pi}^{w}X(e^{j\theta})\delta (w-\theta-\pi)d\theta =X(e^{j(w-\pi)})$
> 注：取$2\pi$积分区间，其中仅包含一个冲激串

### 对偶性质

1. 离散时间傅里叶级数的对偶
$x[n]\overset{\text{DFS}}{\longleftrightarrow}X(e^{jw})$
$a_n\overset{\text{DFS}}{\longleftrightarrow}\frac{1}{N}x[-k]$

2. 离散时间傅里叶变换与连续时间傅里叶级数间的对偶
$X(e^{jw})=\sum^\infty_{n=-\infty}x[n]e^{-jwn}$是周期为$2\pi$的连续函数，将其视为连续时间信号得
$X(e^{jt})=\sum^\infty_{k=-\infty}a_ke^{jkt}$
$\therefore a_k=\frac{1}{2\pi}\int_{2\pi}X(e^{jt})e^{-jkt}dt=x[-k]$
最终
$x[n]\overset{\text{DTFT}}{\longleftrightarrow}X(e^{jw})$
$X(e^{jt})\overset{\text{CFS}}{\longleftrightarrow}x[-k]$

3. 信号在时域特性和在频域特性之间对应关系

| 时域       | | 频域       |
| :--------- | :------: | :--------- |
| 周期性     |    ↔     | 离散性     |
| 非周期性   |    ↔     | 连续性     |
| 离散性     |    ↔     | 周期性     |
| 连续性     |    ↔     | 非周期性   |

### 由线性常系数差分方程表征得系统

$\sum_{k=0}^Na_ky[n-k]=\sum_{k=0}^Mb_kx[n-k]$
$H(e^{jw})=\frac{\sum^M_{k=0}b_ke^{-jkw}}{\sum^N_{k=0}a_ke^{-jkw}}$


