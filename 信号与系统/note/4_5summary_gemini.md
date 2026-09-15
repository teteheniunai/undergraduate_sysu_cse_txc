## 一、对偶性大统一

**一个域的“离散”对应另一个域的“周期”；一个域的“连续”对应另一个域的“非周期”。**

| 变换类型 | 时域特性 | 频域特性 | 典型特征 |
| :--- | :--- | :--- | :--- |
| **CTFS** | 连续、**周期** | **离散**、非周期 | 谱线图（只有 $n\omega_0$ 处有值） |
| **CTFT** | 连续、非周期 | 连续、非周期 | 频谱是密集的曲线 |
| **DTFT** | **离散**、非周期 | 连续、**周期** | 频谱以 $2\pi$ 为周期 |
| **DFS** | **离散**、**周期** | **离散**、**周期** | 时频两域全是“铁丝网” |

---

## 二、 CTFT vs. DTFT：关键性质深度对比表

这是考试中最常考的性质对比，注意观察它们的**细微差别**。

| 性质 | 连续时间 (CTFT) | 离散时间 (DTFT) | **考试陷阱/重点** |
| :--- | :--- | :--- | :--- |
| **时移** | $x(t-t_0) \leftrightarrow X(j\omega)e^{-j\omega t_0}$ | $x[n-n_0] \leftrightarrow X(e^{j\omega})e^{-j\omega n_0}$ | 极其重要！相位移动是频率的线性函数。 |
| **频移** | $x(t)e^{j\omega_0 t} \leftrightarrow X(j(\omega-\omega_0))$ | $x[n]e^{j\omega_0 n} \leftrightarrow X(e^{j(\omega-\omega_0)})$ | 调制解调的基础。 |
| **微分/差分** | $\frac{dx(t)}{dt} \leftrightarrow j\omega X(j\omega)$ | $x[n]-x[n-1] \leftrightarrow (1-e^{-j\omega})X(e^{j\omega})$ | CTFT 把微分变代数；DTFT 对应差分。 |
| **频域微分** | $-jtx(t) \leftrightarrow \frac{dX(j\omega)}{d\omega}$ | $nx[n] \leftrightarrow j\frac{dX(e^{j\omega})}{d\omega}$ | 注意 DTFT 前面那个 $j$ 不要丢了。 |
| **积分/累加** | $\displaystyle \int_{-\infty}^t x(\tau)d\tau \leftrightarrow \frac{X(j\omega)}{j\omega} + \pi X(0)\delta(\omega)$ | $\displaystyle \sum_{k=-\infty}^n x[k] \leftrightarrow \frac{X(e^{j\omega})}{1-e^{-j\omega}} + \pi X(e^{j0})\sum_{k=-\infty}^{\infty} \delta(\omega - 2\pi k)$| **高频考点**：必须要记住后面的 $\delta$ 项（直流分量）。 |
| **尺度变换** | $x(at) \leftrightarrow \frac{1}{\|a\|}X(j\frac{\omega}{a})$ | $x_k[n] \leftrightarrow X(e^{j\omega k})$ | **陷阱**：离散时间只有“插零”扩展，没有连续缩放。 |
| **卷积** | $x*h \leftrightarrow X \cdot H$ | $x*h \leftrightarrow X \cdot H$ | 系统分析的灵魂。 |
| **乘积** | $x \cdot y \leftrightarrow \frac{1}{2\pi} X * Y$ | $x \cdot y \leftrightarrow \frac{1}{2\pi} X \otimes Y$ | **重点**：DTFT 的频域卷积是**周期卷积**。 |

---

## 三、 常考重点与避坑指南

### 1. 周期信号的 FT 表示（统一论）
考试经常要求你用 CTFT/DTFT 的公式去表示周期信号。
* **秘诀**：周期信号在频域一定表现为**冲激串** $\delta(\dots)$。
* **CTFT** 的冲激在 $k\omega_0$。
* **DTFT** 的冲激不仅在 $k\omega_0$，还要记得**每隔 $2\pi$ 周期重复**。

### 2. 能量/功率（Parseval 定理）
* **考法**：题目给出一个复杂的 $x[n]$ 或 $x(t)$，让你求能量，但直接积分/求和很难。
* **对策**：转换到频域，求频谱模平方的积分。
* **注意**：CTFT 前面有 $\frac{1}{2\pi}$，DTFT 积分区间是 $2\pi$ 长度。

### 3. 对偶性
CTFT 的对偶性最强：若 $x(t) \leftrightarrow X(j\omega)$，则 $X(jt) \leftrightarrow 2\pi x(-\omega)$。
* **应用场景**：当你看到一个时域信号长得像某个频谱（比如 $sinc$ 函数），直接套对偶性出答案，别傻傻地去算积分。

### 4. 离散插值（Interpolation）
DTFT 的 $x_k[n]$ 性质是难点：
* 时域每两个点之间插 $k-1$ 个零 $\implies$ 频域压缩（波形变瘦），周期变成原来的 $1/k$。
* 这个性质在数字信号处理（DSP）的多抽样率系统中是绝对的考点。

### 5. 理想低通滤波器（LPF）
* 时域是 $sinc$ 函数，频域是矩形窗。
* **重点**：记住 $sinc$ 函数过零点的位置（CTFT 是 $\frac{\pi}{W}$，DTFT 注意 $\omega$ 的单位）。

---

## 四、 记忆口诀

> **连续非周频谱散，离散非周频谱连（且周期）。**
> **时域平移挂指数，频域平移加旋转。**
> **微分差分乘因子，卷积相乘两重天。**
> **直流分量莫忘记，积分累加带冲激。**
