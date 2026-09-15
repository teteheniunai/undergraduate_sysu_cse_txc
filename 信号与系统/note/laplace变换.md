# 拉普拉斯变换

### 拉普拉斯变换

1. def：
- 正变换：$X(s)=\int_{\infty}^{\infty}x(t)e^{-st}dt$,其中$s=\delta+jw$ 
- 反变换：$x(t)=\frac{1}{2\pi j}\int^{\delta+jw}_{\delta-jw}X(s)e^{st}ds$

2. 信号响应
- $y(t)=H(s)e^{st}$
- $y[n]=H(z)z^{n}$

3. 与傅里叶变换关系：$X(s)=F(x(t)e^{-\delta t})$

4. 零极点图：矩形区域

### lappace T的ROC（**必须考虑**，加上列式才能进行一一对应）

1. ①绝对可积②时限信号的ROC是整个平面
2. **收敛域包括了jw轴/绝对收敛**，则x(t)的FT存在

### 由零极点图->F几何求值

1. 
$X(s)=\frac{N(s)}{D(s)}=M\frac{\prod^R_{i=1}(s-\beta_i)}{\prod^P_{j=1}(s-\alpha_i)}$
- $|X(s_1)|=|M|\frac{\prod^R_{i=1}|s-\beta_i|}{\prod^P_{j=1}|s-\alpha_i|}$
- $\angle X(s_1)=\sum^R_{i=1}\angle (\overrightarrow{s_1}-\overrightarrow{\beta_i})-\sum^P_{j=1}(\overrightarrow{s_1}-\overrightarrow{\alpha_j})$
总之记住零点幅乘角加，极点幅除角减

2. 例子
- 一阶系统：因果，右边
- 全通系统：$|H(jw)|=1$,零点极点关于jw轴对称

### laplace变换性质（**考虑ROC**）

1. 线性：ROC——至少$R_1 \cap R_2$
2. 时移：ROC不变
3. s域平移：ROC——R+Re[$s_0$]
4. 时域尺度变换：$x(at)\leftrightarrow \frac{1}{|a|}X(\frac{s}{a})$,ROC——aR
- 特例：$x(-t)\leftrightarrow X(-s)$,ROC——-R
5. 共轭对称性：$x*(t)\leftrightarrow X*(s*)$,ROC——R
- 实信号LT复数零、极点必共轭成对出现
6. 卷积性质：$x_1(t)*x_2(t)\leftrightarrow X_1(s)X_2(s)$,ROC包括$R_1\cap R_2$
7. 时域微分：$\frac{dx(t)}{dt}\leftrightarrow sX(s)$,ROC包括R
8. s域微分：$-tx(t)\leftrightarrow \frac{dX(s)}{ds}$,R
- s域分母多次
9.  时域积分：$\int_{-\infty}^tx(\tau)d\tau \leftrightarrow \frac{1}{s}X(s)$,$R\cap (Re[s]\ge 0)$
10. 初值与终值定理
- 初值：①因果信号②t=0时不含奇异函数：$x(o^+)=lim_{s\rightarrow \infty}sX(s)$
- 终值：①因果②……③X(s)除了在s=0可以有单阶极点外，其余极点均在s平面的左半边：$lim_{t\rightarrow \infty}x(t)=lim_{s\rightarrow 0}sX(s)$

### LT分析与表征LTI系统

1. 系统函数/传递函数/转移函数：$H(s)$
2. 因果与反因果系统
3. 稳定系统：$\int^\infty_{-\infty}|h(t)|dt\le \infty$ 
- H(s)收敛域包括jw轴
- 因果稳定系统：H(s)收敛域包括jw轴的s右半平面

### 线性常系数微分方程描述的LTI系统

若系统满足初始松弛条件，则系统是因果LTI的
$\sum^N_{k=0}a_k\frac{d^ky(t)}{dt^k}=\sum^M_{k=0}b_k\frac{d^kx(t)}{dt^k}$
则有
$H(s)=\frac{Y(s)}{X(s)}=\frac{\sum^M_{k=0}b_ks^k}{\sum^N_{k=0}a_ks^k}$

### 单变laplace变换（**分析具有非零初始条件**）

1. 变换
- 正变换：$\chi(s)=\int^\infty_{o^-}x(t)e^{-et}dt$，ROC为最右边极点右边
- 反变换(没变)：$x(t)=\frac{1}{2\pi j}\int^{\delta+j\infty}_{\delta-j\infty}\chi(s)e^{st}ds$

2. 性质$x(t)\leftrightarrow \chi(s)$
- $\frac{dx(t)}{dt}\leftrightarrow s\chi(s)-x(0^-)$
- $\frac{d^2x(t)}{dt^2}\leftrightarrow s^2\chi(s)-s\chi(o^-)-x'(0^-)$

### ==重要例子==

1. $x(t)=e^{-at}u(t)\leftrightarrow X(s)=\frac{1}{s+a},Re[s]\ge -a$
2. $x(t)=-e^{-at}u(-t)\leftrightarrow X(s)=\frac{1}{s+a},Re[s]\le -a$
3. $u(t)\leftrightarrow \frac{1}{s}$
4. $\delta(t)\leftrightarrow 1$
5. $u_n(t)\leftrightarrow s^n$