## 线性时不变系统
1. ⭐$$x[n]=\sum_{k=-\infty}^{\infty}x[k]\delta[n-k]\rightarrow y[n]=\sum_{k=-\infty}^{\infty}x[k]h[n-k]=x[n]*h[n]$$
>使用$\delta[n]\ or\ \delta(t)$替换$x[n]\ or\ x(t)$

2. **卷积计算**：重叠部分乘积和
- 等比数列求和
- 使用阶跃函数表达输入信号

3. 卷积性质
- 交换律：$x[n]*h[n]=h[n]*x[n]$
- 分配律：$h_1[n]+h_2[n]$
- 结合率：级联
  - 级联次序可交换（当LTI+涉及卷积计算收敛）
- **时域反转性**：$x(t)*h(t)=y(t)\rightarrow x(-t)*h(-t)=y(-t)$
- **微分**、**积分**、时移

4. ==LTI系统性质==
- 无记忆：$h[n]=k\delta[n]$
- 可逆性：$h(t)*g(t)=\delta(t)$
- 因果性：$h[n]=0,n<0$
- 稳定性：$\sum_{n=-\infty}^{\infty}|h[n]|<\infty$，则$y[n]$有界

### 微分和差分方程描述因果LTI
1. 求解
2. 方框图表示

### 奇异函数
1. $\delta(t)$
- 卷积定义：$x(t)=x(t)*\delta(t)$ and $\int_{-\infty}^{\infty}\delta(\tau)d\tau=1$
- 积分定义：$g(0)=\int_{-\infty}^{\infty}g(\tau)\delta(\tau)d\tau$
- 性质
  - $f(t)\delta(t)=f(0)\delta(0)$
  - $\delta(at)=\frac{1}{|a|}\delta(t)$
  - ==偶函数==—— $\delta(t)=\delta(-t)$
2. 单位冲激偶$u_1(t)=\frac{d\delta(t)}{dt}$
- 运算定义：$x(t)*u_1(t)=\frac{d}{dt}x(t)$
- 短脉冲求导
- 性质
  - $x(t)=1\rightarrow \int_{-\infty}^{\infty}u_1(t)dt=0$
  - $-g'(0)=\int_{-\infty}^{\infty}g(\tau)u_1(\tau)d\tau$
3. $u_{-1}(t)=\int_{-\infty}^{t}\delta(\tau)d\tau$
- 运算定义：$x(t)*u_{-1}(t)=\int_{-\infty}^tx(\tau)d\tau=u(t)$
4. 单位斜坡函数$u_{-2}=u_{-1}*u_{-1}=\int^{t}_{-\infty}u(t)dt=tu(t)$
- 运算定义：$x(t)*u_{-2}(t)=\int^t_{-\infty}(\int^{\tau}_{-\infty}x(\delta)d\delta)d\tau$
  