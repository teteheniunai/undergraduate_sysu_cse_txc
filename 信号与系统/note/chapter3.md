## 周期信号的傅里叶级数（周期信号）表示

1. ==傅里叶级数：==
$$ x(t)=\sum_ka_ke^{jkw_0t}\rightarrow y(t)=\sum_ka_kH(jkw_0)e^{jkw_0t}\\
x[n]=\sum_ka_ke^{j\frac{2\pi}{N}kn}\rightarrow y[n]=\sum_ka_kH(e^{j\frac{2\pi}{N}k})e^{jk\frac{2\pi}{N}n}$$

其中，
$ H(jw)=\int^{\infty}_{-\infty}h(t)e^{-jwt}dt$
$H(e^{jw})=\sum_{-\infty}^{\infty}h[n]e^{-jwn}$,**对$w$而言是以$2\pi$为周期**

2. 频域表示法：频谱图（条线图）
3. **傅里叶级数系数**：$a_k=\frac{1}{T}\int_Tx(t)e^{-jkw_0t}dt$
4. 傅里叶级数收敛的充分条件
- 平方可积
- 迪利克雷条件
  - 绝对收敛
- **Gibbs现象**

### 滤波器
1. 频率成形滤波器
$H(jw)=\int_{-\infty}^{\infty}u_1(t)e^{-jwt}dt=jw$
2. 频率选择性滤波器
- 低通
- 高通
- 带通
