import numpy as np
import os
import random
import matplotlib.pyplot as plt
import psutil, os
import sys

sys.stdout.reconfigure(encoding='utf-8')
process = psutil.Process(os.getpid())

# 遗传算法类
class GeneticAlgTSP:
    def __init__(self, filename, pop_size=100, pc=0.9, pm=0.1):
        self.pop_size = pop_size #种群大小
        self.pc = pc  # 交叉概率
        self.pm = pm  # 变异概率

        self.cities = self._load_tsp_data(filename)
        self.n = len(self.cities)
        
        self.dist_mat = self._calc_dist_matrix(self.cities)
        
        self.population = [random.sample(range(self.n), self.n) for _ in range(pop_size)]
        
        # 用于记录全局最优
        self.best_route = None
        self.min_dist = float('inf')
    
    # 数据解析器 城市编号 经度/X坐标 纬度/Y坐标
    def _load_tsp_data(self, filename):
        coords = []

        if not os.path.exists(filename):
             raise FileNotFoundError(f"找不到文件: {filename}")

        with open(filename, 'r') as f:
            lines = f.readlines()
            start = False
            for line in lines:
                if "NODE_COORD_SECTION" in line:
                    start = True
                    continue
                if "EOF" in line or line.strip() == "EOF" or line.strip() == "":
                    if start: break 
                    continue
                if start:
                    parts = line.split()
                    if len(parts) >= 3: 
                        coords.append([float(parts[1]), float(parts[2])])
        return np.array(coords)

    # 距离矩阵计算
    def _calc_dist_matrix(self, coords):
        n = len(coords)
        dist_mat = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = np.sqrt(np.sum((coords[i] - coords[j])**2))
                dist_mat[i, j] = dist_mat[j, i] = d
        return dist_mat

    #计算适应度（1/总距离）
    def fitness(self, route):
        dist = sum(self.dist_mat[route[i], route[(i+1)%self.n]] for i in range(self.n))
        return 1.0 / dist, dist

    # 锦标赛选择 (Tournament Selection)——挑出优秀个体
    def selection(self, fitnesses):
        selected = []
        for _ in range(self.pop_size):
            candidates = random.sample(list(range(self.pop_size)), 5)
            best = max(candidates, key=lambda i: fitnesses[i])
            selected.append(self.population[best].copy())
        return selected

    #部分映射交叉⭐
    def pmx_crossover(self, p1, p2):
    
        if random.random() > self.pc: return p1.copy()
        c = [None] * self.n
        start, end = sorted(random.sample(range(self.n), 2))
        c[start:end+1] = p1[start:end+1]
        p2_pos = {val: i for i, val in enumerate(p2)}
        
        for i in range(start, end+1):
            if p2[i] not in p1[start:end+1]:
                curr = p2[i]
                pos = i
                
                while start <= pos <= end:
                    val_in_p1 = p1[pos]  
                    pos = p2_pos[val_in_p1]
                c[pos] = curr
        
        for i in range(self.n):
            if c[i] is None: c[i] = p2[i]
        return c

    #变异操作
    def inversion_mutation(self, route):
        if random.random() < self.pm:
            start, end = sorted(random.sample(range(self.n), 2))
            route[start:end+1] = route[start:end+1][::-1]
        return route
    
    #局部搜索启发式算法
    def two_opt(self, route):
        best_route = list(route)
        # 只进行有限次的尝试
        for _ in range(20): 
            i, j = sorted(random.sample(range(1, self.n - 1), 2))
            
            A, B = best_route[i-1], best_route[i]
            C, D = best_route[j], best_route[(j+1)%self.n]
            
            d_old = self.dist_mat[A, B] + self.dist_mat[C, D]
            d_new = self.dist_mat[A, C] + self.dist_mat[B, D]
            
            if d_new < d_old:
                best_route[i:j+1] = reversed(best_route[i:j+1])
        return best_route

    # 求解方法：基于当前种群迭代，并返回 1-n 格式的解
    def iterate(self, generations):
        best_dist_history = []

        for g in range(generations):
            fits_and_dists = [self.fitness(ind) for ind in self.population]
            fits = [f[0] for f in fits_and_dists]
            dists = [f[1] for f in fits_and_dists]
            
            # 记录最优
            current_min = min(dists)
            if current_min < self.min_dist:
                self.min_dist = current_min
                self.best_route = self.population[dists.index(current_min)].copy()
            best_dist_history.append(self.min_dist)

            # 生成下一代
            new_pop = self.selection(fits)
            next_generation = []
            for i in range(0, self.pop_size, 2):
                p1, p2 = new_pop[i], new_pop[(i+1)%self.pop_size]
                next_generation.append(self.inversion_mutation(self.pmx_crossover(p1, p2)))
                next_generation.append(self.inversion_mutation(self.pmx_crossover(p2, p1)))
            #精英保留机制：将历史绝对最优解覆盖到新种群的第一个位置
            next_generation[0] = self.best_route.copy()
            next_generation[0] = self.two_opt(next_generation[0])
            self.population = next_generation
            
            if g % 100 == 0: print(f"Gen {g}: Best Distance = {self.min_dist:.2f}")
            
        return [city + 1 for city in self.best_route], best_dist_history

def plot_route(coords, route_1_n, title):
    # 将 1-n 转回 0-(n-1) 用于绘图索引
    route_0_idx = [c - 1 for c in route_1_n]
    ordered_coords = coords[route_0_idx + [route_0_idx[0]]] 
    plt.figure(figsize=(8, 8))
    plt.plot(ordered_coords[:, 0], ordered_coords[:, 1], 'ro-')
    plt.title(title)
    plt.show()

#小型用于快速调试，验证代码逻辑是否能跑通。
#中型用于评估算法性能。
file_name = "qa194.tsp"
file_path = os.path.join(os.path.dirname(__file__), "data", file_name)

seeds = [42, 10, 123] 
#seeds=[42]
results = []
best_path_overall = None
min_dist_overall = float('inf')
plt.figure(figsize=(10, 6))

for s in seeds:
    random.seed(s)      
    np.random.seed(s)   
    
    #注意：城市越多需要调大pop_size500+和evolve2000+来使结果波动小
    ga = GeneticAlgTSP(file_path, pop_size=500, pc=0.9, pm=0.1) 
    best_path, history = ga.iterate(1500)
    
    # 绘制当前种子的收敛曲线
    plt.plot(history, label=f'Seed {s}')

    final_dist = history[-1]
    results.append(final_dist)

    if final_dist < min_dist_overall:
        min_dist_overall = final_dist
        best_path_overall = best_path

    print(f"种子 {s} 完成，最优距离: {final_dist:.2f}")

plt.title(f'Genetic Algorithm Convergence ({file_name})', fontsize=14)
plt.xlabel('Generation', fontsize=12)
plt.ylabel('Best Distance', fontsize=12)
plt.legend()  
plt.grid(True, linestyle='--', alpha=0.6)

plt.savefig('convergence_curve.png') 
plt.show()

print(f"\n实验汇总 ({len(seeds)}次运行):")
print(f"最佳距离: {min(results):.2f}")
print(f"平均距离: {np.mean(results):.2f}")
print(f"标准差: {np.std(results):.2f}")

print(f"\n较优路线 :")
print(best_path_overall)

plot_route(ga.cities, best_path_overall, f"Best Route for {file_name}")
print(f"内存占用：{process.memory_info().rss / 1024 / 1024:.2f} MB")