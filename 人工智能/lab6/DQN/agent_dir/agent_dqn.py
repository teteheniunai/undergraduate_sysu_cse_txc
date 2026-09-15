import os
import random
import copy
import csv
import time
import numpy as np
import torch
from pathlib import Path
from tensorboardX import SummaryWriter
from torch import nn, optim
from agent_dir.agent import Agent

# ------------神经网络模型Q(s,a)---------------
class QNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(QNetwork, self).__init__()
        # 三层全连接 MLP
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, inputs):
        return self.net(inputs)

# --------------经验放回池--------------------
class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = []
        self.position = 0 # 循环写入指针

    def __len__(self):
        return len(self.buffer)

    # 存入经验(s,a,r,s′,done)
    def push(self, *transition):
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(None)

        self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.buffer_size

    # 随机采样
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        # 转为numpy数组，方便后续转tensor
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        )
    
    def clean(self):
        self.buffer = []
        self.position = 0

#---------------优先经验放回--------------------
# α 控制优先程度（α=0 退化为普通均匀回放），β 控制权重修正强度——抵消高频采样高优样本带来的权重偏离（训练后期逐渐升到 1）
class PrioritizedReplayBuffer:
    def __init__(self, buffer_size, alpha=0.6, eps=1e-6):
        self.buffer_size = buffer_size
        self.alpha = alpha # 优先级放大系数
        self.eps = eps # 极小值，防止TD误差为0时优先级=0，无法被采样
        self.buffer = [] 
        self.priorities = np.zeros((buffer_size,), dtype=np.float32)
        self.position = 0

    def __len__(self):
        return len(self.buffer)

    # 存入一条交互转移样本
    def push(self, *transition):
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(None)

        # 获取当前池中最大优先级
        max_priority = self.priorities[:len(self.buffer)].max() if len(self.buffer) > 0 else 1.0
        if max_priority <= 0:
            max_priority = 1.0

        # 写入样本和对应优先级
        self.buffer[self.position] = transition
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.buffer_size

    def sample(self, batch_size, beta=0.4):
        current_size = len(self.buffer)
        priorities = self.priorities[:current_size]
        probs = priorities ** self.alpha # 加权优先级：α调节优先强度
        probs = probs / probs.sum() # 归一化

        indices = np.random.choice(current_size, batch_size, p=probs)
        batch = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*batch)

        # 计算重要性采样权重 IS weights
        weights = (current_size * probs[indices]) ** (-beta)
        weights = weights / weights.max() # 权重归一化到[0,1]，避免梯度爆炸

        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
            indices,
            weights.astype(np.float32)
        )

    def update_priorities(self, indices, td_errors):
        if torch.is_tensor(td_errors):
            td_errors = td_errors.detach().cpu().numpy()

        td_errors = np.asarray(td_errors).reshape(-1)
        # 优先级更新公式 p = |TD_error| + eps
        self.priorities[indices] = np.abs(td_errors) + self.eps

    def clean(self):
        self.buffer = []
        self.position = 0
        self.priorities = np.zeros((self.buffer_size,), dtype=np.float32)

# DQN算法
class AgentDQN(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """
        super(AgentDQN, self).__init__(env)
        self.args = args # 超参数配置
        self.env = env # CartPole环境

        # 自动选择GPU/CPU设备
        self.device = torch.device(
            "cuda" if args.use_cuda and torch.cuda.is_available() else "cpu"
        )

        self.state_dim = env.observation_space.shape[0] # 状态维度
        self.action_dim = env.action_space.n # 动作维度

        # 创建两个网络
        #   Q网络
        self.policy_net = QNetwork(
            self.state_dim,
            args.hidden_size,
            self.action_dim
        ).to(self.device)
        # 目标网络
        self.target_net = QNetwork(
            self.state_dim,
            args.hidden_size,
            self.action_dim
        ).to(self.device)

        # 初始同步两个网络参数
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval() # 目标网络关闭梯度计算，只做预测

        # 优化器
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=args.lr)
        # 损失函数：Huber Loss，PER需要逐样本loss来乘IS权重
        self.loss_fn = nn.SmoothL1Loss(reduction="none")

        self.replay_type = args.replay_type
        self.per_beta_start = args.per_beta_start
        self.per_beta_frames = args.per_beta_frames

        if args.replay_type == "per":
            self.replay_buffer = PrioritizedReplayBuffer(
                args.buffer_size,
                alpha=args.per_alpha,
                eps=args.per_eps
            )
        else:
            self.replay_buffer = ReplayBuffer(args.buffer_size)

        # 超参数赋值
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.grad_norm_clip = args.grad_norm_clip
        self.warmup_size = args.warmup_size
        self.target_update_freq = args.target_update_freq
        self.solved_reward = args.solved_reward
        self.solved_window = args.solved_window

        self.epsilon = args.epsilon_start
        self.epsilon_start = args.epsilon_start
        self.epsilon_end = args.epsilon_end
        self.epsilon_decay = args.epsilon_decay

        self.total_steps = 0
        self.episode_count = 0

        # Tensorboard日志文件夹
        self.log_dir = Path("runs") / "dqn_cartpole_basic"
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(str(self.log_dir))

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
    
    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        ##################
        # YOUR CODE HERE #
        ##################
        pass

    def train(self):
        """
        DQN参数更新逻辑
        """
        # 检查经验池：经验少不训练
        if len(self.replay_buffer) < self.warmup_size:
            return None

        if self.replay_type == "per":
            beta = min(
                1.0,
                self.per_beta_start + self.total_steps * (1.0 - self.per_beta_start) / self.per_beta_frames
            )
            # PER sampling returns indices for priority update and IS weights for loss correction.
            states, actions, rewards, next_states, dones, indices, weights = self.replay_buffer.sample(
                self.batch_size,
                beta=beta
            )
        else:
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(
                self.batch_size
            )

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        if self.replay_type == "per":
            weights = torch.FloatTensor(weights).unsqueeze(1).to(self.device)

        # 计算当前Q：policy_net根据真实动作取出对应Q(s,a)
        current_q = self.policy_net(states).gather(1, actions)

        # 计算目标Q：r+γmaxQtarget​(s′)
        with torch.no_grad(): # 目标网络不计算梯度，节省显存
            # Double DQN: 策略网络选择动作，目标网络评估价值
            next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, next_actions)
            target_q = rewards + self.gamma * next_q * (1 - dones)

        # 计算Huber损失
        elementwise_loss = self.loss_fn(current_q, target_q)
        if self.replay_type == "per":
            loss = (weights * elementwise_loss).mean()
        else:
            loss = elementwise_loss.mean()
        td_errors = target_q - current_q

        self.optimizer.zero_grad()
        loss.backward() # 反向传播
        # 梯度裁剪：防止梯度爆炸
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.grad_norm_clip) 
        self.optimizer.step() # 更新

        if self.replay_type == "per":
            # Update priorities after learning, using latest TD errors without gradients.
            self.replay_buffer.update_priorities(indices, td_errors)

        if self.total_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent
        Input:observation
        Return:action
        """
        if test:
            epsilon = 0.0
        else:
            epsilon = self.epsilon

        if random.random() < epsilon:
            return self.env.action_space.sample() # 探索

        state = torch.FloatTensor(observation).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_values = self.policy_net(state)
            action = q_values.argmax(dim=1).item()

        return action
    
    # 环境兼容工具
    def _reset_env(self):
        result = self.env.reset()

        if isinstance(result, tuple):
            state, _ = result
        else:
            state = result

        return state

    def _step_env(self, action):
        result = self.env.step(action)

        if len(result) == 5:
            next_state, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            next_state, reward, done, info = result

        return next_state, reward, done, info

    # reset->选动作->step->存经验->训练->更新目标网络->下一步
    def _save_training_metrics(self, metrics):
        csv_path = self.log_dir / "training_metrics.csv"
        with open(str(csv_path), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "episode",
                "frame",
                "reward",
                "avg20_reward",
                "loss",
                "epsilon",
                "episode_steps",
                "episode_time_sec",
                "steps_per_sec",
            ])

            for idx, reward in enumerate(metrics["rewards"]):
                writer.writerow([
                    idx,
                    metrics["frames"][idx],
                    reward,
                    metrics["avg20_rewards"][idx],
                    metrics["losses"][idx],
                    metrics["epsilons"][idx],
                    metrics["episode_steps"][idx],
                    metrics["episode_times"][idx],
                    metrics["episode_fps"][idx],
                ])

        return csv_path

    def _save_training_plots(self, metrics):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("Matplotlib is not available, skip saving training curves.")
            return None

        if len(metrics["rewards"]) == 0:
            return None

        episodes = np.arange(len(metrics["rewards"]))
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        axes[0, 0].plot(episodes, metrics["rewards"], label="Episode Reward", alpha=0.65)
        axes[0, 0].plot(episodes, metrics["avg20_rewards"], label="Avg{} Reward".format(self.solved_window), linewidth=2)
        axes[0, 0].axhline(self.solved_reward, color="r", linestyle="--", linewidth=1, label="Solved Threshold")
        axes[0, 0].set_title("Reward")
        axes[0, 0].set_xlabel("Episode")
        axes[0, 0].set_ylabel("Reward")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(episodes, metrics["losses"], color="tab:orange")
        axes[0, 1].set_title("Training Loss")
        axes[0, 1].set_xlabel("Episode")
        axes[0, 1].set_ylabel("Mean Loss")
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].plot(episodes, metrics["epsilons"], color="tab:green")
        axes[1, 0].set_title("Exploration Rate")
        axes[1, 0].set_xlabel("Episode")
        axes[1, 0].set_ylabel("Epsilon")
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].plot(episodes, metrics["episode_times"], label="Episode Time(s)")
        axes[1, 1].plot(episodes, metrics["episode_fps"], label="Steps/s")
        axes[1, 1].set_title("Runtime Efficiency")
        axes[1, 1].set_xlabel("Episode")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        fig.tight_layout()
        plot_path = self.log_dir / "training_analysis.png"
        fig.savefig(str(plot_path), dpi=150)
        plt.close(fig)

        return plot_path

    def _print_training_summary(self, metrics, elapsed_time, solved_episode, solved_frame):
        rewards = metrics["rewards"]
        if len(rewards) == 0:
            print("No completed episode, skip training summary.")
            return

        best_reward = max(rewards)
        final_avg20 = metrics["avg20_rewards"][-1]
        mean_reward = float(np.mean(rewards))
        mean_loss = float(np.mean(metrics["losses"]))
        mean_episode_time = float(np.mean(metrics["episode_times"]))
        mean_steps_per_sec = float(np.mean(metrics["episode_fps"]))
        total_frames = metrics["frames"][-1]

        print("\n========== DQN Training Summary ==========")
        print("Episodes: {} | Frames: {} | Total time: {:.2f}s".format(
            len(rewards), total_frames, elapsed_time
        ))
        print("Best reward: {:.1f} | Mean reward: {:.1f} | Final Avg{}: {:.1f}".format(
            best_reward, mean_reward, self.solved_window, final_avg20
        ))
        print("Mean loss: {:.4f} | Mean episode time: {:.3f}s | Mean speed: {:.1f} steps/s".format(
            mean_loss, mean_episode_time, mean_steps_per_sec
        ))

        if solved_episode is not None:
            print("Solved episode: {} | Solved frame: {}".format(solved_episode, solved_frame))
        else:
            print("Solved episode: Not reached | Solved threshold: Avg{} >= {:.1f}".format(
                self.solved_window,
                self.solved_reward
            ))

    def run(self):
        """
        Implement the interaction between agent and environment here
        """
        reward_history = []
        metrics = {
            "rewards": [],
            "avg20_rewards": [],
            "losses": [],
            "epsilons": [],
            "frames": [],
            "episode_steps": [],
            "episode_times": [],
            "episode_fps": [],
        }

        state = self._reset_env()
        episode_reward = 0
        episode_loss = []
        episode_steps = 0
        start_time = time.time()
        episode_start_time = start_time
        solved_episode = None
        solved_frame = None

        # 总训练最大帧数循环
        for frame_idx in range(1, self.args.n_frames + 1):
            self.total_steps = frame_idx
            # 选动作、和环境交互
            action = self.make_action(state, test=False)
            next_state, reward, done, _ = self._step_env(action)

            self.replay_buffer.push(state, action, reward, next_state, done)

            loss = self.train()
            if loss is not None:
                episode_loss.append(loss)

            state = next_state
            episode_reward += reward
            episode_steps += 1

            # 当前对局结束(done=True)，统计日志、衰减epsilon
            if done:
                episode_time = time.time() - episode_start_time
                episode_fps = episode_steps / max(episode_time, 1e-8)
                reward_history.append(episode_reward)
                avg_reward = np.mean(reward_history[-self.solved_window:])

                mean_loss = np.mean(episode_loss) if episode_loss else 0.0
                metrics["rewards"].append(float(episode_reward))
                metrics["avg20_rewards"].append(float(avg_reward))
                metrics["losses"].append(float(mean_loss))
                metrics["epsilons"].append(float(self.epsilon))
                metrics["frames"].append(frame_idx)
                metrics["episode_steps"].append(episode_steps)
                metrics["episode_times"].append(float(episode_time))
                metrics["episode_fps"].append(float(episode_fps))

                self.writer.add_scalar("Reward/Episode", episode_reward, self.episode_count)
                self.writer.add_scalar("Reward/Avg{}".format(self.solved_window), avg_reward, self.episode_count)
                self.writer.add_scalar("Loss", mean_loss, self.episode_count)
                self.writer.add_scalar("Epsilon", self.epsilon, self.episode_count)
                self.writer.add_scalar("Runtime/EpisodeTime", episode_time, self.episode_count)
                self.writer.add_scalar("Runtime/StepsPerSecond", episode_fps, self.episode_count)

                print(
                    "Episode: {:4d} | Frame: {:6d} | Reward: {:6.1f} | Avg{}: {:6.1f} | Epsilon: {:.3f} | Loss: {:.4f} | Time: {:.2f}s | FPS: {:.1f}".format(
                        self.episode_count,
                        frame_idx,
                        episode_reward,
                        self.solved_window,
                        avg_reward,
                        self.epsilon,
                        mean_loss,
                        episode_time,
                        episode_fps
                    )
                )

                # epsilon衰减，逐步降低探索概率
                self.epsilon = max(
                    self.epsilon_end,
                    self.epsilon * self.epsilon_decay
                )

                state = self._reset_env()
                episode_reward = 0
                episode_loss = []
                episode_steps = 0
                episode_start_time = time.time()
                self.episode_count += 1

                if avg_reward >= self.solved_reward and len(reward_history) >= self.solved_window:
                    solved_episode = self.episode_count - 1
                    solved_frame = frame_idx
                    print("Solved! Avg{} reward reached {:.1f}".format(
                        self.solved_window,
                        avg_reward
                    ))
                    break

        elapsed_time = time.time() - start_time
        csv_path = self._save_training_metrics(metrics)
        plot_path = self._save_training_plots(metrics)
        self._print_training_summary(metrics, elapsed_time, solved_episode, solved_frame)

        print("Metrics CSV saved to: {}".format(csv_path))
        if plot_path is not None:
            print("Training analysis plot saved to: {}".format(plot_path))

        self.writer.close()
