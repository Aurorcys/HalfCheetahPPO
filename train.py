import numpy as np
import gymnasium as gym
import os

class RunningMeanStd:
    def __init__(self, shape):
        self.mean = np.zeros(shape)
        self.var = np.ones(shape)
        self.count = 1e-4

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        self.mean += delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / tot_count
        self.var = M2 / tot_count
        self.count = tot_count

    def normalize(self, x):
        return (x - self.mean) / np.sqrt(self.var + 1e-8)


class PPOAgent:
    def __init__(self, obs_dim, act_dim, lr=3e-5):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.lr = lr

        self.W1_a = np.random.randn(obs_dim, 64) * np.sqrt(2.0 / obs_dim)
        self.b1_a = np.zeros((1, 64))
        self.W2_a = np.random.randn(64, 64) * np.sqrt(2.0 / 64)
        self.b2_a = np.zeros((1, 64))
        self.W3_a = np.random.randn(64, act_dim) * np.sqrt(2.0 / 64)
        self.b3_a = np.zeros((1, act_dim))

        self.log_std = np.zeros((1, act_dim))

        self.W1_c = np.random.randn(obs_dim, 64) * np.sqrt(2.0 / obs_dim)
        self.b1_c = np.zeros((1, 64))
        self.W2_c = np.random.randn(64, 64) * np.sqrt(2.0 / 64)
        self.b2_c = np.zeros((1, 64))
        self.W3_c = np.random.randn(64, 1) * np.sqrt(2.0 / 64)
        self.b3_c = np.zeros((1, 1))

    def forward_actor(self, s):
        self.z1_a = s @ self.W1_a + self.b1_a
        self.a1_a = np.tanh(self.z1_a)
        self.z2_a = self.a1_a @ self.W2_a + self.b2_a
        self.a2_a = np.tanh(self.z2_a)
        mu = self.a2_a @ self.W3_a + self.b3_a
        
        self.clamped_log_std = np.clip(self.log_std, -1.0, 0.5)
        std = np.exp(self.clamped_log_std)
        return mu, std

    def forward_critic(self, s):
        self.z1_c = s @ self.W1_c + self.b1_c
        self.a1_c = np.tanh(self.z1_c)
        self.z2_c = self.a1_c @ self.W2_c + self.b2_c
        self.a2_c = np.tanh(self.z2_c)
        v = self.a2_c @ self.W3_c + self.b3_c
        return v

    def get_action_and_val(self, s):
        mu, std = self.forward_actor(s)
        val = self.forward_critic(s)
        
        noise = np.random.randn(*mu.shape)
        action = mu + noise * std
        
        log_prob = -0.5 * (((action - mu) / std) ** 2 + 2 * self.clamped_log_std + np.log(2 * np.pi))
        return np.clip(action, -1.0, 1.0), np.sum(log_prob, axis=-1), val

    def train_step(self, states, actions, log_probs_old, returns, advantages, clip_eps=0.2):
        mu, std = self.forward_actor(states)
        v_pred = self.forward_critic(states)

        variance = std ** 2
        log_probs_new = np.sum(-0.5 * (((actions - mu) / std) ** 2 + 2 * self.clamped_log_std + np.log(2 * np.pi)), axis=-1, keepdims=True)
        log_probs_old = log_probs_old.reshape(-1, 1)
        
        ratio = np.exp(log_probs_new - log_probs_old)
        advantages = advantages.reshape(-1, 1)
        surr1 = ratio * advantages
        surr2 = np.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
        
        d_ratio = np.where((surr1 < surr2) | (ratio < 1.0 - clip_eps) | (ratio > 1.0 + clip_eps), 
                           advantages * ratio, 0)
        d_log_pi = -d_ratio / len(states)
        
        d_mu = d_log_pi * ((actions - mu) / variance)
        
        dW3_a = self.a2_a.T @ d_mu
        db3_a = np.sum(d_mu, axis=0, keepdims=True)
        da2_a = d_mu @ self.W3_a.T
        dz2_a = da2_a * (1.0 - self.a2_a ** 2)
        
        dW2_a = self.a1_a.T @ dz2_a
        db2_a = np.sum(dz2_a, axis=0, keepdims=True)
        da1_a = dz2_a @ self.W2_a.T
        dz1_a = da1_a * (1.0 - self.a1_a ** 2)
        
        dW1_a = states.T @ dz1_a
        db1_a = np.sum(dz1_a, axis=0, keepdims=True)

        d_log_std = d_log_pi * (((actions - mu) ** 2 / variance) - 1.0)
        d_log_std_grad = np.sum(d_log_std, axis=0, keepdims=True)

        d_v = 2.0 * (v_pred - returns) / len(states)
        
        dW3_c = self.a2_c.T @ d_v
        db3_c = np.sum(d_v, axis=0, keepdims=True)
        da2_c = d_v @ self.W3_c.T
        dz2_c = da2_c * (1.0 - self.a2_c ** 2)
        
        dW2_c = self.a1_c.T @ dz2_c
        db2_c = np.sum(dz2_c, axis=0, keepdims=True)
        da1_c = dz2_c @ self.W2_c.T
        dz1_c = da1_c * (1.0 - self.a1_c ** 2)
        
        dW1_c = states.T @ dz1_c
        db1_c = np.sum(dz1_c, axis=0, keepdims=True)

        for w, dw in zip([self.W1_a, self.W2_a, self.W3_a, self.W1_c, self.W2_c, self.W3_c],
                         [dW1_a, dW2_a, dW3_a, dW1_c, dW2_c, dW3_c]):
            np.clip(dw, -0.5, 0.5, out=dw)
            w -= self.lr * dw

        self.b1_a -= self.lr * np.clip(db1_a, -0.5, 0.5)
        self.b2_a -= self.lr * np.clip(db2_a, -0.5, 0.5)
        self.b3_a -= self.lr * np.clip(db3_a, -0.5, 0.5)
        self.b1_c -= self.lr * np.clip(db1_c, -0.5, 0.5)
        self.b2_c -= self.lr * np.clip(db2_c, -0.5, 0.5)
        self.b3_c -= self.lr * np.clip(db3_c, -0.5, 0.5)
        
        self.log_std -= self.lr * np.clip(d_log_std_grad, -0.5, 0.5)


def compute_gae(rewards, values, next_values, dones, gamma=0.99, lam=0.95):
    advantages = np.zeros_like(rewards)
    last_gae = 0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_values[t] * (1 - dones[t]) - values[t]
        advantages[t] = last_gae = delta + gamma * lam * (1 - dones[t]) * last_gae
    returns = advantages + values
    return advantages, returns


if __name__ == "__main__":
    env = gym.make("HalfCheetah-v4")
    
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    
    agent = PPOAgent(obs_dim, act_dim, lr=3e-5)
    rms = RunningMeanStd(shape=(obs_dim,))
    
    baseline_dir = "/kaggle/input/datasets/aurorcys/gooddd/"
    if os.path.exists(os.path.join(baseline_dir, "rms_var.npy")):
        print("Loading baseline model...")
        agent.W1_a = np.load(os.path.join(baseline_dir, 'W1_a.npy'))
        agent.b1_a = np.load(os.path.join(baseline_dir, 'b1_a.npy'))
        agent.W2_a = np.load(os.path.join(baseline_dir, 'W2_a.npy'))
        agent.b2_a = np.load(os.path.join(baseline_dir, 'b2_a.npy'))
        agent.W3_a = np.load(os.path.join(baseline_dir, 'W3_a.npy'))
        agent.b3_a = np.load(os.path.join(baseline_dir, 'b3_a.npy'))
        
        if os.path.exists(os.path.join(baseline_dir, 'W1_c.npy')):
            agent.W1_c = np.load(os.path.join(baseline_dir, 'W1_c.npy'))
            agent.b1_c = np.load(os.path.join(baseline_dir, 'b1_c.npy'))
            agent.W2_c = np.load(os.path.join(baseline_dir, 'W2_c.npy'))
            agent.b2_c = np.load(os.path.join(baseline_dir, 'b2_c.npy'))
            agent.W3_c = np.load(os.path.join(baseline_dir, 'W3_c.npy'))
            agent.b3_c = np.load(os.path.join(baseline_dir, 'b3_c.npy'))
            
        if os.path.exists(os.path.join(baseline_dir, 'log_std.npy')):
            agent.log_std = np.load(os.path.join(baseline_dir, 'log_std.npy'))
            
        rms.mean = np.load(os.path.join(baseline_dir, 'rms_mean.npy'))
        rms.var = np.load(os.path.join(baseline_dir, 'rms_var.npy'))
        rms.count = 1e5
        print("Baseline loaded.")
    else:
        print("No baseline found, starting fresh.")

    total_timesteps = 500_000
    rollout_len = 2048
    epochs = 10
    batch_size = 64
    
    state, info = env.reset()
    timestep = 0
    iteration = 0
    top_checkpoints = []
    
    while timestep < total_timesteps:
        states, actions, rewards, dones, values, log_probs = [], [], [], [], [], []
        
        for _ in range(rollout_len):
            rms.update(state.reshape(1, -1))
            norm_state = rms.normalize(state)
            
            action, log_prob, val = agent.get_action_and_val(norm_state.reshape(1, -1))
            
            next_state, reward, terminated, truncated, info = env.step(action[0])
            done = terminated or truncated
            
            torso_pitch = state[1]
            if abs(torso_pitch) > 0.5:
                reward -= 2.0 * abs(torso_pitch)
            
            states.append(norm_state)
            actions.append(action[0])
            rewards.append(reward)
            dones.append(done)
            values.append(val[0, 0])
            log_probs.append(log_prob[0])
            
            state = next_state
            timestep += 1
            
            if done:
                state, info = env.reset()
        
        _, _, next_val = agent.get_action_and_val(rms.normalize(state).reshape(1, -1))
        next_values = np.append(values[1:], next_val[0, 0])
        
        advantages, returns = compute_gae(
            np.array(rewards), np.array(values), next_values, np.array(dones)
        )
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        
        obs_batch = np.array(states)
        act_batch = np.array(actions)
        log_probs_batch = np.array(log_probs)
        returns_batch = np.array(returns)
        advantages_batch = np.array(advantages)
        
        dataset_size = len(obs_batch)
        for epoch in range(epochs):
            indices = np.arange(dataset_size)
            np.random.shuffle(indices)
            
            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                batch_idx = indices[start:end]
                
                agent.train_step(
                    states=obs_batch[batch_idx],
                    actions=act_batch[batch_idx],
                    log_probs_old=log_probs_batch[batch_idx],
                    returns=returns_batch[batch_idx].reshape(-1, 1),
                    advantages=advantages_batch[batch_idx]
                )
        
        iteration += 1
        mean_reward = np.sum(rewards) / (np.sum(dones) if np.sum(dones) > 0 else 1)
        print(f"Iter {iteration:3d} | Steps {timestep:7d} | Reward {mean_reward:7.2f} | Std {np.exp(agent.clamped_log_std[0, 0]):.4f}")
        
        if iteration % 20 == 0:
            if len(top_checkpoints) < 5 or mean_reward > top_checkpoints[0][0]:
                if len(top_checkpoints) < 5:
                    slot = len(top_checkpoints)
                else:
                    _, worst_file = top_checkpoints.pop(0)
                    slot = int(worst_file.split('_')[1].split('.')[0])
                
                filename = f"slot_{slot}.npz"
                np.savez(filename,
                         W1_a=agent.W1_a, b1_a=agent.b1_a, W2_a=agent.W2_a, b2_a=agent.b2_a,
                         W3_a=agent.W3_a, b3_a=agent.b3_a,
                         W1_c=agent.W1_c, b1_c=agent.b1_c, W2_c=agent.W2_c, b2_c=agent.b2_c,
                         W3_c=agent.W3_c, b3_c=agent.b3_c,
                         log_std=agent.log_std, rms_mean=rms.mean, rms_var=rms.var)
                
                top_checkpoints.append((mean_reward, filename))
                top_checkpoints.sort(key=lambda x: x[0])
                print(f"  -> Saved {filename} | Scores: {[round(x[0],1) for x in top_checkpoints]}")

    env.close()