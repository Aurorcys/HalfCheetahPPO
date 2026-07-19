import numpy as np 

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