import os
os.environ["MUJOCO_GL"] = "egl"

import numpy as np
import gymnasium as gym
import imageio
from IPython.display import HTML, display
import base64

class EvaluatorAgent:
    def __init__(self, dataset_dir):
        if os.path.isdir(dataset_dir):
            self.W1_a = np.load(os.path.join(dataset_dir, 'W1_a.npy'))
            self.b1_a = np.load(os.path.join(dataset_dir, 'b1_a.npy'))
            self.W2_a = np.load(os.path.join(dataset_dir, 'W2_a.npy'))
            self.b2_a = np.load(os.path.join(dataset_dir, 'b2_a.npy'))
            self.W3_a = np.load(os.path.join(dataset_dir, 'W3_a.npy'))
            self.b3_a = np.load(os.path.join(dataset_dir, 'b3_a.npy'))
            self.rms_mean = np.load(os.path.join(dataset_dir, 'rms_mean.npy'))
            self.rms_var = np.load(os.path.join(dataset_dir, 'rms_var.npy'))
        else:
            data = np.load(dataset_dir)
            self.W1_a = data['W1_a']
            self.b1_a = data['b1_a']
            self.W2_a = data['W2_a']
            self.b2_a = data['b2_a']
            self.W3_a = data['W3_a']
            self.b3_a = data['b3_a']
            self.rms_mean = data['rms_mean']
            self.rms_var = data['rms_var']

    def normalize(self, s):
        return (s - self.rms_mean) / np.sqrt(self.rms_var + 1e-8)

    def get_action(self, s):
        s_norm = self.normalize(s)
        z1 = s_norm @ self.W1_a + self.b1_a
        a1 = np.tanh(z1)
        z2 = a1 @ self.W2_a + self.b2_a
        a2 = np.tanh(z2)
        mean = a2 @ self.W3_a + self.b3_a
        return np.clip(mean[0], -1.0, 1.0)

if __name__ == "__main__":
    base_path = "/kaggle/input/datasets/aurorcys/slotsss"
    checkpoints = ["slot_0", "slot_1", "slot_2", "slot_3", "slot_4"]
    
    env = gym.make("HalfCheetah-v4", render_mode="rgb_array")
    num_episodes = 5

    for checkpoint in checkpoints:
        path = os.path.join(base_path, checkpoint)
        if not os.path.exists(path) and os.path.exists(path + ".npz"):
            path = path + ".npz"
            
        print(f"\nEvaluating: {checkpoint}")
        
        try:
            agent = EvaluatorAgent(path)
        except Exception as e:
            print(f"Failed to load {checkpoint}: {e}")
            continue

        best_score = -float('inf')
        best_frames = []
        
        for ep in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0
            done = False
            frames = []
            
            while not done:
                frame = env.render()
                if frame is not None:
                    frames.append(frame)
                        
                action = agent.get_action(state.reshape(1, -1))
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_reward += reward
                
            print(f"  Episode {ep + 1}: {episode_reward:.2f}")
            
            if episode_reward > best_score:
                best_score = episode_reward
                best_frames = frames
                print(f"  -> New best: {best_score:.2f}")

        video_path = f"video_{checkpoint}.mp4"
        imageio.mimsave(video_path, best_frames, fps=20)
        
        with open(video_path, 'rb') as f:
            video_bytes = f.read()
        b64 = base64.b64encode(video_bytes).decode('utf-8')
        
        display(HTML(f'''
        <div style="margin-bottom:30px;padding:15px;border-radius:10px;background:#f8f9fa;border:1px solid #e9ecef;">
            <h4 style="margin-top:0;">Model: {checkpoint}</h4>
            <p style="font-weight:bold;color:#2b8a3e;">Best Score: {best_score:.2f}</p>
            <video width="640" height="360" controls autoplay loop style="border-radius:8px;">
                <source src="data:video/mp4;base64,{b64}" type="video/mp4">
            </video>
        </div>
        '''))
        
        del best_frames

    env.close()
    print("\nDone.")