import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
import random
from collections import deque
import gymnasium as gym
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.base_env import ActionTuple
from imitation.data import types
from imitation.algorithms import bc
from stable_baselines3.common.policies import ActorCriticPolicy

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

class GraspingNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super(GraspingNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        return self.network(x)


class DemonstrationDataset(Dataset):
    def __init__(self, observations, actions):
        self.observations = observations
        self.actions = actions
        
    def __len__(self):
        return len(self.observations)
    
    def __getitem__(self, idx):
        return self.observations[idx], self.actions[idx]


class ExperienceBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
        
    def add(self, obs, action, reward, next_obs, done):
        self.buffer.append((obs, action, reward, next_obs, done))
        
    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        obs, actions, rewards, next_obs, dones = zip(*samples)
        return np.array(obs), np.array(actions), np.array(rewards), np.array(next_obs), np.array(dones)
    
    def __len__(self):
        return len(self.buffer)


class BehaviorCloningAgent:
    def __init__(self, env, input_dim, output_dim, hidden_dim=256, lr=1e-4):
        self.env = env
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.policy = GraspingNetwork(input_dim, output_dim, hidden_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        
    
        self.demos = ExperienceBuffer()
        
    def collect_demonstrations(self, num_episodes=10, expert_policy=None):
        """
        Collect expert demonstrations, either from human or existing policy
        """
        print("Collecting expert demonstrations...")
        
        for episode in range(num_episodes):
            obs, _ = self.env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                # Get expert action 
                if expert_policy is None:
                    # replace with demonstration capture using the ROS bridge
                    action = self._simulate_expert_action(obs)
                else:
                    action = expert_policy(obs)
                    
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                # Store transition
                self.demos.add(obs, action, reward, next_obs, done)
                
                obs = next_obs
                episode_reward += reward
                
            print(f"Episode {episode+1}/{num_episodes}, Reward: {episode_reward}")
            
    def _simulate_expert_action(self, obs):
        """
        Simulate expert behavior with a heuristic policy
        Replace with actual human demonstration capture later
        """
      
        return np.random.normal(0, 0.1, size=self.output_dim)  
    
    def train(self, num_epochs=100, batch_size=64):
        """
        Train the policy network using behavior cloning on demonstrations
        """
        if len(self.demos) < batch_size:
            print("Not enough demonstrations to train")
            return
            
        print(f"Training on {len(self.demos)} demonstration transitions")
        
        obs_data, action_data = [], []
        for i in range(len(self.demos.buffer)):
            obs, action, _, _, _ = self.demos.buffer[i]
            obs_data.append(obs)
            action_data.append(action)
            
        obs_data = np.array(obs_data, dtype=np.float32)
        action_data = np.array(action_data, dtype=np.float32)
        
        dataset = DemonstrationDataset(
            torch.FloatTensor(obs_data).to(device),
            torch.FloatTensor(action_data).to(device)
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(num_epochs):
            epoch_loss = 0
            for obs_batch, action_batch in dataloader:
                self.optimizer.zero_grad()
                predicted_actions = self.policy(obs_batch)
                loss = self.loss_fn(predicted_actions, action_batch)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                
            avg_loss = epoch_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")
            
    def save_model(self, path="models/bc_grasping_policy.pth"):
        """Save the trained policy"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.policy.state_dict(), path)
        print(f"Model saved to {path}")
        
    def load_model(self, path="models/bc_grasping_policy.pth"):
        """Load a trained policy"""
        self.policy.load_state_dict(torch.load(path, map_location=device))
        print(f"Model loaded from {path}")
        
    def evaluate(self, num_episodes=5):
        """Evaluate the trained policy"""
        total_rewards = 0
        success_count = 0
        
        for episode in range(num_episodes):
            obs, _ = self.env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
                with torch.no_grad():
                    action = self.policy(obs_tensor).cpu().numpy()[0]
                
                # Take action in environment
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                obs = next_obs
                episode_reward += reward
                
            total_rewards += episode_reward
            if episode_reward > 0.5: 
                success_count += 1
                
            print(f"Episode {episode+1}: Reward = {episode_reward}")
            
        avg_reward = total_rewards / num_episodes
        success_rate = success_count / num_episodes
        print(f"Average Reward: {avg_reward}, Success Rate: {success_rate}")
        
        return avg_reward, success_rate


class ImitationBCAgent:
    """
    Alternative implementation using the imitation library's BC algorithm
    """
    def __init__(self, env, demonstrations=None):
        self.env = env
        self.demonstrations = demonstrations
        
    def train(self, n_epochs=50):
        """Train BC policy using the imitation library"""
        # Create BC trainer
        bc_trainer = bc.BC(
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            demonstrations=self.demonstrations,
            rng=np.random.default_rng(),
        )
        
        # Train BC policy
        bc_trainer.train(n_epochs=n_epochs)
        
        self.policy = bc_trainer.policy
        return self.policy
    
    def save_model(self, path="models/imitation_bc_policy"):
        """Save the trained policy"""
        os.makedirs(path, exist_ok=True)
        self.policy.save(path)
        print(f"Model saved to {path}")
        
    def load_model(self, path="models/imitation_bc_policy"):
        """Load a trained policy"""
        self.policy = ActorCriticPolicy.load(path)
        print(f"Model loaded from {path}")
    
    def evaluate(self, n_episodes=10):
        """Evaluate the trained policy"""
        rewards = []
        
        for _ in range(n_episodes):
            obs, _ = self.env.reset()
            done = False
            total_reward = 0
            
            while not done:
                action, _ = self.policy.predict(obs)
                obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                total_reward += reward
                
            rewards.append(total_reward)
            
        return np.mean(rewards), np.std(rewards)


class UnityRobotArmEnv:
    """
    Wrapper for the Unity ML-Agents environment
    Specifically adapted for the UR5e robotic arm setup
    """
    def __init__(self, unity_env, behavior_name):
        self.unity_env = unity_env
        self.behavior_name = behavior_name
        self.behavior_spec = self.unity_env.behavior_specs[self.behavior_name]
    
        obs_shape = self._get_obs_shape()
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32
        )
        
        if self.behavior_spec.action_spec.continuous_size > 0:
            action_dim = self.behavior_spec.action_spec.continuous_size
            self.action_space = gym.spaces.Box(
                low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32
            )
            self.is_continuous = True
        else:
            action_dim = self.behavior_spec.action_spec.discrete_branches[0]
            self.action_space = gym.spaces.Discrete(action_dim)
            self.is_continuous = False
            
    def _get_obs_shape(self):
        """Determine observation shape from behavior spec"""
        # Assume a single-agent setup for now (update later)
        return (self.behavior_spec.observation_specs[0].shape[0],)

    def reset(self):
        """Reset the environment"""
        self.unity_env.reset()
        decision_steps, _ = self.unity_env.get_steps(self.behavior_name)
        
        # Get observation for the first agent
        agent_id = decision_steps.agent_id[0]
        obs = self._process_observations(decision_steps, agent_id)
        
        return obs, {}  
    
    def step(self, action):
        """Take a step in the environment"""
        if self.is_continuous:
            action_tuple = ActionTuple(continuous=np.array([action]), discrete=None)
        else:
            action_tuple = ActionTuple(continuous=None, discrete=np.array([[action]]))

        self.unity_env.set_actions(self.behavior_name, action_tuple)
        self.unity_env.step()
        decision_steps, terminal_steps = self.unity_env.get_steps(self.behavior_name)

        # Assuming single-agent setup
        agent_id = list(decision_steps.agent_id)[0] if decision_steps.agent_id else list(terminal_steps.agent_id)[0]

        if agent_id in decision_steps:
            obs = self._process_observations(decision_steps, agent_id)
            reward = decision_steps[agent_id].reward
            done = False
            truncated = False
        else:
            obs = self._process_observations(terminal_steps, agent_id)
            reward = terminal_steps[agent_id].reward
            done = True
            truncated = False  

        return obs, reward, done, truncated, {}

    def _process_observations(self, steps, agent_id):
        obs = steps[agent_id].obs[0]
        return obs

    def close(self):
        self.unity_env.close()

def main():
    channel = EngineConfigurationChannel()
    
    channel.set_configuration_parameters(
        width=1280,
        height=720,
        quality_level=5,
        time_scale=1.0,
        target_frame_rate=-1,
        capture_frame_rate=60
    )
    
    env = UnityEnvironment(
        file_name=None,  
        seed=42,
        side_channels=[channel],
        no_graphics=False, 
        worker_id=0
    )
    
    env.reset()

    behavior_names = list(env.behavior_specs.keys())
    if len(behavior_names) != 1:
        print(f"Expected 1 behavior, found {len(behavior_names)}: {behavior_names}")
    
    behavior_name = behavior_names[0]
    print(f"Using behavior: {behavior_name}")
    
    robot_env = UnityRobotArmEnv(env, behavior_name)
    
    obs_dim = robot_env.observation_space.shape[0]
    if isinstance(robot_env.action_space, gym.spaces.Box):
        action_dim = robot_env.action_space.shape[0]
    else:
        action_dim = robot_env.action_space.n
    
    print(f"Observation dim: {obs_dim}, Action dim: {action_dim}")
    
    # Create the behavior cloning agent
    agent = BehaviorCloningAgent(
        env=robot_env,
        input_dim=obs_dim,
        output_dim=action_dim,
        hidden_dim=256,
        lr=1e-4
    )
    
    # Collect demonstrations (change later since using ROS bridge)
    agent.collect_demonstrations(num_episodes=20)
    agent.train(num_epochs=100, batch_size=64)
    agent.save_model()
    agent.evaluate(num_episodes=10)
    robot_env.close()
    

if __name__ == "__main__":
    main()