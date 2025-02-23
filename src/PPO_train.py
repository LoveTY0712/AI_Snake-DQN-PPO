import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from Env4PPO import SnakeGymEnv

def linear_schedule(initial_value, final_value=0.0):
    if isinstance(initial_value, str):
        initial_value = float(initial_value)
        final_value = float(final_value)
        assert (initial_value > 0.0)

    def scheduler(progress):
        return final_value + progress * (initial_value - final_value)
    
    return scheduler

def PPO_train():
    env = SnakeGymEnv()
    lr_schedule = linear_schedule(1.5e-3, 2.5e-4)
    clip_range_schedule = linear_schedule(0.150, 0.025)

    model = PPO(  "MlpPolicy",
                env,
                device="cuda",
                verbose=1,
                n_steps=2048,
                batch_size=512,
                n_epochs=4,
                gamma=0.94,
                learning_rate=lr_schedule,
                clip_range=clip_range_schedule
                )



    model.learn(
        total_timesteps=int(2000000),
    )

if __name__ == "__main__":
    PPO_train()

