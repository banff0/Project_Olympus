import torch
import CustomEnv
from stable_baselines3.common.env_checker import check_env

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from CustomCallback import CustomCallback

TRAIN_STEPS = 5000000
VAL_STEPS = 100000
lr = 0.001
gamma = 0.999
ent_coef = 0.009

ENV = CustomEnv.Hades()
# print(check_env(ENV))

# Set up the custom feature extractor since we are not using an image, but want to use a CNN to maintian spatial data
# policy_kwargs = dict(
#     features_extractor_class=CustomCNN,
#     features_extractor_kwargs=dict(features_dim=128),
# )

# use CUDA if it is avilable
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"using device: {DEVICE}")
cust_callback = CustomCallback()

model = PPO("CnnPolicy", ENV, verbose=1, device=DEVICE, ent_coef=ent_coef, gamma=gamma, learning_rate=lr, n_steps=10, batch_size=10) # use this if you are training from scratch
# # model = PPO("CnnPolicy", ENV, verbose=1, policy_kwargs=policy_kwargs, device=DEVICE, ent_coef=ent_coef, gamma=gamma, learning_rate=lr) # use this if you are training from scratch
# # model = PPO.load("./Saved Policies/BEST_1024_2048_policy", ENV, verbose=1, device=DEVICE, ent_coef=ent_coef, gamma=gamma, learning_rate=lr)
model.learn(total_timesteps=TRAIN_STEPS, callback=cust_callback)
# model.save("ARES")

# Show the end progress
room_type = input("Please Enter The Next Room Type>")
obs, _ = ENV.reset()
ENV.set_room_type(room_type)
ENV.render("human")
for i in range(VAL_STEPS):
    a = ENV.action_space.sample()
    # a, s = model.predict(obs)
    obs, r, done, trunc, info = ENV.step(a)
    if done or trunc:
        room_type = input("Please Enter The Next Room Type>")
        print("DONE")
        obs, _ = ENV.reset()
        ENV.set_room_type(room_type)
