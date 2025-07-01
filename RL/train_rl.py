from stable_baselines3 import PPO
from fishing_env import FishingEnv
import os

# Создание среды
from stable_baselines3 import PPO
from fishing_env import FishingEnv
import os

# Создаем среду с новым интерфейсом
env = FishingEnv()

# Проверяем reset
obs, info = env.reset()
print("Проверка reset:", obs.shape, info)

# Создаем модель
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=512,
    batch_size=64,
    n_epochs=10,
    device="cpu"  # Или "cuda" если есть GPU
)

# Обучение
model.learn(total_timesteps=50_000)

# Сохранение модели
os.makedirs("../rl_models", exist_ok=True)
model.save("../rl_models/fishing_ppo")
