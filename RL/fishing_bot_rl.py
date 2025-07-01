from fishing_env import FishingEnv
from stable_baselines3 import PPO


class RLFishingBot:
    def __init__(self):
        self.env = FishingEnv()
        self.model = PPO.load("rl_models/fishing_ppo")
        self.observation = self.env.reset()

    def run(self):
        try:
            print("RL Бот запущен. Нажмите Ctrl+C для остановки.")
            while True:
                action, _ = self.model.predict(self.observation, deterministic=True)
                self.observation, _, _, _ = self.env.step(action)

                # Визуализация (примерная)
                self._render()

        except KeyboardInterrupt:
            self.env.close()
            print("Бот остановлен.")

    def _render(self):
        # Простая визуализация состояния
        fish_pos, zone_pos, speed, pos_diff, distance, _ = self.observation
        print(
            f"Fish: {fish_pos:.1f} | Zone: {zone_pos:.1f} | Dist: {distance:.1f} | Action: {'HOLD' if self.env.is_fishing else 'RELEASE'}")


if __name__ == "__main__":
    bot = RLFishingBot()
    bot.run()
