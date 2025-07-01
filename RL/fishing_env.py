import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import pyautogui
from ultralytics import YOLO
import time


class FishingEnv(gym.Env):
    def __init__(self, capture_width=200, capture_height=800):
        super(FishingEnv, self).__init__()

        # Инициализация YOLO
        self.detection_model = YOLO('../models/fish_detector_v4.pt')

        screen_width, screen_height = pyautogui.size()
        self.capture_region = (
            (screen_width - capture_width - 200) // 2,
            (screen_height - capture_height) // 2,
            capture_width,
            capture_height
        )

        # Определение пространства действий и состояний
        self.action_space = spaces.Discrete(2)  # 0 - отпустить, 1 - нажать
        self.observation_space = spaces.Box(
            low=np.array([0, 0, -100, 0, 0, -1]),
            high=np.array([800, 800, 100, 800, 800, 1]),
            dtype=np.float32
        )

        # Состояние
        self.prev_fish_pos = None
        self.prev_time = time.time()
        self.current_action = 0
        self.episode_step = 0
        self.max_steps = 200

    def _get_observation(self):
        # Захват экрана
        screenshot = pyautogui.screenshot(region=self.capture_region)
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Детекция объектов
        results = self.detection_model(frame, conf=0.7, verbose=False)

        # Получение позиций
        fish_pos, zone_pos = None, None
        for result in results:
            for box in result.boxes:
                y_center = float((box.xyxy[0][1] + box.xyxy[0][3]) / 2)
                if int(box.cls) == 0:  # fish
                    fish_pos = y_center
                elif int(box.cls) == 1:  # zone
                    zone_pos = y_center

        # Расчет скорости
        current_time = time.time()
        fish_speed = 0.0
        if self.prev_fish_pos is not None and fish_pos is not None:
            time_diff = current_time - self.prev_time
            if time_diff > 0:
                fish_speed = (fish_pos - self.prev_fish_pos) / time_diff

        # Обновление состояния
        if fish_pos is not None:
            self.prev_fish_pos = fish_pos
        self.prev_time = current_time

        # Возвращаем наблюдение
        if fish_pos is None or zone_pos is None:
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        return np.array([
            fish_pos,
            zone_pos,
            fish_speed,
            fish_pos - zone_pos,
            abs(fish_pos - zone_pos),
            np.sign(fish_speed) if fish_speed != 0 else 0
        ], dtype=np.float32)

    def _calculate_reward(self, observation):
        """Система вознаграждений - ключевая часть!"""
        fish_pos, zone_pos, speed, pos_diff, distance, speed_dir = observation

        # Базовые награды
        reward = 0

        # Награда за нахождение в зоне
        if distance < 15:
            reward += 1.0 - distance / 15

        # Штраф за выход из зоны
        if distance > 30:
            reward -= 0.5

        # Награда за правильные действия
        if self.current_action == 1 and distance < 10:
            reward += 0.2
        elif self.current_action == 0 and distance > 20:
            reward += 0.1

        # # Штраф за слишком частые переключения
        # if self.episode_step > 10 and np.abs(speed) > 20:
        #     reward -= 0.3

        return reward

    def step(self, action):
        self.current_action = action
        self.episode_step += 1

        # Применяем действие
        if action == 1 and not self.is_fishing:
            pyautogui.mouseDown()
            self.is_fishing = True
        elif action == 0 and self.is_fishing:
            pyautogui.mouseUp()
            self.is_fishing = False

        # Получаем новое состояние
        observation = self._get_observation()

        # Рассчитываем награду
        reward = self._calculate_reward(observation)

        # Условия завершения эпизода
        terminated = False  # True если достигнут терминальное состояние
        truncated = self.episode_step >= self.max_steps  # True если превышено максимальное число шагов

        # Дополнительная информация
        info = {
            "distance": abs(observation[0] - observation[1]),
            "steps": self.episode_step
        }

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        # Обязательно принимаем и обрабатываем seed и options
        super().reset(seed=seed)

        self.episode_step = 0
        self.is_fishing = False
        pyautogui.mouseUp()

        observation = self._get_observation()
        info = {}  # Дополнительная информация (может быть пустой)

        return observation, info

    def render(self, mode='human'):
        pass  # Отрисовка через основной бот

    def close(self):
        pyautogui.mouseUp()
