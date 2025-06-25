import os
from typing import Tuple

import cv2
import numpy as np
import pyautogui
import time
from ultralytics import YOLO


class FishingBot:
    def __init__(self, capture_width=800, capture_height=600):
        self.model = YOLO('models/fish_detector_2.pt')
        self.is_fishing = False
        self.last_action = time.time()

        # Получаем размеры экрана
        screen_width, screen_height = pyautogui.size()

        # Вычисляем область захвата (центр экрана)
        self.capture_width = capture_width
        self.capture_height = capture_height
        self.start_x = (screen_width - capture_width - 200) // 2
        self.start_y = (screen_height - capture_height) // 2
        self.end_x = self.start_x + capture_width
        self.end_y = self.start_y + capture_height

        print(f"Область захвата: {self.start_x}x{self.start_y} - {self.end_x}x{self.end_y}")
        print(f"Размер области: {capture_width}x{capture_height}")

    def process_frame(self):
        region = (
            self.start_x,
            self.start_y,
            self.capture_width,
            self.capture_height)
        # 1. Делаем скриншот только центральной области
        screenshot = pyautogui.screenshot(region=region)
        img = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        self.show_debug_window(img)

        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        timestamp = time.strftime("%Y%m%d_%H%M%S")  # Метка времени
        screenshot_path = f"screenshots/screenshot_{timestamp}.png"
        os.makedirs("screenshots", exist_ok=True)  # Создаем папку, если её нет
        cv2.imwrite(screenshot_path, frame)
        print(f"Скриншот сохранен: {screenshot_path}")

        # 2. Детекция объектов
        results = self.model(frame, conf=0.6)

        # 3. Поиск рыбы и зоны
        fish = None
        zone = None

        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    center_y = (y1 + y2) // 2

                    if box.cls == 0:  # fish
                        fish = center_y
                    elif box.cls == 1:  # zone
                        zone = center_y

        # 4. Управление
        if fish and zone:
            self.control_fishing(fish, zone)

        # 5. Отладка (опционально)
        # debug_img = results[0].plot()
        # cv2.imshow('Fishing Bot Debug', debug_img)
        # cv2.waitKey(1)

    def control_fishing(self, fish_y, zone_y):
        """Управление ЛКМ на основе позиций"""
        print(f"Fish: {fish_y}, Zone: {zone_y}")

        if fish_y < zone_y - 15:  # Если рыба выше зоны
            if not self.is_fishing:
                pyautogui.mouseDown()
                self.is_fishing = True
                print("Рыба выше! Зажимаем ЛКМ")
        else:
            if self.is_fishing:
                pyautogui.mouseUp()
                self.is_fishing = False
                print("Рыба ниже! Отпускаем ЛКМ")

    def show_debug_window(self, region: np.ndarray) -> None:
        """Окно с визуализацией процесса"""
        debug_img = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)

        cv2.imshow('Fishing Debug', debug_img)
        cv2.waitKey(1)

    def run(self, show_fps=True):
        try:
            print("Бот запущен. Нажмите Ctrl+C для остановки")

            frame_count = 0
            start_time = time.time()

            while True:
                frame_start = time.time()

                self.process_frame()

                # Подсчет FPS
                if show_fps:
                    frame_count += 1
                    if frame_count % 30 == 0:  # Каждые 30 кадров
                        elapsed = time.time() - start_time
                        fps = frame_count / elapsed
                        print(f"FPS: {fps:.1f}")

                # Контроль частоты кадров
                frame_time = time.time() - frame_start
                target_frame_time = 1.0 / 1  # 20 FPS

                if frame_time < target_frame_time:
                    time.sleep(target_frame_time - frame_time)

        except KeyboardInterrupt:
            print("\nОстановка бота...")
        finally:
            if self.is_fishing:
                pyautogui.mouseUp()
            cv2.destroyAllWindows()


# Дополнительные функции для настройки
def test_capture_area():
    """Тестирование разных размеров области захвата"""
    sizes = [
        (400, 300),  # Маленькая область
        (600, 400),  # Средняя область
        (800, 600),  # Большая область
        (1000, 800),  # Очень большая область
    ]

    for width, height in sizes:
        print(f"\nТестирование области {width}x{height}")
        bot = FishingBot(width, height)

        # Делаем тестовый скриншот
        start_time = time.time()
        screenshot = pyautogui.screenshot(region=(
            bot.start_x, bot.start_y,
            bot.capture_width, bot.capture_height
        ))
        end_time = time.time()

        print(f"Время захвата: {(end_time - start_time) * 1000:.1f}ms")

        # Сохраняем для проверки
        screenshot.save(f'test_capture_{width}x{height}.png')


if __name__ == "__main__":
    # Создаем бота с областью захвата 800x600 пикселей
    bot = FishingBot(capture_width=200, capture_height=800)

    # Запустить бота
    bot.run()

    # Для тестирования разных размеров области:
    # test_capture_area()
