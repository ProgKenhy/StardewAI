import csv
import time
import cv2
import numpy as np
import pyautogui
from pynput import mouse
from ultralytics import YOLO
from collections import deque


class FishingDataCollector:
    def __init__(self, capture_width=200, capture_height=800):
        self.model = YOLO('../models/fish_detector_v4.pt')

        # Настройка области захвата
        screen_width, screen_height = pyautogui.size()
        self.capture_region = (
            (screen_width - capture_width - 200) // 2,
            (screen_height - capture_height) // 2,
            capture_width,
            capture_height
        )

        self.data_file = 'data/fishing_data_v30.csv'  # Обновленное имя файла
        self.init_data_file()

        # Для отслеживания скорости
        self.prev_fish_pos = None
        self.prev_time = time.time()
        self.speed_history = deque(maxlen=5)  # Скользящее окно для сглаживания скорости

        # Для отслеживания состояния мыши
        self.mouse_pressed = False
        self.listener = mouse.Listener(on_click=self.on_click)
        self.listener.start()

        # cv2.namedWindow('Fish Control')

    def init_data_file(self):
        """Инициализация файла данных с новыми полями"""
        with open(self.data_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'fish_pos',
                'zone_pos',
                'fish_speed',  # Новая колонка
                'action',
                'timestamp'  # Новая колонка
            ])

    def on_click(self, x, y, button, pressed):
        """Отслеживание кликов мыши"""
        if button == mouse.Button.left:
            self.mouse_pressed = pressed
        return True

    def calculate_speed(self, current_fish_pos):
        """Вычисляет сглаженную скорость рыбы (пикселей в секунду)"""
        current_time = time.time()
        speed = 0

        if self.prev_fish_pos is not None:
            time_diff = current_time - self.prev_time
            if time_diff > 0:
                speed = (current_fish_pos - self.prev_fish_pos) / time_diff
                self.speed_history.append(speed)

        self.prev_fish_pos = current_fish_pos
        self.prev_time = current_time

        # Возвращаем среднюю скорость по окну
        if len(self.speed_history) > 0:
            return sum(self.speed_history) / len(self.speed_history)
        return 0

    def save_data(self, fish_pos, zone_pos, fish_speed, mouse_state):
        """Сохранение данных в CSV с новыми полями"""
        with open(self.data_file, 'a', newline='') as f:
            writer = csv.writer(f)
            fish_val = float(fish_pos.item()) if hasattr(fish_pos, 'item') else float(fish_pos)
            zone_val = float(zone_pos.item()) if hasattr(zone_pos, 'item') else float(zone_pos)
            writer.writerow([
                fish_val,
                zone_val,
                float(fish_speed),
                int(mouse_state),
                time.time()  # Текущая метка времени
            ])

    def process_frame(self):
        screenshot = pyautogui.screenshot(region=self.capture_region)
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        results = self.model(frame, conf=0.6)

        fish_pos, zone_pos = None, None
        for result in results:
            for box in result.boxes:
                y_center = (box.xyxy[0][1] + box.xyxy[0][3]) / 2
                if box.cls == 0:  # fish
                    fish_pos = y_center
                elif box.cls == 1:  # zone
                    zone_pos = y_center

        # Если оба объекта найдены - сохраняем данные
        if fish_pos is not None and zone_pos is not None:
            fish_speed = self.calculate_speed(fish_pos)
            self.save_data(fish_pos, zone_pos, fish_speed, self.mouse_pressed)

        # self.show_control_panel(frame, results, fish_pos, zone_pos)

    def show_control_panel(self, frame, results, fish_pos, zone_pos):
        """Визуализация с отображением скорости"""
        annotated_frame = frame.copy()

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                color = (0, 255, 0) if box.cls == 1 else (0, 0, 255)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                label = f"{self.model.names[int(box.cls)]} {float(box.conf):.2f}"
                cv2.putText(annotated_frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        if fish_pos and zone_pos:
            diff = fish_pos - zone_pos
            speed = self.calculate_speed(fish_pos) if self.prev_fish_pos else 0
            status = "PRESSED" if self.mouse_pressed else "RELEASED"

            info_text = [
                f"Fish: {fish_pos:.1f}",
                f"Zone: {zone_pos:.1f}",
                f"Diff: {diff:.1f}",
                f"Speed: {speed:.1f} px/s",
                f"State: {status}"
            ]

            cv2.putText(annotated_frame, " | ".join(info_text),
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow('Fish Control', annotated_frame)
        cv2.waitKey(1)

    def run(self):
        try:
            print("Data collector started! Control mouse manually (left button). Press Ctrl+C to stop.")
            while True:
                self.process_frame()
                time.sleep(0.05)  # Ограничение частоты кадров
        except KeyboardInterrupt:
            print("\nStopping data collection...")
        finally:
            self.listener.stop()
            cv2.destroyAllWindows()
            print(f"Data saved to {self.data_file}")


if __name__ == "__main__":
    collector = FishingDataCollector()
    collector.run()
