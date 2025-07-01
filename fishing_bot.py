import cv2
import numpy as np
import pyautogui
import time
from ultralytics import YOLO
import joblib
from collections import deque
import warnings

# Игнорируем предупреждения, которые мы уже обработали
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=DeprecationWarning)


class FishingBot:
    def __init__(self, capture_width=200, capture_height=800):
        # Инициализация моделей
        self.detection_model = YOLO('models/fish_detector_v4.pt')
        self.detection_model.fuse()  # Оптимизация для ускорения

        # Загрузка модели и нормализатора
        self.decision_model = joblib.load('perceptron/fishing_model_enhanced.pkl')
        self.scaler = joblib.load('perceptron/scaler_enhanced.pkl')

        # Настройка области захвата
        screen_width, screen_height = pyautogui.size()
        self.capture_region = (
            (screen_width - capture_width - 200) // 2,
            (screen_height - capture_height) // 2,
            capture_width,
            capture_height
        )

        # Состояние бота
        self.is_fishing = False
        self.last_action_time = time.time()
        self.prev_fish_pos = None
        self.prev_time = time.time()
        self.speed_history = deque(maxlen=5)
        self.action_history = []

        # Статистика производительности
        self.frame_count = 0
        self.total_processing_time = 0

        # Оптимизация: предварительное выделение памяти
        self.last_frame = None
        self.feature_names = ['fish_pos', 'zone_pos', 'fish_speed', 'pos_diff', 'distance', 'speed_dir']

        # Окно для отладки
        cv2.namedWindow('Fish Bot Control')

    def calculate_speed(self, current_fish_pos):
        """Вычисление скорости с защитой от деления на ноль"""
        current_time = time.time()
        time_diff = current_time - self.prev_time
        speed = 0.0

        if self.prev_fish_pos is not None and time_diff > 0.0001:
            speed = (current_fish_pos - self.prev_fish_pos) / time_diff
            self.speed_history.append(speed)

        self.prev_fish_pos = current_fish_pos
        self.prev_time = current_time

        return np.mean(self.speed_history) if self.speed_history else 0.0

    def process_frame(self):
        start_time = time.time()

        # Захват экрана с кэшированием
        if self.last_frame is None or time.time() - self.last_action_time > 0.1:
            screenshot = pyautogui.screenshot(region=self.capture_region)
            self.last_frame = np.array(screenshot)

        frame = cv2.cvtColor(self.last_frame, cv2.COLOR_RGB2BGR)

        # Детекция объектов с оптимизацией
        results = self.detection_model(frame, conf=0.7, verbose=False)

        # Получение позиций объектов
        fish_pos, zone_pos = None, None
        for result in results:
            for box in result.boxes:
                y_center = float((box.xyxy[0][1] + box.xyxy[0][3]) / 2)
                if int(box.cls) == 0:  # fish
                    fish_pos = y_center
                elif int(box.cls) == 1:  # zone
                    zone_pos = y_center

        # Принятие решения
        if fish_pos is not None and zone_pos is not None:
            fish_speed = self.calculate_speed(fish_pos)
            self.make_decision(fish_pos, zone_pos, fish_speed)

        # Визуализация (не чаще 30 FPS)
        if self.frame_count % 2 == 0:  # Пропускаем каждый второй кадр для отрисовки
            self.show_control_panel(frame, results, fish_pos, zone_pos)

        # Статистика производительности
        self.frame_count += 1
        self.total_processing_time += time.time() - start_time

    def make_decision(self, fish_pos, zone_pos, fish_speed):
        """Оптимизированное принятие решения"""
        current_time = time.time()
        if current_time - self.last_action_time < 0.1:  # Ограничение частоты
            return

        # Подготовка признаков
        pos_diff = fish_pos - zone_pos
        distance = abs(pos_diff)
        speed_dir = 1.0 if fish_speed >= 0 else -1.0  # Оптимизированная версия np.sign

        # Создание входного вектора с правильными именами признаков
        input_data = np.array([[fish_pos, zone_pos, fish_speed, pos_diff, distance, speed_dir]])

        # Нормализация и предсказание
        try:
            scaled_input = self.scaler.transform(input_data)
            action = self.decision_model.predict(scaled_input)[0]
        except Exception as e:
            print(f"Ошибка предсказания: {str(e)}")
            return

        # Выполнение действия
        if action == 1 and not self.is_fishing:
            pyautogui.mouseDown()
            self.is_fishing = True
            self.last_action_time = current_time
        elif action == 0 and self.is_fishing:
            pyautogui.mouseUp()
            self.is_fishing = False
            self.last_action_time = current_time

    def show_control_panel(self, frame, results, fish_pos, zone_pos):
        """Оптимизированная визуализация"""
        annotated_frame = frame.copy()

        # Рисуем детекции
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                color = (0, 255, 0) if int(box.cls) == 1 else (0, 0, 255)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                label = f"{self.detection_model.names[int(box.cls)]} {float(box.conf):.2f}"
                cv2.putText(annotated_frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Информационная панель
        if fish_pos is not None and zone_pos is not None:
            fps = self.frame_count / max(0.1, self.total_processing_time)
            status = "HOLDING" if self.is_fishing else "RELEASED"
            diff = fish_pos - zone_pos
            speed = self.calculate_speed(fish_pos) if self.prev_fish_pos else 0.0

            info_text = [
                f"FPS: {fps:.1f}",
                f"Fish: {fish_pos:.1f}",
                f"Zone: {zone_pos:.1f}",
                f"Diff: {diff:.1f}",
                f"Speed: {speed:.1f}",
                status
            ]

            cv2.putText(annotated_frame, " | ".join(info_text),
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow('Fish Bot Control', annotated_frame)
        cv2.waitKey(1)

    def run(self):
        try:
            print("Бот запущен. Нажмите ESC для выхода.")
            while True:
                self.process_frame()
                if cv2.waitKey(1) == 27:  # ESC для выхода
                    break
        finally:
            if self.is_fishing:
                pyautogui.mouseUp()
            cv2.destroyAllWindows()
            print(f"Бот остановлен. Средний FPS: {self.frame_count / max(0.1, self.total_processing_time):.1f}")


if __name__ == "__main__":
    bot = FishingBot()
    bot.run()
