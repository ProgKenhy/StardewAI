import time
from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np
import pyautogui


@dataclass
class FishingConfig:
    template_path: str = "template.png"  # Путь к шаблону восклицательного знака
    center_radius: int = 50
    y_offset: int = 145
    match_threshold: float = 0.8  # Порог совпадения (0–1)


def show_debug_window(region: np.ndarray, result: np.ndarray, max_loc: Tuple[int, int], max_val: float) -> None:
    """Окно с визуализацией процесса"""
    debug_img = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)

    # Рисуем прямоугольник вокруг найденного совпадения, если порог превышен
    if max_val > 0.8:
        cv2.rectangle(debug_img, max_loc,
                      (max_loc[0] + 10, max_loc[1] + 10),  # Размер прямоугольника можно скорректировать
                      (0, 255, 0), 2)

    # Текст с уровнем совпадения
    cv2.putText(debug_img, f"{max_val:.2f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 1)

    cv2.imshow('Fishing Debug', debug_img)
    cv2.waitKey(1)


def hooking_fish(config: FishingConfig) -> bool:
    """Подсекаем рыбу при появлении восклицательного знака"""
    center_x, center_y = pyautogui.size()[0] // 2, pyautogui.size()[1] // 2

    # Зона поиска выше головы персонажа
    region = (
        center_x - config.center_radius,
        center_y - config.center_radius - config.y_offset,
        config.center_radius * 2,
        config.center_radius * 2
    )

    # Загрузка шаблона
    template = cv2.imread(config.template_path, cv2.IMREAD_GRAYSCALE)
    template_h, template_w = template.shape[:2]

    cv2.namedWindow('Fishing Debug', cv2.WINDOW_NORMAL)

    try:
        while True:
            screenshot = pyautogui.screenshot(region=region)
            img = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Сопоставление шаблона
            result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

            # Визуализация отладки
            show_debug_window(img, result, max_loc, max_val)

            if max_val >= config.match_threshold:
                print("! обнаружен - подсекаем!")
                pyautogui.mouseDown()
                time.sleep(0.1)
                pyautogui.mouseUp()
                return True

            time.sleep(0.05)
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    time.sleep(3)
    hooking_fish(FishingConfig())
