# chimera/perception.py (Final, syntaktisch korrekt)

import torch
import numpy as np
from transformers import ViTImageProcessor, ViTModel
from PIL import Image
import pyscreenshot as ImageGrab
from pynput import mouse, keyboard
import logging
from threading import Thread, Lock

class PerceptionCore:
    _KEYS_TO_MAP = [
        keyboard.Key.esc, keyboard.Key.f1, keyboard.Key.f2, keyboard.Key.f3, keyboard.Key.f4,
        keyboard.Key.f5, keyboard.Key.f6, keyboard.Key.f7, keyboard.Key.f8, keyboard.Key.f9,
        keyboard.Key.f10, keyboard.Key.f11, keyboard.Key.f12, '`', '1', '2', '3', '4', '5', '6', '7',
        '8', '9', '0', '-', '=', keyboard.Key.backspace, keyboard.Key.tab, 'q', 'w', 'e',
        'r', 't', 'y', 'u', 'i', 'o', 'p', '[', ']', '\\', keyboard.Key.caps_lock, 'a',
        's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', ';', "'", keyboard.Key.enter,
        keyboard.Key.shift, 'z', 'x', 'c', 'v', 'b', 'n', 'm', ',', '.', '/',
        keyboard.Key.shift_r, keyboard.Key.ctrl_l, keyboard.Key.alt_l, keyboard.Key.space,
        keyboard.Key.ctrl_r, keyboard.Key.up, keyboard.Key.left, keyboard.Key.down, keyboard.Key.right
    ]
    KEY_VECTOR_SIZE = len(_KEYS_TO_MAP) + 1; MOUSE_VECTOR_SIZE = 6; VISION_FEATURE_SIZE = 768; NUM_VISION_VECTORS = 17

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_name = "google/vit-base-patch16-224-in21k"
        self.image_processor = ViTImageProcessor.from_pretrained(model_name)
        self.vision_model = ViTModel.from_pretrained(model_name).to(self.device)
        try:
            bbox = ImageGrab.grab().getbbox(); self.screen_width = bbox[2]; self.screen_height = bbox[3]
        except Exception:
            self.screen_width = 1920; self.screen_height = 1080
        self.lock = Lock()
        self.mouse_pos = (0, 0); self.mouse_buttons_state = {mouse.Button.left: 0, mouse.Button.right: 0, mouse.Button.middle: 0}
        self.scroll_delta_y = 0.0; self.last_key_pressed = None
        self.mouse_listener = mouse.Listener(on_move=self._on_move, on_click=self._on_click, on_scroll=self._on_scroll)
        self.keyboard_listener = keyboard.Listener(on_press=self._on_press)
        Thread(target=self.mouse_listener.run, daemon=True).start()
        Thread(target=self.keyboard_listener.run, daemon=True).start()
        logging.info(f"Perception Core initialisiert auf Gerät: {self.device}")

    def _on_move(self, x, y):
        with self.lock:
            self.mouse_pos = (x, y)
    def _on_click(self, x, y, button, pressed):
        with self.lock:
            if button in self.mouse_buttons_state: self.mouse_buttons_state[button] = 1 if pressed else 0
    def _on_scroll(self, x, y, dx, dy):
        with self.lock:
            self.scroll_delta_y += dy
    def _on_press(self, key):
        with self.lock:
            try: self.last_key_pressed = key.char.lower()
            except AttributeError: self.last_key_pressed = key

    def _perceive_vision(self) -> np.ndarray:
        screenshot_pil = self._take_screenshot()
        vectors = [self._process_image(screenshot_pil)]
        img_width, img_height = screenshot_pil.size
        tile_width, tile_height = img_width // 4, img_height // 4
        for i in range(4):
            for j in range(4):
                tile = screenshot_pil.crop((j * tile_width, i * tile_height, (j + 1) * tile_width, (i + 1) * tile_height))
                vectors.append(self._process_image(tile))
        return np.concatenate(vectors, axis=0)
    
    def _take_screenshot(self) -> Image:
        try:
            img = ImageGrab.grab()
            return img.convert("RGB")
        except Exception as e:
            logging.warning(f"Screenshot fehlgeschlagen: {e}. Verwende schwarzes Bild.")
            return Image.new("RGB", (self.screen_width, self.screen_height), color="black")

    def _process_image(self, image: Image) -> np.ndarray:
        inputs = self.image_processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.vision_model(**inputs)
        return outputs.pooler_output.cpu().numpy().flatten()
    def _perceive_tactile(self) -> (np.ndarray, np.ndarray):
        with self.lock:
            current_mouse_pos = self.mouse_pos
            current_buttons = self.mouse_buttons_state.copy()
            current_scroll_delta = self.scroll_delta_y
            current_key = self.last_key_pressed
            self.scroll_delta_y = 0.0
            self.last_key_pressed = None
        mouse_vector = np.array([current_mouse_pos[0]/self.screen_width, current_mouse_pos[1]/self.screen_height, current_buttons[mouse.Button.left], current_buttons[mouse.Button.right], current_buttons[mouse.Button.middle], np.clip(current_scroll_delta, -1.0, 1.0)], dtype=np.float32)
        key_vector = np.zeros(self.KEY_VECTOR_SIZE, dtype=np.float32)
        try:
            key_index = self._KEYS_TO_MAP.index(current_key)
        except (ValueError, AttributeError):
            key_index = -1
        key_vector[key_index] = 1.0
        return mouse_vector, key_vector
    def perceive(self) -> np.ndarray:
        vision_vectors = self._perceive_vision()
        mouse_vector, keyboard_vector = self._perceive_tactile()
        return np.concatenate([vision_vectors, mouse_vector, keyboard_vector])
    def close(self):
        logging.info("Beende Perception Core...")
        if self.mouse_listener.is_alive(): self.mouse_listener.stop()
        if self.keyboard_listener.is_alive(): self.keyboard_listener.stop()