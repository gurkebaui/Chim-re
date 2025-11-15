# Unverändert seit Sprint 2
import numpy as np
from pynput.mouse import Button, Controller as MouseController
from pynput.keyboard import Controller as KeyboardController
import logging
from chimera.state import AgentStateModule
from chimera.perception import PerceptionCore

class ActionCore:
    MOUSE_ACTION_SIZE = 5
    KEYBOARD_ACTION_SIZE = len(PerceptionCore._KEYS_TO_MAP)
    BODY_ACTION_SIZE = 2
    SYMBOL_ACTION_SIZE = 1
    TOTAL_OUTPUT_VECTOR_SIZE = MOUSE_ACTION_SIZE + KEYBOARD_ACTION_SIZE + BODY_ACTION_SIZE + SYMBOL_ACTION_SIZE

    def __init__(self, state: AgentStateModule, screen_width: int, screen_height: int):
        self.agent_state = state
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.mouse_controller = MouseController()
        self.keyboard_controller = KeyboardController()
        self.MOUSE_SENSITIVITY = 50.0 
        self.BODY_SENSITIVITY = 30.0

    def _execute_mouse_action(self, action_vector: np.ndarray):
        # Für Imitationslernen: Vektor ist absolute Position, kein Delta
        target_x_norm, target_y_norm, left_click_prob, right_click_prob, middle_click_prob = action_vector
        target_x = (target_x_norm + 1) / 2 * self.screen_width
        target_y = (target_y_norm + 1) / 2 * self.screen_height
        self.mouse_controller.position = (int(target_x), int(target_y))
        if left_click_prob > 0.8: self.mouse_controller.click(Button.left)
        if right_click_prob > 0.8: self.mouse_controller.click(Button.right)
        if middle_click_prob > 0.8: self.mouse_controller.click(Button.middle)

    def _execute_keyboard_action(self, action_vector: np.ndarray):
        if np.max(action_vector) > 0.9:
            key_index = np.argmax(action_vector)
            key_to_press = PerceptionCore._KEYS_TO_MAP[key_index]
            try:
                self.keyboard_controller.press(key_to_press)
                self.keyboard_controller.release(key_to_press)
            except Exception as e:
                logging.warning(f"Konnte Taste '{key_to_press}' nicht drücken: {e}")

    def _execute_body_action(self, action_vector: np.ndarray):
        delta_x, delta_y = action_vector
        current_x, current_y = self.agent_state.position_on_screen
        new_x = int(current_x + delta_x * self.BODY_SENSITIVITY)
        new_y = int(current_y + delta_y * self.BODY_SENSITIVITY)
        self.agent_state.position_on_screen = (max(0, min(self.screen_width - 64, new_x)), max(0, min(self.screen_height - 64, new_y)))

    def _execute_symbol_action(self, action_vector: np.ndarray):
        if action_vector[0] > 0.9: logging.info("Symbol-Aktion ausgelöst.")

    def execute(self, output_vector: np.ndarray):
        if output_vector.shape[0] != self.TOTAL_OUTPUT_VECTOR_SIZE: return
        s = 0
        mouse_vec = output_vector[s:s+self.MOUSE_ACTION_SIZE]; s += self.MOUSE_ACTION_SIZE
        key_vec = output_vector[s:s+self.KEYBOARD_ACTION_SIZE]; s += self.KEYBOARD_ACTION_SIZE
        body_vec = output_vector[s:s+self.BODY_ACTION_SIZE]; s += self.BODY_ACTION_SIZE
        symbol_vec = output_vector[s:s+self.SYMBOL_ACTION_SIZE]
        self._execute_mouse_action(mouse_vec)
        self._execute_keyboard_action(key_vec)
        self._execute_body_action(body_vec)
        self._execute_symbol_action(symbol_vec)
        self.agent_state.daily_action_counter += 1