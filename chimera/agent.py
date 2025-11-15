# chimera/agent.py (Finale Version mit Queue für Feedback)

import time
import logging
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from threading import Thread, Event
from torch.distributions import Normal
from queue import Queue, Empty

from chimera.perception import PerceptionCore
from chimera.state import AgentStateModule
from chimera.action import ActionCore
from chimera.visualizer import AgentVisualizer
from chimera.brain import Brain

class Agent:
    def __init__(self, shutdown_event: Event):
        self.shutdown_event = shutdown_event
        logging.info("Initialisiere Kern-Module des Agenten...")
        
        self.state = AgentStateModule()
        self.perception = PerceptionCore()
        self.action = ActionCore(state=self.state, screen_width=self.perception.screen_width, screen_height=self.perception.screen_height)
        
        perception_size = (PerceptionCore.NUM_VISION_VECTORS * PerceptionCore.VISION_FEATURE_SIZE + PerceptionCore.MOUSE_VECTOR_SIZE + PerceptionCore.KEY_VECTOR_SIZE)
        action_size = ActionCore.TOTAL_OUTPUT_VECTOR_SIZE
        
        self.brain = Brain(perception_size, action_size)
        self.device = self.perception.device
        self.brain.to(self.device)
        logging.info(f"Gehirn wurde auf Gerät '{self.device}' verschoben.")

        self.optimizer = optim.AdamW(self.brain.parameters(), lr=1e-4)
        self.loss_fn = nn.MSELoss()

        self._mouse_controller = self.action.mouse_controller
        self._screen_width = self.perception.screen_width
        self._screen_height = self.perception.screen_height

        self.visualizer = AgentVisualizer(self.state)

        # --- START DER ÄNDERUNG ---
        # Ersetze das einfache Signal durch eine thread-sichere Queue
        self.reward_queue = Queue()
        # --- ENDE DER ÄNDERUNG ---
        self.saved_log_probs = []

    def add_reward_to_queue(self, reward: float):
        """Wird vom ControlListener aufgerufen, um ein Feedback in die Queue zu legen."""
        self.reward_queue.put(reward)

    def _learn_from_feedback(self, total_reward: float):
        """Führt den Backpropagation-Schritt für RL aus."""
        if not self.saved_log_probs:
            return

        loss_list = [-log_prob * total_reward for log_prob in self.saved_log_probs]
        if not loss_list: return

        self.optimizer.zero_grad()
        policy_loss = torch.stack(loss_list).sum()
        policy_loss.backward()
        self.optimizer.step()
        self.saved_log_probs = []
        
    def _get_target_action_vector(self) -> torch.Tensor:
        mx, my = self._mouse_controller.position
        dx_norm = (mx / self._screen_width) * 2 - 1
        dy_norm = (my / self._screen_height) * 2 - 1
        mouse_vec = np.array([dx_norm, dy_norm, 0, 0, 0], dtype=np.float32)
        key_vec = np.zeros(ActionCore.KEYBOARD_ACTION_SIZE, dtype=np.float32)
        body_vec = np.zeros(ActionCore.BODY_ACTION_SIZE, dtype=np.float32)
        symbol_vec = np.zeros(ActionCore.SYMBOL_ACTION_SIZE, dtype=np.float32)
        target_vector = np.concatenate([mouse_vec, key_vec, body_vec, symbol_vec])
        return torch.from_numpy(target_vector).float().to(self.device)
        
    def clear_memory(self):
        self.saved_log_probs = []
        logging.info("Agenten-Kurzzeitgedächtnis geleert.")

    def reinitialize_optimizer(self):
        self.optimizer = optim.AdamW(self.brain.parameters(), lr=1e-4)
        logging.info("Optimizer-Zustand wurde zurückgesetzt.")

    def run(self):
        vis_thread = Thread(target=self.visualizer.run, daemon=True)
        vis_thread.start()
        # ... (logging)
        logging.info("\n--- Starte Agenten-Lebenszyklus ---")
        logging.info(">>> F8 = Reinforcement (Aktiv) | F9 = Observational (Lernen) <<<")
        logging.info(">>> STRG + ESC zum Beenden. <<<\n")

        while not self.shutdown_event.is_set():
            # ... (observational mode bleibt unverändert)
            if self.state.learning_mode == 'observational':
                self.brain.train()
                self.optimizer.zero_grad()
                zustandsvektor = self.perception.perceive()
                zustandsvektor_tensor = torch.from_numpy(zustandsvektor).float().to(self.device)
                mu, _ = self.brain(zustandsvektor_tensor.unsqueeze(0))
                predicted_action_vector = mu.squeeze(0)
                target_action_vector = self._get_target_action_vector()
                loss = self.loss_fn(predicted_action_vector, target_action_vector)
                loss.backward()
                self.optimizer.step()
                print(f"Observational Mode: Loss = {loss.item():.6f}", end='\r')
            
            elif self.state.learning_mode == 'reinforcement':
                self.brain.train()
                zustandsvektor = self.perception.perceive()
                zustandsvektor_tensor = torch.from_numpy(zustandsvektor).float().to(self.device)
                mu, log_std = self.brain(zustandsvektor_tensor.unsqueeze(0))
                std = torch.exp(log_std)
                dist = Normal(mu, std)
                action = dist.sample()
                action = torch.clamp(action, -1, 1)
                log_prob = dist.log_prob(action).sum()
                self.saved_log_probs.append(log_prob)
                self.action.execute(action.squeeze(0).detach().cpu().numpy())

                # --- START DER ÄNDERUNG ---
                # Verarbeite alle Belohnungen aus dem "Briefkasten"
                total_reward_this_cycle = 0.0
                try:
                    while True: # Leere die Queue vollständig
                        total_reward_this_cycle += self.reward_queue.get_nowait()
                except Empty:
                    pass # Die Queue ist jetzt leer

                if total_reward_this_cycle != 0.0:
                    self._learn_from_feedback(total_reward_this_cycle)
                # --- ENDE DER ÄNDERUNG ---
                
                print(f"Reinforcement Mode: Zyklus {self.state.daily_action_counter} | Memory Size: {len(self.saved_log_probs)}", end='\r')
            
            time.sleep(0.05)
    
    def shutdown(self):
        logging.info("Fahre Agenten-Module herunter...")
        if hasattr(self, 'perception'): self.perception.close()
        if hasattr(self, 'visualizer'): self.visualizer.close()