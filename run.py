import logging
from threading import Event, Thread
from pynput import keyboard
from chimera.agent import Agent, AgentStateModule


class KillSwitchListener:
    """Ein robuster Listener für die STRG + ESC Not-Aus-Kombination."""
    def __init__(self, shutdown_event: Event):
        self.shutdown_event = shutdown_event
        self.pressed_keys = set()
        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)

    def on_press(self, key):
        self.pressed_keys.add(key)
        if (keyboard.Key.ctrl_l in self.pressed_keys or keyboard.Key.ctrl_r in self.pressed_keys) and \
           keyboard.Key.esc in self.pressed_keys:
            logging.warning("STRG + ESC gedrückt! Not-Aus wird eingeleitet...")
            self.shutdown_event.set()
            return False

    def on_release(self, key):
        try:
            self.pressed_keys.remove(key)
        except KeyError:
            pass

    def start(self):
        Thread(target=self.listener.run, daemon=True).start()

    def stop(self):
        if self.listener.running:
            self.listener.stop()

from chimera.agent import Agent

# Die Klasse wird umbenannt und erweitert
class ControlListener:
    """Überwacht Hotkeys für Lernmodus-Wechsel und Feedback-Signale."""
    def __init__(self, agent: Agent):
        self.agent = agent
        self.listener = keyboard.Listener(on_press=self.on_press)

    def on_press(self, key):
        try:
            # ... (Feedback-Logik bleibt gleich)
            if key.char == '+':
                self.agent.set_reward(1.0)
                print("\n--- Belohnung (+) registriert ---")
            elif key.char == '-':
                self.agent.set_reward(-1.0)
                print("\n--- Bestrafung (-) registriert ---")
        except AttributeError:
            if key == keyboard.Key.f8:
                if self.agent.state.learning_mode != 'reinforcement':
                    self.agent.state.learning_mode = 'reinforcement'
                    self.agent.clear_memory()
                    self.agent.reinitialize_optimizer() # <-- OPTIMIZER ZURÜCKSETZEN
                    print("\n--- Modus zu REINFORCEMENT (Aktiv/Lernen) gewechselt ---")
            elif key == keyboard.Key.f9:
                if self.agent.state.learning_mode != 'observational':
                    self.agent.state.learning_mode = 'observational'
                    self.agent.clear_memory()
                    self.agent.reinitialize_optimizer() # <-- OPTIMIZER ZURÜCKSETZEN
                    print("\n--- Modus zu OBSERVATIONAL (Passiv/Imitieren) gewechselt ---")

    def start(self):
        Thread(target=self.listener.run, daemon=True).start()

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    shutdown_event = Event()
    kill_switch = KillSwitchListener(shutdown_event)
    kill_switch.start()
    
    agent = None
    try:
        agent = Agent(shutdown_event)
        control_listener = ControlListener(agent) # <- Angepasste Instanziierung
        control_listener.start()
        agent.run()
    except Exception as e:
        logging.error(f"Ein kritischer Fehler im Agenten-Zyklus ist aufgetreten: {e}", exc_info=True)
        shutdown_event.set()
    finally:
        logging.info("\nProgramm wird beendet...")
        if agent:
            agent.shutdown()
        kill_switch.stop()
        logging.info("Alle Module heruntergefahren. Programm beendet.")

if __name__ == "__main__":
    main()