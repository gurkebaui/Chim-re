# chimera/visualizer.py (Robuste, nicht-transparente Version)

import tkinter as tk
from PIL import Image, ImageTk
import logging
from threading import Event
from chimera.state import AgentStateModule

class AgentVisualizer:
    def __init__(self, state: AgentStateModule):
        self.state = state
        self.root = None
        self._shutdown_event = Event()

    def _setup_window(self):
        self.root = tk.Tk()
        try:
            image = Image.open("avatar.png")
            self.photo_image = ImageTk.PhotoImage(image)
        except FileNotFoundError:
            logging.warning("avatar.png nicht gefunden! Verwende leeres Bild.")
            image = Image.new('RGBA', (64, 64), (255, 0, 0, 255)) # Rotes Quadrat als Fallback
            self.photo_image = ImageTk.PhotoImage(image)
        
        # Rahmenloses Fenster, immer im Vordergrund
        self.root.overrideredirect(True)
        self.root.geometry(f"+{self.state.position_on_screen[0]}+{self.state.position_on_screen[1]}")
        self.root.wm_attributes("-topmost", True)
        
        # Einfaches Label ohne Transparenz
        label = tk.Label(self.root, image=self.photo_image, bd=0)
        label.pack()

    def _update_loop(self):
        if self._shutdown_event.is_set():
            try:
                self.root.destroy()
            except tk.TclError:
                pass # Fenster wurde bereits zerstört
            return
            
        if self.root:
            self.root.geometry(f"+{self.state.position_on_screen[0]}+{self.state.position_on_screen[1]}")
            self.root.after(50, self._update_loop)

    def run(self):
        try:
            logging.info("Visualizer-Thread gestartet.")
            self._setup_window()
            self.root.after(50, self._update_loop)
            self.root.mainloop()
        except Exception as e:
            logging.error(f"Visualizer-Thread abgestürzt: {e}")
        finally:
            logging.info("Visualizer-Thread beendet.")

    def close(self):
        self._shutdown_event.set()