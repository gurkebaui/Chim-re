from dataclasses import dataclass

@dataclass
class AgentStateModule:
    """
    Ein einfacher Daten-Container, der den physischen Zustand des Agenten verwaltet.
    """
    position_on_screen: tuple[int, int] = (100, 100)
    animation_state: str = 'idle'
    daily_action_counter: int = 0
    learning_mode: str = 'reinforcement' # Standardmäßig im aktiven Modus