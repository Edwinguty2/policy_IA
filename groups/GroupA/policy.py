import numpy as np
import math
import time
import pickle
import os
import gzip
from connect4.policy import Policy
from typing import override

C_PARAM = 1.414  # Constante de exploración para UCB1

class StateStats:
    """
    Clase para almacenar estadísticas de un estado específico.
    """
    __slots__ = ['wins', 'visits']
    def __init__(self, wins=0.0, visits=0):
        self.wins = wins
        self.visits = visits

class Node:
    """
    Representa un nodo en el árbol de búsqueda MCTS.
    Almacena el estado del tablero, estadísticas de victorias/visitas y la estructura del árbol.
    """
    __slots__ = ['state', 'player', 'parent', 'action', 'children', 'untried', 'wins', 'visits']
    def __init__(self, state, player, parent=None, action=None):
        self.state = state
        self.player = player
        self.parent = parent
        self.action = action
        self.children = []
        # Identificar columnas válidas (no llenas) para expansión futura
        self.untried = [c for c in range(7) if state[0, c] == 0]
        self.wins = 0.0
        self.visits = 0

    def expand(self):
        """
        Expande el árbol creando un nuevo nodo hijo a partir de una acción no probada.
        """
        action = self.untried.pop()
        next_state = apply_move(self.state, action, self.player)
        # El nuevo nodo tendrá el turno del jugador opuesto
        child_node = Node(next_state, -self.player, parent=self, action=action)
        self.children.append(child_node)
        return child_node

    def best_child(self):
        """
        Selecciona el mejor nodo hijo utilizando la fórmula de UCB1.
        
        """
        best = None
        best_val = -float('inf')
        # Evitar error log(0) si es la primera visita al padre (caso borde)
        log_n = math.log(self.visits) if self.visits > 0 else 0
        
        for child in self.children:
            # Prioridad absoluta a nodos no visitados para evitar división por cero
            if child.visits == 0: return child
            
            ucb = (child.wins / child.visits) + C_PARAM * math.sqrt(log_n / child.visits)
            if ucb > best_val:
                best_val = ucb
                best = child
        return best

    def update(self, reward):
        """Actualiza las estadísticas del nodo tras una simulación."""
        self.visits += 1
        self.wins += reward


def apply_move(state, col, p):
    """Aplica una jugada en la columna dada y retorna el nuevo estado del tablero."""
    b = state.copy()
    # Iterar de abajo hacia arriba para encontrar la primera celda vacía
    for r in range(5, -1, -1):
        if b[r, col] == 0:
            b[r, col] = p
            return b
    return b

def check_win(b, p):
    """Verifica si el jugador p ha ganado en horizontal, vertical o diagonal."""
    # Horizontal
    for r in range(6):
        for c in range(4):
            if b[r,c]==p and b[r,c+1]==p and b[r,c+2]==p and b[r,c+3]==p: return True
    # Vertical
    for c in range(7):
        for r in range(3):
            if b[r,c]==p and b[r+1,c]==p and b[r+2,c]==p and b[r+3,c]==p: return True
    # Diagonal principal
    for r in range(3):
        for c in range(4):
            if b[r,c]==p and b[r+1,c+1]==p and b[r+2,c+2]==p and b[r+3,c+3]==p: return True
            if b[r+3,c]==p and b[r+2,c+1]==p and b[r+1,c+2]==p and b[r,c+3]==p: return True
    return False

def fast_rollout(state, player):
    """
    Ejecuta una simulación aleatoria rápida (Rollout) desde el estado actual.
    Limita la profundidad a 20 movimientos para optimizar tiempo de cómputo.
    """
    b = state.copy()
    curr = player
    for _ in range(20): 
        valid = [c for c in range(7) if b[0, c] == 0]
        if not valid: return 0 # Empate
        
        # Selección aleatoria uniforme de movimiento válido
        move = valid[np.random.randint(len(valid))]
        for r in range(5, -1, -1):
            if b[r, move] == 0:
                b[r, move] = curr
                break
        
        if check_win(b, curr): return curr
        curr = -curr
    return 0

# --- Motor de Búsqueda MCTS ---

def run_mcts(root_state, player, time_limit, knowledge_base):
    """
    Ejecuta el algoritmo Monte Carlo Tree Search dentro del límite de tiempo establecido.
    Integra conocimiento persistente (knowledge_base) para inicializar nodos conocidos.
    """
    root = Node(root_state, player)
    
    # Cargar estadísticas previas si el estado raíz ya fue visitado en entrenamientos anteriores
    root_key = root_state.tobytes()
    if root_key in knowledge_base:
        s = knowledge_base[root_key]
        root.visits = s.visits
        root.wins = s.wins

    start_time = time.time()
    
    # Bucle principal de búsqueda limitado por tiempo
    while (time.time() - start_time) < time_limit:
        # Ejecución por lotes (50 iteraciones) para reducir la sobrecarga de time.time()
        for _ in range(50): 
            node = root
            
            # 1. Selección
            while node.untried == [] and node.children:
                node = node.best_child()
            
            # 2. Expansión
            if node.untried:
                node = node.expand()
                # Consultar base de conocimiento para inicializar estadísticas del nuevo nodo
                k = node.state.tobytes()
                if k in knowledge_base:
                    node.visits = knowledge_base[k].visits
                    node.wins = knowledge_base[k].wins

            # 3. Simulación
            winner = fast_rollout(node.state, node.player)
            
            # 4. Backpropagation
            curr = node
            while curr.parent is not None:
                move_maker = curr.parent.player
                # Asignar recompensa relativa al jugador que realizó el movimiento
                reward = 1.0 if winner == move_maker else (0.0 if winner == -move_maker else 0.5)
                curr.update(reward)
                
                # Actualizar base de conocimiento en memoria
                k = curr.state.tobytes()
                if k not in knowledge_base:
                    knowledge_base[k] = StateStats()
                knowledge_base[k].visits += 1
                knowledge_base[k].wins += reward
                
                curr = curr.parent
            root.visits += 1
            
        # Verificación de tiempo límite tras completar el lote
        if (time.time() - start_time) > time_limit: break

    # Retornar la acción del nodo hijo más visitado
    if not root.children:
        valid = [c for c in range(7) if root_state[0, c] == 0]
        return valid[0] if valid else 0
    return max(root.children, key=lambda c: c.visits).action

class WinortzPolicy(Policy):
    def __init__(self):
        self.time_out = 9
        self.knowledge_base = {}
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.knowledge_file = os.path.join(current_dir, "brain_optimized.pkl.gz")

    @override
    def mount(self, time_out: int) -> None:
        """
        Inicializa la política: carga el límite de tiempo y la base de conocimiento.
        """
        self.time_out = float(time_out)
        
        if os.path.exists(self.knowledge_file):
            try:
                with gzip.open(self.knowledge_file, "rb") as f:
                    raw_data = pickle.load(f)
                # Reconstrucción de objetos StateStats a partir de datos serializados
                self.knowledge_base = {
                    k: StateStats(wins=v[0], visits=v[1]) 
                    for k, v in raw_data.items()
                }
            except: self.knowledge_base = {}

    @override
    def act(self, s: np.ndarray) -> int:
        total = np.count_nonzero(s)
        player = 1 if total % 2 == 0 else -1
        
        # Limitar a 1.5s para evitar timeout
        limit = min(self.time_out * 0.9, 1.5)
        # Reducir tiempo en fases finales del juego
        if total > 30: limit = 0.5
        
        return run_mcts(s, player, limit, self.knowledge_base)

    def save_smart_knowledge(self, min_visits=5, max_states=40000):
        
        # Filtrado por relevancia
        candidates = [(k, v) for k, v in self.knowledge_base.items() if v.visits >= min_visits]
        
        if len(candidates) > max_states:
            candidates.sort(key=lambda item: item[1].visits, reverse=True)
            candidates = candidates[:max_states]
            
       
        optimized = {k: (v.wins, v.visits) for k, v in candidates}
        
        try:
            with gzip.open(self.knowledge_file, "wb") as f:
                pickle.dump(optimized, f)
            print(f"Datos guardados: {len(optimized)} estados procesados.")
            # Actualización de memoria local con datos optimizados
            self.knowledge_base = {k: StateStats(wins=v[0], visits=v[1]) for k, v in optimized.items()}
        except Exception as e:
            print(e)
