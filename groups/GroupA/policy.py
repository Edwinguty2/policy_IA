import numpy as np
import math
import time
import pickle
import os
from connect4.policy import Policy
from typing import override, Dict

C_PARAM = 1.414 

class StateStats:
    def __init__(self, wins=0.0, visits=0):
        self.wins = wins
        self.visits = visits

class Node:
    def __init__(self, state, player, parent=None, action=None):
        self.state = state
        self.player = player
        self.parent = parent
        self.action = action
        self.children = []
        self.untried = [c for c in range(7) if state[0, c] == 0]
        self.wins = 0.0
        self.visits = 0

    def expand(self):
        action = self.untried.pop()
        next_state = apply_move(self.state, action, self.player)
        child_node = Node(next_state, -self.player, parent=self, action=action)
        self.children.append(child_node)
        return child_node

    def best_child(self):
        best = None
        best_val = -float('inf')
        log_n = math.log(self.visits) if self.visits > 0 else 0
        for child in self.children:
            if child.visits == 0: return child
            ucb = (child.wins / child.visits) + C_PARAM * math.sqrt(log_n / child.visits)
            if ucb > best_val:
                best_val = ucb
                best = child
        return best

    def update(self, reward):
        self.visits += 1
        self.wins += reward

def apply_move(state, col, p):
    b = state.copy()
    for r in range(5, -1, -1):
        if b[r, col] == 0:
            b[r, col] = p
            return b
    return b

def check_win(b, p):
    for r in range(6):
        for c in range(4):
            if b[r,c]==p and b[r,c+1]==p and b[r,c+2]==p and b[r,c+3]==p: return True
    for c in range(7):
        for r in range(3):
            if b[r,c]==p and b[r+1,c]==p and b[r+2,c]==p and b[r+3,c]==p: return True
    for r in range(3):
        for c in range(4):
            if b[r,c]==p and b[r+1,c+1]==p and b[r+2,c+2]==p and b[r+3,c+3]==p: return True
            if b[r+3,c]==p and b[r+2,c+1]==p and b[r+1,c+2]==p and b[r,c+3]==p: return True
    return False

def fast_rollout(state, player):
    b = state.copy()
    curr = player
    while True:
        valid = [c for c in range(7) if b[0, c] == 0]
        if not valid: return 0
        move = valid[np.random.randint(len(valid))]
        for r in range(5, -1, -1):
            if b[r, move] == 0:
                b[r, move] = curr
                break
        if check_win(b, curr): return curr
        curr = -curr

def run_mcts(root_state, player, time_limit, knowledge_base):
    root = Node(root_state, player)
    
    root_key = root_state.tobytes()
    if root_key in knowledge_base:
        s = knowledge_base[root_key]
        root.visits = s.visits
        root.wins = s.wins

    start_time = time.time()
    while (time.time() - start_time) < time_limit:
        node = root
        while node.untried == [] and node.children:
            node = node.best_child()
        
        if node.untried:
            node = node.expand()
            k = node.state.tobytes()
            if k in knowledge_base:
                node.visits = knowledge_base[k].visits
                node.wins = knowledge_base[k].wins

        winner = fast_rollout(node.state, node.player)
        
        curr = node
        while curr.parent is not None:
            move_maker = curr.parent.player
            reward = 1.0 if winner == move_maker else (0.0 if winner == -move_maker else 0.5)
            curr.update(reward)
            
            k = curr.state.tobytes()
            if k not in knowledge_base:
                # Crea objeto pesado
                knowledge_base[k] = StateStats()
            knowledge_base[k].visits += 1
            knowledge_base[k].wins += reward
            
            curr = curr.parent
        root.visits += 1

    if not root.children: return 0
    return max(root.children, key=lambda c: c.visits).action

class WinortzPolicy(Policy):
    def __init__(self):
        self.time_out = 10
        self.knowledge_base = {}
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.knowledge_file = os.path.join(current_dir, "brain_full.pkl")

    @override
    def mount(self, time_out: int) -> None:
        self.time_out = float(time_out)
        if os.path.exists(self.knowledge_file):
            try:
                with open(self.knowledge_file, "rb") as f:
                    self.knowledge_base = pickle.load(f)
            except: self.knowledge_base = {}

    @override
    def act(self, s: np.ndarray) -> int:
        return run_mcts(s, 1 if np.count_nonzero(s)%2==0 else -1, self.time_out, self.knowledge_base)

    def save_knowledge(self):
        try:
            with open(self.knowledge_file, "wb") as f:
                pickle.dump(self.knowledge_base, f)
            print(f"Cerebro guardado: {len(self.knowledge_base)} estados.")
        except Exception as e:
            print(e)
