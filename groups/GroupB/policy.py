import numpy as np
import math
import time
from connect4.policy import Policy
from typing import override



class Node:
    def __init__(self, state, player, parent=None, action=None):
        self.state = state
        self.player = player          # jugador que mueve en este nodo
        self.parent = parent
        self.action = action          # acción que llevó a este nodo
        self.children = []
        self.untried = self.valid_actions(state)
        self.wins = 0
        self.visits = 0

    def valid_actions(self, s):
        return [c for c in range(7) if s[0, c] == 0]

    def expand(self):
        c = self.untried.pop()
        child_state = apply_move(self.state, c, self.player)
        next_player = -self.player
        child = Node(child_state, next_player, self, c)
        self.children.append(child)
        return child

    def best_child(self, c_param=1.4):
        best = None
        best_val = -1e9
        for ch in self.children:
            ucb = (ch.wins / (ch.visits + 1e-9)) + \
                  c_param * math.sqrt(math.log(self.visits + 1) / (ch.visits + 1e-9))
            if ucb > best_val:
                best_val = ucb
                best = ch
        return best

    def update(self, result):
        self.visits += 1
        self.wins += result


def apply_move(state, col, p):
    b = state.copy()
    for r in range(5, -1, -1):
        if b[r, col] == 0:
            b[r, col] = p
            return b
    return b


def check_win(b, p):
    # horizontal
    for r in range(6):
        for c in range(4):
            if b[r,c]==p and b[r,c+1]==p and b[r,c+2]==p and b[r,c+3]==p:
                return True
    # vertical
    for r in range(3):
        for c in range(7):
            if b[r,c]==p and b[r+1,c]==p and b[r+2,c]==p and b[r+3,c]==p:
                return True
    # diag "\"
    for r in range(3):
        for c in range(4):
            if b[r,c]==p and b[r+1,c+1]==p and b[r+2,c+2]==p and b[r+3,c+3]==p:
                return True
    # diag "/"
    for r in range(3):
        for c in range(3,7):
            if b[r,c]==p and b[r+1,c-1]==p and b[r+2,c-2]==p and b[r+3,c-3]==p:
                return True
    return False


def rollout(state, player):
    b = state.copy()
    p = player
    while True:
        actions = [c for c in range(7) if b[0, c] == 0]
        if not actions:
            return 0

        c = np.random.choice(actions)
        b = apply_move(b, c, p)

        if check_win(b, p):
            return p 

        p = -p


def mcts(root_state, player, time_limit):
    root = Node(root_state, player)
    end = time.time() + time_limit

    while time.time() < end:
        node = root

        # 1. Selección
        while node.untried == [] and node.children:
            node = node.best_child()

        # 2. Expansión
        if node.untried:
            node = node.expand()

        # 3. Simulación
        result = rollout(node.state, node.player)

        if result == player:
            value = 1
        elif result == -player:
            value = -1
        else:
            value = 0

        # 4. Backpropagation
        while node is not None:
            node.update(value if node.player != player else -value)
            node = node.parent

   
    best = max(root.children, key=lambda ch: ch.visits)
    return best.action


class WinPolicy(Policy):

    @override
    def mount(self, time_out: int):
        pass

    @override
    def act(self, s: np.ndarray) -> int:
        total = np.count_nonzero(s)
        player = 1 if total % 2 == 0 else -1
        return mcts(s, player, time_limit=0.3)
