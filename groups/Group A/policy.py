import time
import math
import random
import numpy as np
from connect4.policy import Policy
from typing import override


class MCTSAgent(Policy):
    @override
    def mount(self) -> None:

        pass

   
    def valid_actions(self, board: np.ndarray):
        return [c for c in range(board.shape[1]) if board[0, c] == 0]

    def apply_action(self, board: np.ndarray, col: int, player: int):
        b = board.copy()
        for r in range(b.shape[0] - 1, -1, -1):
            if b[r, col] == 0:
                b[r, col] = player
                return b
        return b  

    def check_winner(self, board: np.ndarray, player: int) -> bool:
        rows, cols = board.shape
        # Horizontal
        for r in range(rows):
            for c in range(cols - 3):
                if (
                    board[r, c] == player
                    and board[r, c + 1] == player
                    and board[r, c + 2] == player
                    and board[r, c + 3] == player
                ):
                    return True
        # Vertical
        for r in range(rows - 3):
            for c in range(cols):
                if (
                    board[r, c] == player
                    and board[r + 1, c] == player
                    and board[r + 2, c] == player
                    and board[r + 3, c] == player
                ):
                    return True
        # Diagonal 
        for r in range(rows - 3):
            for c in range(cols - 3):
                if (
                    board[r, c] == player
                    and board[r + 1, c + 1] == player
                    and board[r + 2, c + 2] == player
                    and board[r + 3, c + 3] == player
                ):
                    return True
        # Diagonal 
        for r in range(rows - 3):
            for c in range(3, cols):
                if (
                    board[r, c] == player
                    and board[r + 1, c - 1] == player
                    and board[r + 2, c - 2] == player
                    and board[r + 3, c - 3] == player
                ):
                    return True
        return False

    def terminal(self, board: np.ndarray) -> bool:
        return (
            self.check_winner(board, 1)
            or self.check_winner(board, -1)
            or np.all(board[0] != 0)
        )


    def immediate_winning_move(self, board: np.ndarray, player: int):
        for c in self.valid_actions(board):
            b2 = self.apply_action(board, c, player)
            if self.check_winner(b2, player):
                return c
        return None


    def rollout(self, board: np.ndarray, player: int):
        b = board.copy()
        current = player
        # juego hasta terminal
        while not self.terminal(b):
            acts = self.valid_actions(b)
            if not acts:
                break
            a = random.choice(acts)
            b = self.apply_action(b, a, current)
            current = -current
        if self.check_winner(b, player):
            return 1
        if self.check_winner(b, -player):
            return -1
        return 0


    def mcts(self, board: np.ndarray, player: int, time_limit: float = 8.5):
  
        N = {}  
        W = {}  
        def state_key(b: np.ndarray):
            return b.tobytes()

        def uct_value(state_b: np.ndarray, a: int):
            key = state_key(state_b)
            if (key, a) not in N:
                return float("inf")
            total = sum(N.get((key, x), 0) for x in self.valid_actions(state_b))
            if N[(key, a)] == 0:
                return float("inf")
            return (W[(key, a)] / N[(key, a)]) + math.sqrt(2 * math.log(max(1, total)) / N[(key, a)])

        start = time.time()
        max_time = time_limit  
        max_iters = 1000000

        iters = 0
        while iters < max_iters and (time.time() - start) < max_time:
            iters += 1
            path = []
            b = board.copy()
            current = player

            while True:
                acts = self.valid_actions(b)
                if not acts or self.terminal(b):
                    break

                unexplored = [a for a in acts if (state_key(b), a) not in N]
                if unexplored:
                    a = random.choice(unexplored)
                    path.append((state_key(b), a))
                    b = self.apply_action(b, a, current)
                    current = -current
                    break
                else:
                  
                    a = max(acts, key=lambda x: uct_value(b, x))
                    path.append((state_key(b), a))
                    b = self.apply_action(b, a, current)
                    current = -current

            # rollout desde b
            outcome = self.rollout(b, player)

            # Backpropagation
            for (k, a) in path:
                if (k, a) not in N:
                    N[(k, a)] = 0
                    W[(k, a)] = 0.0
                N[(k, a)] += 1
                W[(k, a)] += outcome

        
        actions = self.valid_actions(board)
        if not actions:
            return 0  

        root_key = state_key(board)
       
        def score_for(a):
            visits = N.get((root_key, a), 0)
            wins = W.get((root_key, a), 0.0)
            winrate = wins / visits if visits > 0 else -9999.0
            # preferencia por centro en desempate (columna 3)
            center_bonus = -abs(a - 3) * 1e-6
            return (visits, winrate, center_bonus)

        best = max(actions, key=score_for)
        return best


    @override
    def act(self, s: np.ndarray) -> int:
       
        
        total_pieces = int(np.count_nonzero(s))
        player = 1 if total_pieces % 2 == 0 else -1

      
        win = self.immediate_winning_move(s, player)
        if win is not None:
            return int(win)

       
        block = self.immediate_winning_move(s, -player)
        if block is not None:
            return int(block)

       
        if s[0, 3] == 0:
            return 3

        
        return int(self.mcts(s, player, time_limit=8.5))