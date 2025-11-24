import numpy as np
import os
import sys

sys.path.append(os.getcwd())
try:
    from groups.GroupA.policy import WinortzPolicy
    class RandomPolicy:
        def mount(self, t): pass
        def act(self, s):
            v = [c for c in range(7) if s[0,c]==0]
            return np.random.choice(v) if v else 0
except: sys.exit()

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

def train_cycle(episodes=50):
    hero = WinortzPolicy()
    hero.mount(0.5) 
    rival = RandomPolicy()

    print(f"INICIANDO ENTRENAMIENTO DE ({episodes} Partidas)")
    
    wins = 0
    for i in range(episodes):
        board = np.zeros((6, 7))
        player = 1
        game_over = False
        
        while not game_over:
            agent = hero if player == 1 else rival
            try: action = agent.act(board)
            except: action = 0
            
            for r in range(5, -1, -1):
                if board[r, action] == 0:
                    board[r, action] = player
                    break
            
            if check_win(board, player):
                game_over = True
                if agent == hero: wins += 1
            elif np.count_nonzero(board) >= 42:
                game_over = True
            player = -player
        print(".", end="", flush=True)

    print(f"\nWins: {wins}")
    
    hero.save_knowledge()

if __name__ == "__main__":
    train_cycle(episodes=50)