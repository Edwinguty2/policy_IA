import numpy as np
import os
import sys

sys.path.append(os.getcwd())
try:
    from groups.GroupA.policy import MCTSPolicy
    from groups.GroupB.policy import WinPolicy
except:
    print("Error importando policies")
    sys.exit()

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
    nueva = MCTSPolicy()
    nueva.mount(0.5) 
    
    try: 
        vieja = WinPolicy()
        vieja.mount(0.5)
    except: 
        print("No hay rival, entrenando contra random.")
        class RandomPolicy:
            def act(self, s): 
                valid = [c for c in range(7) if s[0,c]==0]
                return np.random.choice(valid) if valid else 0
        vieja = RandomPolicy()

    print(f"INICIANDO ENTRENAMIENTO: {episodes} Partidas")
    print(f"Memoria Inicial: {len(nueva.knowledge_base)} estados")

    wins = 0
    
    for i in range(episodes):
        board = np.zeros((6, 7))
        player = 1
        game_over = False
        moves = 0
        
        p1, p2 = (nueva, vieja) if i % 2 == 0 else (vieja, nueva)
        
        while not game_over:
            if p1 is None or p2 is None: break
            
            agent = p1 if player == 1 else p2
            
            try:
                if agent == vieja:
                    try: action = agent.act(board)
                    except: action = np.random.choice([c for c in range(7) if board[0,c]==0])
                else:
                    action = agent.act(board)
            except: break 
            
            for r in range(5, -1, -1):
                if board[r, action] == 0:
                    board[r, action] = player
                    break
            
            if check_win(board, player):
                game_over = True
                if (agent == nueva): wins += 1
            elif moves >= 41:
                game_over = True
            
            player = -player
            moves += 1
            
        print(".", end="", flush=True)

    print(f"\nVictorias Nueva: {wins}/{episodes} ({ (wins/episodes)*100 }%)")
    
    nueva.save_smart_knowledge(min_visits=3, max_states=40000)

if __name__ == "__main__":
    train_cycle(episodes=50)
