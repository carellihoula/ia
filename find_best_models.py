import os
import torch
import numpy as np
from itertools import combinations

# Fonctions auxiliaires (à remplacer par vos propres fonctions)
from utile import has_tile_to_flip, initialze_board, get_legal_moves

def load_model(model_path, device):
    model = torch.load(model_path, map_location=device)
    model.eval()
    return model
def apply_flip(best_move,board_stat,NgBlackPsWhith):
    """
    Apply tile flipping on the Othello board based on the best move.

    Parameters:
    - best_move (tuple): Coordinates (row, column) of the best move.
    - board_stat (numpy.ndarray): 2D array representing the current state of the Othello board.
    - NgBlackPsWhith (int): Indicator for the current player (Black: -1, White: 1).

    Returns:
    - numpy.ndarray: Updated Othello board after applying tile flipping.
    """
    
    MOVE_DIRS = [(-1, -1), (-1, 0), (-1, +1),
             (0, -1),           (0, +1),
             (+1, -1), (+1, 0), (+1, +1)]

    for direction in MOVE_DIRS:
        if has_tile_to_flip(best_move, direction,board_stat,NgBlackPsWhith):
            i = 1
            while True:
                row = best_move[0] + direction[0] * i
                col = best_move[1] + direction[1] * i
                if board_stat[row][col] == board_stat[best_move[0], best_move[1]]:
                    break
                else:
                    board_stat[row][col] = board_stat[best_move[0], best_move[1]]
                    i += 1
                    
    return board_stat

def find_best_move(move1_prob,legal_moves):
    """
    Finds the best move based on the provided move probabilities and legal moves.

    Parameters:
    - move1_prob (numpy.ndarray): 2D array representing the probabilities of moves.
    - legal_moves (list): List of legal moves.

    Returns:
    - tuple: The best move coordinates (row, column).
    """

    # Initialize the best move with the first legal move
    best_move=legal_moves[0]
    
    # Initialize the maximum score with the probability of the first legal move
    max_score=move1_prob[legal_moves[0][0],legal_moves[0][1]]
    
    # Iterate through all legal moves to find the one with the maximum probability
    for i in range(len(legal_moves)):
        # Update the best move if the current move has a higher probability
        if move1_prob[legal_moves[i][0],legal_moves[i][1]]>max_score:
            max_score=move1_prob[legal_moves[i][0],legal_moves[i][1]]
            best_move=legal_moves[i]
    return best_move

def input_seq_generator(board_stats_seq,length_seq):
    
    board_stat_init=initialze_board()

    if len(board_stats_seq) >= length_seq:
        input_seq=board_stats_seq[-length_seq:]
    else:
        input_seq=[board_stat_init]
        #Padding starting board state before first index of sequence
        for i in range(length_seq-len(board_stats_seq)-1):
            input_seq.append(board_stat_init)
        #adding the inital of game as the end of sequence sample
        for i in range(len(board_stats_seq)):
            input_seq.append(board_stats_seq[i])
            
    return input_seq

def play_match(model1, model2, device):
    # Initialisation du plateau de jeu
    board_stat = initialze_board()
    board_stats_seq = []
    pass2player = False
    current_model = model1
    next_model = model2

    while not np.all(board_stat) and not pass2player:
        # Sélection du modèle courant et mise à jour du plateau
        board_stats_seq.append(board_stat.copy())
        input_seq = input_seq_generator(board_stats_seq, current_model.len_inpout_seq)
        model_input = torch.tensor([input_seq]).float().to(device)
        move_prob = current_model(model_input).cpu().detach().numpy().reshape(8, 8)

        # Trouver et appliquer le meilleur coup
        NgBlackPsWhith = -1 if current_model is model1 else 1
        legal_moves = get_legal_moves(board_stat, NgBlackPsWhith)

        if legal_moves:
            best_move = find_best_move(move_prob, legal_moves)
            board_stat[best_move[0], best_move[1]] = NgBlackPsWhith
            board_stat = apply_flip(best_move, board_stat, NgBlackPsWhith)
        else:
            if pass2player:
                break
            pass2player = True

        # Échanger les modèles pour le prochain tour
        current_model, next_model = next_model, current_model

    # Résultat du match
    score = np.sum(board_stat)
    if score < 0:
        return 'win' if model1 is current_model else 'loss'
    elif score > 0:
        return 'loss' if model1 is current_model else 'win'
    else:
        return 'draw'

def run_tournament(models_dir, device):
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pt')]
    models = {model_file: load_model(os.path.join(models_dir, model_file), device) for model_file in model_files}

    scores = {model: {'wins': 0, 'losses': 0, 'draws': 0} for model in model_files}

    for model1, model2 in combinations(model_files, 2):
        print(f"Match: {model1} vs {model2}")
        result = play_match(models[model1], models[model2], device)
        
        if result == 'win':
            scores[model1]['wins'] += 1
            scores[model2]['losses'] += 1
        elif result == 'loss':
            scores[model1]['losses'] += 1
            scores[model2]['wins'] += 1
        else:
            scores[model1]['draws'] += 1
            scores[model2]['draws'] += 1

    return scores

def main():
    models_dir = 'save_models_LSTM'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    tournament_results = run_tournament(models_dir, device)

    # Affichage des résultats
    for model, score in sorted(tournament_results.items(), key=lambda x: (x[1]['wins'], -x[1]['losses']), reverse=True):
        print(f"Model {model}: Wins: {score['wins']}, Losses: {score['losses']}, Draws: {score['draws']}")

if __name__ == "__main__":
    main()
