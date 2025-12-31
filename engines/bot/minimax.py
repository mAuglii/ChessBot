from .evaluation import get_evaluation
import numpy as np

def ordered_moves(board):
  moves = list(board.legal_moves)
  # checks first, then captures, then others
  moves.sort(key=lambda m: (board.gives_check(m), board.is_capture(m)), reverse=True)
  return moves


def minimax(board, depth, alpha, beta, maximizing_player):
  if depth == 0 or board.is_game_over():
    return get_evaluation(board)
  
  if maximizing_player:
    max_eval = -np.inf
    for move in ordered_moves(board):
      board.push(move)
      eval = minimax(board, depth - 1, alpha, beta, False)
      board.pop()
      max_eval = max(max_eval, eval)
      alpha = max(alpha, eval)
      if beta <= alpha:
        break
    return max_eval
  else:
    min_eval = np.inf
    for move in ordered_moves(board):
      board.push(move)
      eval = minimax(board, depth - 1, alpha, beta, True)
      board.pop()
      min_eval = min(min_eval, eval)
      beta = min(beta, eval)
      if beta <= alpha:
        break
    return min_eval