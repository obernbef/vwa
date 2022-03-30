__author__ = "https://github.com/KeithGalli/Connect4-Python/"
__editor__ = "Florian Obernberger"

import math
import random
from typing import Optional

from game import Col, Game, GameState, Piece, Position, is_winner


def minimax(game: Game, piece: Piece, depth: int, alpha: int, beta: int,
            maximize: bool) -> tuple[Optional[Col], int]:
    valid_locations: list[Col] = game.valid_columns
    game_state: GameState = game.game_state
    if depth == 0 or game_state != GameState.PLAYING:
        if is_winner(piece, game_state):
            return (None, 100000000000000)
        elif is_winner(piece, game_state):
            return (None, -10000000000000)
        elif game_state == GameState.DRAW:
            return (None, 0)
        else:  # Depth is zero
            return (None, game.calculate_reward(piece))
    if maximize:
        value: int = -math.inf
        column: Col = random.choice(valid_locations)
        for col in valid_locations:
            pos: Position = Position(row=game.get_next_open_row(col), col=col)
            g_copy: Game = game.copy()
            g_copy.drop(pos, piece)
            new_score: int = minimax(g_copy, piece, depth - 1, alpha, beta, False)[1]
            if new_score > value:
                value = new_score
                column = col
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return column, value

    else:  # Minimizing player
        value: int = math.inf
        column: Col = random.choice(valid_locations)
        for col in valid_locations:
            pos: Position = Position(row=game.get_next_open_row(col), col=col)
            g_copy = game.copy()
            g_copy.drop(pos, ~piece)
            new_score = minimax(g_copy, ~piece, depth - 1, alpha, beta, True)[1]
            if new_score < value:
                value = new_score
                column = col
            beta = min(beta, value)
            if alpha >= beta:
                break
        return column, value