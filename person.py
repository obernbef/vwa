#%%
from enum import Enum
import math
from random import choice
from typing import Generic, Optional, TypeVar, Union

from game import Col, Game, Piece, Position
from ai_keras import ConnectFourModel
from ai_deepq import ConnectFourAgent
from ai_minimax import minimax


class Strategy(Enum):
    MODEL: str = "model"
    RANDOM: str = "random"
    INPUT: str = "input"
    MINIMAX: str = "minimax"


Model = Union[ConnectFourAgent, ConnectFourModel]

S = TypeVar("S", bound=Strategy)


class Player(Generic[S]):
    def __init__(
        self,
        piece: Piece,
        strategy: S,
        model: Optional[Model] = None,
        minimax_depth: Optional[int] = None,
    ) -> None:
        if strategy == Strategy.MODEL:
            assert model != None
        elif strategy == Strategy.MINIMAX:
            assert minimax_depth != None
        self.piece: Piece = piece
        self.strategy: Strategy = strategy
        self.model: Model = model
        self.depth: int = minimax_depth

    def get_move(self, game: Game) -> Col:
        if self.strategy == Strategy.RANDOM:
            return choice(game.valid_columns)
        elif self.strategy == Strategy.INPUT:
            print(f"You are Player {Piece.to_str(self.piece)}.")
            print("1 2 3 4 5 6 7")
            game.print_board()
            chosen: Col = -1
            while chosen == -1:
                try:
                    _chosen = int(
                        input(
                            f"Choose your column! {list(map(lambda x: x + 1, game.valid_columns))} "
                        ))
                    _chosen -= 1
                    if _chosen in game.valid_columns:
                        chosen = _chosen
                    else:
                        print(f"{_chosen} is not a valid column.")
                except ValueError:
                    pass
                except KeyboardInterrupt:
                    print("\nEnding game...")
                    from sys import exit
                    exit()
                except:
                    pass
            return chosen

        elif self.strategy == Strategy.MINIMAX:
            col: Col = minimax(game, self.piece, self.depth, -math.inf, math.inf, True)[0]
            return col

        elif self.strategy == Strategy.MODEL:
            if isinstance(self.model, ConnectFourModel):
                max_value: int = 0
                best_col: Col = game.valid_columns[0]
                for column in game.valid_columns:
                    pos: Position = Position(col=column, row=game.get_next_open_row(column))
                    temp_game = game.copy()
                    temp_game.drop(pos, self.piece)
                    if self.piece == Piece.RED:
                        value = self.model.predict(temp_game.board, 2)
                    else:
                        value = self.model.predict(temp_game.board, 0)
                    if value > max_value:
                        max_value = value
                        best_col = column
                return best_col
            elif isinstance(self.model, ConnectFourAgent):
                pass
        else:
            raise ValueError(f"Strategy {self.strategy} is not defined.")