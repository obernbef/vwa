from __future__ import annotations
from ctypes import Union
import numpy as np
from numpy.typing import NDArray
from enum import IntEnum
from typing import Callable, NamedTuple, Optional
"""Defining default values for rows and columns.
"""
ROWS: int = 6
COLS: int = 7

Col = int
Row = int
Board = NDArray

Position = NamedTuple('Position', [('row', Row), ('col', Col)])


class Piece(IntEnum):
    RED: int = -1
    YELLOW: int = 1
    EMPTY: int = 0

    def __invert__(self) -> Piece:
        if self == Piece.RED:
            return Piece.YELLOW
        elif self == Piece.YELLOW:
            return Piece.RED
        else:
            return Piece.EMPTY

    @staticmethod
    def to_str(piece: Piece) -> str:
        if piece == Piece.RED:
            return "X"
        elif piece == Piece.YELLOW:
            return "O"
        else:
            return "."

    def __str__(self) -> str:
        return self.to_str(self)


class GameState(IntEnum):
    """Defines the State of the game.
    """
    PLAYING: int = 2
    DRAW: int = 0
    RED_WON: int = Piece.RED.numerator
    YELLOW_WON: int = Piece.YELLOW.numerator


def is_winner(piece: Piece, state: GameState):
    return piece.numerator == state.numerator


class Game:
    """A class that handles all Game related logic.
    """
    board: Board
    board_history: list[Board]

    def __init__(
        self,
        rows: int = ROWS,
        cols: int = COLS,
        starting_piece: Piece = Piece.RED,
        dense: Optional[bool] = False,
        verbose: Optional[bool] = False,
    ) -> None:
        """Creates a Game instance.

        Parameters
        ----------
        rows : int, optional
            the number of rows of the game board, by default ROWS
        cols : int, optional
            the number of columns of the game board, by default COLS
        starting_piece : Piece, optional
            the piece / player that should start, may not be Piece.EMPTY, by default Piece.PLAYER_1
        """
        assert starting_piece != Piece.EMPTY
        self.rows: int = rows
        self.cols: int = cols
        self.starting_piece: Piece = starting_piece
        self.current_piece: Piece = starting_piece
        self.dense: bool = dense
        self.verbose: bool = verbose
        self.reset()

    @property
    def __dense_board(self) -> Board:
        return self.board.copy().reshape(-1, (self.rows * self.cols))[0]

    def reset(self) -> Board:
        """Resets board and board_history.
        """
        self.board = np.zeros((self.rows, self.cols), dtype=np.int8)
        self.board_history = []

        obs: Board = self.__dense_board if self.dense else self.board.copy()
        if self.verbose:
            print(self.board)
            if self.dense:
                print()
                print(obs)

        return obs

    def print_board(self, number_row: Optional[bool] = False, row_start: int = 0) -> None:
        if number_row:
            row = " ".join((str(r) for r in range(row_start, row_start + self.cols)))
            print(row)
        print("\n".join([" ".join([Piece.to_str(i) for i in j]) for j in self.board][::-1]))

    def drop(self, pos: Position, piece: Piece) -> None:
        """Drops a piece into the given position.

        Parameters
        ----------
        pos : Position
            the position of the
        piece : Piece
            the type of Piece
        """
        self.board[pos.row][pos.col] = piece
        self.board_history.append(self.board.copy())

    def is_valid_column(self, col: Col) -> bool:
        """Checks if there is still space in the given column.

        Parameters
        ----------
        col : Col
            the column to be checked

        Returns
        -------
        bool
            if there is still space
        """
        # We can say [rows - 1][col] (=> last slot in the column) because we just need to check if there is
        # still space in the row
        return self.board[self.rows - 1][col] == Piece.EMPTY

    @property
    def valid_columns(self) -> list[Col]:
        """The current valid columns of the board.

        Returns
        -------
        list[int]
            the valid columns
        """
        valid_locations: list[int] = []
        for col in range(self.cols):
            if self.is_valid_column(col):
                valid_locations.append(col)
        return valid_locations

    def get_next_open_row(self, col: Col) -> Row:
        """Returns the next empty row for a given column

        Parameters
        ----------
        col : Col
            the row to be searched

        Returns
        -------
        Row
            the next open row
        """
        for row in range(self.rows):
            if self.board[row][col] == Piece.EMPTY:
                return row

    def copy(self) -> Game:
        """Returns a copy of the board with new memory

        Returns
        -------
        Game
            the game copy
        """
        game = Game(self.rows, self.cols, starting_piece=self.current_piece)
        game.board = self.board.copy()
        return game

    @property
    def game_state(self) -> GameState:
        winner_found = False
        current_winner = None
        piece = None
        if not winner_found:
            # check horizontal locations for win
            for c in range(self.cols - 3):
                for r in range(ROWS):
                    if (self.board[r][c]) == self.board[r][c + 1] == self.board[r][ c + 2]\
                        == self.board[r][c + 3]:
                        piece = self.board[r][c]
                        if (piece) != Piece.EMPTY:
                            current_winner = Piece(piece)
                            winner_found = True

        if not winner_found:
            # check vertical locations for win
            for c in range(self.cols):
                for r in range(self.rows - 3):
                    if (self.board[r][c]) == self.board[r + 1][c] == self.board[r + 2][c] \
                        == self.board[r + 3][c]:
                        piece = self.board[r][c]
                        if (piece) != Piece.EMPTY:
                            current_winner = Piece(piece)
                            winner_found = True

        if not winner_found:
            # check positively sloped diagonals
            for c in range(self.cols - 3):
                for r in range(self.rows - 3):
                    if (self.board[r][c]) == self.board[r + 1][c + 1] == self.board[r + 2][c + 2]\
                        == self.board[r + 3][c + 3]:
                        piece = self.board[r][c]
                        if (piece) != Piece.EMPTY:
                            current_winner = Piece(piece)
                            winner_found = True

        if not winner_found:
            # check negatively sloped diagonals
            for c in range(self.cols - 3):
                for r in range(3, self.rows):
                    if (self.board[r][c]) == self.board[r - 1][c + 1] == self.board[r - 2][c + 2]\
                        == self.board[r - 3][c + 3]:
                        piece = self.board[r][c]
                        if (piece) != Piece.EMPTY:
                            current_winner = Piece(piece)
                            winner_found = True

        if winner_found:
            if current_winner == Piece.RED:
                return GameState.RED_WON
            elif current_winner == Piece.YELLOW:
                return GameState.YELLOW_WON
        else:
            drawFound = not Piece.EMPTY in self.board
            # Checks if there is an empty piece left in the board
            if drawFound:
                return GameState.DRAW
            else:
                return GameState.PLAYING

    def count_n_in_a_row(self,
                         piece: Union[Piece, list[Piece]],
                         n: int,
                         board: Optional[Board] = None) -> int:
        assert isinstance(piece, (Piece, int, list))
        if isinstance(piece, list):
            assert len(piece) == n
            assert isinstance(piece[0], (Piece, int))
            ctrl = piece
        else:
            ctrl: list[Piece] = [piece] * n

        board: Board = board if isinstance(board, np.ndarray) else self.board

        count: int = 0

        # cols
        for c in range(self.cols):
            for r in range(self.rows):
                if r + n > self.rows: continue
                batch: list[Piece] = [board[r + i][c] for i in range(n)]
                if batch == ctrl:
                    count += 1

        # rows
        for c in range(self.cols):
            for r in range(self.rows):
                if c + n > self.cols: continue
                batch: list[Piece] = [board[r][c + i] for i in range(n)]
                if batch == ctrl:
                    count += 1

        # positive diagonal
        for c in range(self.cols):
            for r in range(self.rows):
                if c + n > self.cols or r + n > self.rows:
                    continue

                batch: list[Piece] = [board[r + i][c + i] for i in range(n)]
                if batch == ctrl:
                    count += 1

        # negative diagonal
        for c in range(self.cols):
            for r in range(self.rows):
                if c + n > self.cols or r - n < 0:
                    continue

                batch: list[Piece] = [board[r - i][c + i] for i in range(n)]
                if batch == ctrl:
                    count += 1

        return count

    def count_two_in_a_row(self, piece: Piece, board: Optional[Board] = None) -> int:
        ctrl_a: list[Piece] = [piece, piece, Piece.EMPTY, Piece.EMPTY]
        ctrl_b: list[Piece] = [piece, Piece.EMPTY, Piece.EMPTY, piece]
        ctrl_c: list[Piece] = [Piece.EMPTY, Piece.EMPTY, piece, piece]
        ctrl_d: list[Piece] = [Piece.EMPTY, piece, piece, Piece.EMPTY]
        ctrl_d: list[Piece] = [piece, Piece.EMPTY, Piece.EMPTY, piece]
        return self.count_n_in_a_row(ctrl_a, 4, board=board) \
               + self.count_n_in_a_row(ctrl_b, 4, board=board) \
               + self.count_n_in_a_row(ctrl_c, 4, board=board) \
               + self.count_n_in_a_row(ctrl_d, 4, board=board)

    def count_three_in_a_row(self, piece: Piece, board: Optional[Board] = None) -> int:
        ctrl_a: list[Piece] = [piece, piece, piece, Piece.EMPTY]
        ctrl_b: list[Piece] = [Piece.EMPTY, piece, piece, piece]
        ctrl_c: list[Piece] = [piece, piece, Piece.EMPTY, piece]
        ctrl_d: list[Piece] = [piece, Piece.EMPTY, piece, piece]
        return self.count_n_in_a_row(ctrl_a, 4, board=board) \
               + self.count_n_in_a_row(ctrl_b, 4, board=board) \
               + self.count_n_in_a_row(ctrl_c, 4, board=board) \
               + self.count_n_in_a_row(ctrl_d, 4, board=board)

    def count_three_blocked(self, piece: Piece, board: Optional[Board] = None) -> int:
        ctrl_a: list[Piece] = [~piece, ~piece, ~piece, piece]
        ctrl_b: list[Piece] = [piece, ~piece, ~piece, ~piece]
        ctrl_c: list[Piece] = [~piece, ~piece, piece, ~piece]
        ctrl_d: list[Piece] = [~piece, piece, ~piece, ~piece]
        return self.count_n_in_a_row(ctrl_a, 4, board=board) \
               + self.count_n_in_a_row(ctrl_b, 4, board=board) \
               + self.count_n_in_a_row(ctrl_c, 4, board=board) \
               + self.count_n_in_a_row(ctrl_d, 4, board=board)

    def calculate_reward(self, piece: Piece, col: Col, old_board: Board) -> int:
        game_state: GameState = self.game_state
        rew: int = 0

        if is_winner(game_state, piece):
            return 45
        elif is_winner(game_state, ~piece):
            return -50

        def diff(func: Callable[[Piece, Board], int], p: Piece) -> int:
            return func(p, self.board)
            # o: int = func(p, old_board)
            # n: int = func(p, self.board)
            # if o <= n: return n - o
            # else: return 0

        rew += diff(self.count_three_blocked, piece) * 7
        rew -= diff(self.count_three_blocked, ~piece) * 3
        rew += diff(self.count_three_in_a_row, piece) * 5
        rew -= diff(self.count_three_in_a_row, ~piece) * 5
        rew += diff(self.count_two_in_a_row, piece) * 2
        rew -= diff(self.count_two_in_a_row, ~piece) * 1
        rew += 4 if col == 3 and col in self.valid_columns else 0

        return rew

    def step(self,
             action: Col,
             piece: Piece,
             calculate_reward: Optional[bool] = True) -> tuple[Board, int, bool]:
        """A function that performs the following tasks:

        - drops piece in the given column
        - generates a reward for the current board and piece
        - gets the current game state and turns it into a boolean
          specifying if the game has ended or not

        Returns
        -------
        tuple[Board, int, bool]
            new observation, reward, done
        """
        reward: int = 0
        old_board: Board = self.board.copy()

        # If the chosen column is not a valid column the board doesn't
        # change and the reward is very negative
        if action not in self.valid_columns:
            obs: Board = self.__dense_board if self.dense else self.board.copy()
            reward -= 1000
            return obs, reward, False

        # Play move
        pos: Position = Position(row=self.get_next_open_row(action), col=action)
        self.drop(pos, piece)

        # Calculate the reward
        if calculate_reward: reward = self.calculate_reward(piece, action, old_board)

        obs: Board = self.__dense_board if self.dense else self.board.copy()
        done: bool = not self.game_state == GameState.PLAYING

        if self.verbose:
            print(f"{piece} played {action}")
            print()
            print(self.board)
            if self.dense:
                print()
                print(obs)

        return obs, reward, done
