# %%
from typing import Optional
from game import Board, Col, Game, GameState, Piece, Position
from person import Player
from platform import system as platform
from os import system


class GameController:
    def __init__(self, game: Game, red_player: Player, yellow_player: Player) -> None:
        assert red_player.piece != yellow_player.piece
        self.game: Game = game
        self.red_player: Player = red_player
        self.yellow_player: Player = yellow_player
        self.training_history: list[tuple[GameState, Board]] = []
        self.outcome_history: list[GameState] = []

    def play_game(self, verbose: Optional[bool] = False, clear: Optional[bool] = False) -> None:
        if clear:
            if platform() == "Windows": system("cls")
            else: system("clear")
        player_to_move: Player = self.red_player if self.game.starting_piece == Piece.RED else self.yellow_player
        while self.game.game_state == GameState.PLAYING:
            move_col: Col = player_to_move.get_move(self.game)
            move: Position = Position(col=move_col, row=self.game.get_next_open_row(move_col))
            self.game.drop(move, player_to_move.piece)
            if player_to_move == self.red_player:
                player_to_move = self.yellow_player
            else:
                player_to_move = self.red_player

        if verbose:
            self.game.print_board()
            if self.game.game_state == GameState.DRAW:
                print("It's a draw!")
            else:
                print(
                    f"Player {'Red' if self.game.game_state == GameState.RED_WON else 'Yellow'} won!"
                )
        self.outcome_history.append(self.game.game_state)
        for history_item in self.game.board_history:
            self.training_history.append((self.game.game_state, history_item.copy()))

    def simulate_games(self,
                       number_of_games: int,
                       verbose: bool = True,
                       pverbose: bool = False) -> None:
        red_player_wins: int = 0
        yellow_player_wins: int = 0
        draws = 0
        for i in range(number_of_games):
            if verbose:
                print(f"Game Number {i}")
            self.game.reset()
            self.play_game(verbose=pverbose)
            if self.game.game_state == GameState.RED_WON:
                red_player_wins += 1
            elif self.game.game_state == GameState.YELLOW_WON:
                yellow_player_wins += 1
            else:
                draws += 1

        total_games = red_player_wins + yellow_player_wins + draws
        print(f'Red Wins:    {int(red_player_wins * 100 / total_games)}%')
        print(f'Yellow Wins: {int(yellow_player_wins * 100 / total_games)}%')
        print(f'Draws:       {int(draws * 100 / total_games)}%')