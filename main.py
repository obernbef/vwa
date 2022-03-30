from ai_minimax import minimax
from game import Game, GameState
from person import Player, Piece, Strategy
from game_controller import GameController
from ai_keras import ConnectFourModel


def main() -> None:
    first_game: Game = Game()
    red_player: Player = Player(Piece.RED, strategy=Strategy.RANDOM)
    yellow_player: Player = Player(Piece.YELLOW, strategy=Strategy.RANDOM)

    game_controller: GameController = GameController(first_game, red_player, yellow_player)
    print("Playing with both players with random strategies")
    game_controller.simulate_games(1000)

    inputs: int = first_game.cols * first_game.rows
    outputs: int = 3  # draw, red won or yellow won
    batch_size: int = 50
    epochs: int = 100
    model: ConnectFourModel = ConnectFourModel(inputs, outputs, batch_size, epochs)
    model.train(game_controller.training_history)

    yellow_ai_player: Player = Player(Piece.YELLOW, strategy=Strategy.MODEL, model=model)
    second_game: Game = Game()
    game_controller: Game = GameController(second_game, red_player, yellow_ai_player)
    print("Playing with yellow player as Neural Network")
    game_controller.simulate_games(1000)


def keras_random():
    game: Game = Game()

    batch_size: int = 16
    epochs: int = 32
    model: ConnectFourModel = ConnectFourModel(game.rows * game.cols, 3, batch_size, epochs)

    red_player: Player[Strategy.RANDOM] = Player(piece=Piece.RED, strategy=Strategy.RANDOM)
    yellow_player: Player[Strategy.MODEL] = Player(piece=Piece.YELLOW,
                                                   strategy=Strategy.MODEL,
                                                   model=model)

    controller: GameController = GameController(game=game,
                                                red_player=red_player,
                                                yellow_player=yellow_player)
    controller.simulate_games(1000)
    print(controller.outcome_history)


def minimax_person():
    game: Game = Game(starting_piece=Piece.RED)
    minimax_player = Player(Piece.YELLOW, Strategy.MINIMAX, minimax_depth=5)
    input_player = Player(Piece.RED, Strategy.INPUT)

    controller: GameController = GameController(game,
                                                red_player=input_player,
                                                yellow_player=minimax_player)

    controller.play_game(verbose=True, clear=True)


if __name__ == "__main__":
    minimax_person()
