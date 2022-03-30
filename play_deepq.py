from typing import NamedTuple
from ai_deepq import ConnectFourAgent
from game import Board, Game, Piece, Col
from sys import exit as sys_exit
from os import system
from platform import system as platform

ai_piece: Piece = Piece.RED
human_piece: Piece = ~ai_piece
GamesResult = NamedTuple('GamesResult', [('red_wins', int), ('yel_wins', int)])


def play() -> None:
    env: Game = Game(dense=True, verbose=False, starting_piece=ai_piece)
    agent: ConnectFourAgent = ConnectFourAgent(gamma=0.99,
                                               epsilon=0.001,
                                               alpha=0.0005,
                                               input_dims=env.rows * env.cols,
                                               n_actions=7,
                                               mem_size=1_000_000,
                                               batch_size=64,
                                               epsilon_end=0.01,
                                               fname="dqn_model_red.h5")

    # Clear the TensorFlow output
    if platform() == "Windows": system("cls")
    else: system("clear")

    agent.load_model()
    done: bool = False

    observation: Board = env.reset()
    while True:
        ai_action: Col = agent.choose_action(observation)
        _, _, done = env.step(ai_action, ai_piece, calculate_reward=False)

        if done:

            break

        env.print_board(number_row=True)
        print(f"You are player {human_piece}")
        human_action: Col = get_player_input(env)
        observation_, _, done = env.step(human_action, human_piece, calculate_reward=False)

        observation = observation_

    env.print_board(number_row=True)
    print("Human Won")


def playRandom() -> None:
    env: Game = Game(dense=True, verbose=False, starting_piece=ai_piece)
    agent: ConnectFourAgent = ConnectFourAgent(gamma=0.99,
                                               epsilon=0.001,
                                               alpha=0.0005,
                                               input_dims=env.rows * env.cols,
                                               n_actions=7,
                                               mem_size=1_000_000,
                                               batch_size=64,
                                               epsilon_end=0.001,
                                               fname="dqn_model_red.h5")

    # Clear the TensorFlow output
    if platform() == "Windows": system("cls")
    else: system("clear")

    agent.load_model()
    for j in range(5):
        yellow_wins: int = 0
        red_wins: int = 0
        for i in range(100):
            done: bool = False
            observation: Board = env.reset()
            while True:
                ai_action: Col = agent.choose_action(observation)
                _, _, done = env.step(ai_action, ai_piece, calculate_reward=False)

                if done:
                    red_wins += 1
                    break

                # env.print_board(number_row=True)
                # print(f"You are player {human_piece}")
                human_action: Col = choice(env.valid_columns)
                observation_, _, done = env.step(human_action, human_piece, calculate_reward=False)

                observation = observation_

                if done:
                    yellow_wins += 1
                    break

        print("Red (ai): ", red_wins)
        print("Yellow: ", yellow_wins)


def get_player_input(env: Game) -> Col:
    move: Col = -1
    opts: list[Col] = env.valid_columns
    while move not in opts:
        try:
            move = Col(input(f"Choose a column {str(opts)} "))
        except KeyboardInterrupt:
            print("\nEnding game...")
            sys_exit()
        except:
            # Please... just input the correct stuff...
            pass

    return move


if __name__ == "__main__":
    play()
