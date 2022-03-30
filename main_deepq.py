from webbrowser import get
from game import Col, Game, Piece
from ai_deepq import ConnectFourAgent
import numpy as np
from random import choice
import json
from sys import exit as sys_exit


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
    env: Game = Game(dense=True, verbose=False)
    n_games: int = 50

    gamma: float = 0.9
    epsilon: float = 1.0
    alpha: float = 0.1
    input_dims: int = env.rows * env.cols
    n_actions: int = 7
    mem_size: int = 10
    batch_size: int = 16
    epsilon_end: float = 0.01
    epsilon_decrement: float = 0.99
    agent_yellow = ConnectFourAgent(gamma=gamma,
                                    epsilon=epsilon,
                                    alpha=alpha,
                                    input_dims=input_dims,
                                    n_actions=n_actions,
                                    mem_size=mem_size,
                                    batch_size=batch_size,
                                    epsilon_end=epsilon_end,
                                    epsilon_decrement=epsilon_decrement,
                                    fname='dqn_model_yellow.h5')

    agent_red = ConnectFourAgent(gamma=gamma,
                                 epsilon=epsilon,
                                 alpha=alpha,
                                 input_dims=input_dims,
                                 n_actions=n_actions,
                                 mem_size=mem_size,
                                 batch_size=batch_size,
                                 epsilon_end=epsilon_end,
                                 epsilon_decrement=epsilon_decrement,
                                 fname='dqn_model_red.h5')

    scores_yellow: list[int] = []
    scores_red: list[int] = []

    for i in range(n_games):
        done: bool = False
        score_red: int = 0
        score_yellow: int = 0
        rounds: int = 0
        observation = env.reset()
        while True:
            # Red Player
            action: Col = -1
            while action not in env.valid_columns:
                action = agent_red.choose_action(observation)

            observation_, reward, done = env.step(action, Piece.RED)
            score_red += reward
            reward_ = reward
            agent_red.remember(observation, action, reward, observation_, done)
            observation = observation_
            agent_red.learn()
            rounds += 1

            if done:
                reward_ = env.calculate_reward(Piece.YELLOW, -1, env.board)
                agent_yellow.remember(observation, action, reward_, observation_, done)
                agent_yellow.learn()
                score_yellow += reward_
                break

            # Yellow Player
            # env.print_board(number_row=True)
            # action: Col = get_player_input(env)
            while action not in env.valid_columns:
                action = agent_yellow.choose_action(observation)

            observation_, reward, done = env.step(action, Piece.YELLOW)
            score_yellow += reward
            agent_yellow.remember(observation, action, reward, observation_, done)
            observation = observation_
            agent_yellow.learn()
            rounds += 1

            if done:
                reward_ = env.calculate_reward(Piece.RED, -1, env.board)
                agent_red.remember(observation, action, reward_, observation_, done)
                agent_red.learn()
                score_red += reward_
                break

        scores_red.append(score_red)
        scores_yellow.append(score_yellow)

        avg_score_red = np.mean(scores_red[max(0, i - 50):(i + 1)])
        avg_score_yellow = np.mean(scores_yellow[max(0, i - 50):(i + 1)])
        _state = env.game_state
        state = 'r' if _state.numerator == -1 else 'y'
        state = 'd' if _state.numerator == 0 else state
        print(f'episode {i}:',
              f'rounds {rounds}',
              f'score red: {score_red:.0f}',
              f'score yel: {score_yellow:.0f}',
              f'average score red: {avg_score_red:.2f}',
              f'average score yel: {avg_score_yellow:.2f}',
              f'epsilon={agent_red.epsilon:.4f}',
              f'state: {state}',
              sep='\t')

        with open(f"./qgames/game_{i}.json", "a") as f:
            json.dump(list(map(lambda x: np.flipud(x).tolist(), env.board_history)), f)
        if not (i % 10) and i > 0:
            agent_red.save_model()
            agent_yellow.save_model()
