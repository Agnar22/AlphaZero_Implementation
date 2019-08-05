import ResNet
import MCTS
import time
import numpy as np
import Files
# from Othello import Gamerendering
# from Othello import Gamelogic
from TicTacToe import Gamelogic
from TicTacToe import Config
from keras.optimizers import Adam


# Potential causes of not improving play:
# [ ] the amount of data is to low/needs to train for a longer time
# [ ] there is a bug in the search
# [ ] it needs further tuning of hyper parameters
#
#
#   => train it for a long time and observe results
#   => simplify game
#   => change hyper parameters and observe different behaviour
#   => go through search line by line


def train(game, config, num_sim=800, epochs=100, games_pr_epoch=1000):
    Files.create_directories(config.name)

    height, width, depth = game.get_board().shape
    agent = ResNet.ResNet.build(height, width, depth, 64, config.policy_output_dim, num_res_blocks=2, reg=0.0001)
    agent.compile(loss=['categorical_crossentropy', 'mean_squared_error'], optimizer=Adam(lr=0.05, epsilon=10E-8))
    print(agent.summary())

    tree = MCTS.MCTS()
    tree.dirichlet_noise = False
    tree.NN_input_dim = config.board_dims
    tree.policy_output_dim = config.policy_output_dim
    tree.NN_output_to_moves_func = config.NN_output_to_moves
    tree.move_to_number_func = config.move_to_number
    tree.number_to_move_func = config.number_to_move
    tree.set_evaluation(agent)
    tree.set_game(game)

    for epoch in range(epochs):
        for game_num in range(games_pr_epoch):

            game.__init__()
            history = []
            policy_targets = []
            player_moved_list = []
            prior_probs = []
            positions = []

            # game.execute_move(0)
            # game.execute_move(3)
            # game.execute_move(1)
            # game.execute_move(4)

            # game.execute_move(0)
            # game.execute_move(1)
            # game.execute_move(4)
            # game.execute_move(8)
            # game.execute_move(3)
            # game.execute_move(6)

            while not game.is_final():
                tree.search_series(num_sim)

                positions.append(game.get_board())

                state = game.get_state()
                # print(tree.get_prior_probabilities(game.get_state()))
                prior_probs.append(tree.get_prior_probabilities(game.get_state()))
                most_searched_move = tree.get_most_seached_move(state)
                history.append(most_searched_move)
                policy_targets.append(tree.get_posterior_probabilities(state))
                player_moved_list.append(game.get_turn())

                game.execute_move(most_searched_move)
                tree.reset_search()

            game_outcome = game.get_outcome()
            value_targets = [game_outcome[x] for x in player_moved_list]
            print(policy_targets)

            agent.fit(
                x=np.array(positions),
                y=[np.array(policy_targets), np.array([[game_outcome[x]] for x in player_moved_list])],
                batch_size=len(positions), epochs=1
            )
            # print(agent.predict(np.array(positions)))
            Files.store_game(config.name, history, value_targets, policy_targets, epoch, game_outcome,
                             prior_probs=prior_probs)

        Files.save_model(agent, config.name, epoch)


train(Gamelogic.TicTacToe(), Config, num_sim=800, epochs=100, games_pr_epoch=1000)

# # Setting up game, ResNet and MCTS
# game = Gamelogic.TicTacToe()
# height, width, depth = game.get_board().shape
# agent = ResNet.ResNet.build(height, width, depth, 256)
# tree = MCTS.MCTS()
# tree.set_evaluation(agent)
# tree.set_game(game)
# print(agent.summary())
#
# now = time.time()
#
# for x in range(10000):
#     tree.search()
#     # print(tree.tree_children)
#
# print(time.time() - now)
# print(agent.predict(game.get_board().reshape(1, 8, 8, 2)))
# print(tree.get_search_probabilities(game.get_states()[0]))
# y_targ = tree.get_search_probabilities(game.get_states()[0])
# print(y_targ.shape)
# y_targ = y_targ.reshape(64)
# x = np.array([game.get_board().reshape(8, 8, 2) for _ in range(100)])
# agent.compile(loss=['categorical_crossentropy', 'mean_squared_error'], optimizer=Adam(lr=0.002, epsilon=10E-8))
# agent.fit(x, [np.array([y_targ for _ in range(100)]),
#               np.array([[1] for _ in range(100)])],
#           batch_size=32, epochs=15)
# print(agent.predict(game.get_board().reshape(1, 8, 8, 2)))
