from TicTacToe import Gamelogic
import MCTS
import ResNet
from TicTacToe import Config
import Files

game = Gamelogic.TicTacToe()
config = Config

height, width, depth = game.get_board().shape
agent, agent1 = ResNet.ResNet.build(height, width, depth, 128, config.policy_output_dim, num_res_blocks=5,
                                    reg=0.0001)
agent3, agent4 = ResNet.ResNet.build(height, width, depth, 128, config.policy_output_dim, num_res_blocks=5,
                                     reg=0.0001)
print(agent.summary())

Files.load_model(agent, config.name+"_0_4_8", 2)
Files.load_model(agent3, config.name+"_complete", 0)

tree = MCTS.MCTS()
tree.dirichlet_noise = False
tree.NN_input_dim = config.board_dims
tree.policy_output_dim = config.policy_output_dim
tree.NN_output_to_moves_func = config.NN_output_to_moves
tree.move_to_number_func = config.move_to_number
tree.number_to_move_func = config.number_to_move
tree.set_game(game)
tree.set_evaluation(agent)
sum = 0
while True:
    print("Started\n\n")
    game.__init__()
    # game.execute_move(0)
    # game.execute_move(4)
    game.execute_move(8)
    game.execute_move(0)
    # game.execute_move(6)
    # game.execute_move(3)
    # game.execute_move(5)
    # for move in game.get_moves():
        # game.execute_move(move)
    while not game.is_final():
        game.print_board()
        input()
        print(game.get_moves())
        print(game.get_legal_NN_output())
        if game.get_turn() == 1:
            tree.set_evaluation(agent)
        else:
            tree.set_evaluation(agent3)
        # if game.get_turn() == 1:
        tree.search_series(180)
        print("post", tree.get_posterior_probabilities(game.get_state()))
        print("pri", tree.get_prior_probabilities(game.get_state()))
        game.execute_move(tree.get_most_seached_move(game.get_state()))
        tree.reset_search()
        # else:
        #     move = int(input("Move: "))
        #     game.execute_move(move)
        # game.print_board()
    sum += 1
    print("Finished\n\n")

        # game.__init__()
        # game.execute_move(0)
        # game.execute_move(4)
        # game.execute_move(8)
    # break