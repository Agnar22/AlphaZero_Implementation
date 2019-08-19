from TicTacToe import Gamelogic
import MCTS
import ResNet
from TicTacToe import Config
import Files

game = Gamelogic.TicTacToe()
config = Config

height, width, depth = game.get_board().shape
agent, agent1 = ResNet.ResNet.build(height, width, depth, 128, config.policy_output_dim, num_res_blocks=4,
                                    reg=0.0001)
print(agent.summary())

Files.load_model(agent, config.name, 0)

tree = MCTS.MCTS()
tree.dirichlet_noise = False
tree.NN_input_dim = config.board_dims
tree.policy_output_dim = config.policy_output_dim
tree.NN_output_to_moves_func = config.NN_output_to_moves
tree.move_to_number_func = config.move_to_number
tree.number_to_move_func = config.number_to_move
tree.set_evaluation(agent)
tree.set_game(game)

while True:
    print("Started\n\n")
    game.__init__()
    game.execute_move(0)
    game.execute_move(4)
    game.execute_move(8)


    while not game.is_final():
        game.print_board()
        print(game.get_moves())
        print(game.get_legal_NN_output())

        if game.get_turn() == 0:
            tree.search_series(2)
            tree.search()
            print("post", tree.get_posterior_probabilities(game.get_state()))
            print("pri",tree.get_prior_probabilities(game.get_state()))
            game.execute_move(tree.get_most_seached_move(game.get_state()))
            tree.reset_search()
        else:
            move = int(input("Move: "))
            game.execute_move(move)
    game.print_board()
    print("Finished\n\n")