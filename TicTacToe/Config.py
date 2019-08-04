name = "TicTacToe"
board_dims = (1, 3, 3, 2)
policy_output_dim = (9)


def NN_output_to_moves(array):
    return [num for num, x in enumerate(array) if x > 0]


def number_to_move(number):
    return int(number)


def move_to_number(action):
    return int(action)
