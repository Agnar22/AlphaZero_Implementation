board_dims = (1, 8, 8, 2)
policy_output_dim = (64, 1)


def array_to_moves(array):
    return [[num // 8, num % 8] for num, x in enumerate(array) if x > 0]
