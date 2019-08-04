# TODO: more effective way of finding moves
# [ ] effective move finding by storing opponent pieces in each direction
# [ ] effective move finding by bit shifting and parallelization
# [X] redo/undo more than once does not show correct moves when one player moves double


# game.get_moves()
# game.execute_move()
# game.undo_move()
# game.is_final()
# game.get_score()
# game.get_states()
# game.get_turn()

import numpy as np
import random


class Othello:

    def __init__(self):
        self._reset_board()
        self.history = []
        self.turn = 0
        self.score = [2, 2]
        self.calculated_moves = False
        self.legal_moves = []
        self.position_moves = {}
        self.legal_directions = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]]
        self.inp_move_validation = False

        self.get_moves()

    def _reset_board(self):
        self.board = np.zeros([8, 8, 2])
        self.board[3, 3, 0] = 1
        self.board[4, 4, 0] = 1
        self.board[3, 4, 1] = 1
        self.board[4, 3, 1] = 1

    def print_game(self):
        for x in range(8):
            line = ""
            for y in range(8):
                if self.board[x, y, 0] == 1:
                    line += '1'
                elif self.board[x, y, 1] == 1:
                    line += '2'
                else:
                    line += '0'
            print(line)
        print()
        print()

    def get_board(self):
        return self.board

    def get_turn(self):
        return self.turn

    def get_legal_moves_matrix(self):
        mat = np.zeros((8, 8))
        for pos in self.legal_moves:
            mat[pos[0], pos[1]] = 1
        return mat

    def get_moves(self):
        if self.calculated_moves or self.get_state() in self.position_moves:
            self.legal_moves = self.position_moves[self.get_state()]
            self.calculated_moves = True
            return self.position_moves[self.get_state()]
        moves = []
        for row in range(8):
            for col in range(8):
                if self._pos_check(row, col):
                    moves.append([row, col])
        self.legal_moves = moves
        self.calculated_moves = True
        self.position_moves[self.get_state()] = moves
        return self.legal_moves

    # Checking if a square is possible move
    def _pos_check(self, row, col):
        if self.board[row, col, 0] == 1 or self.board[row, col, 1] == 1:
            return False
        for dir in self.legal_directions:
            if self._direction_check(row, col, dir):
                return True

    def _directions_affected(self, row, col):
        directions = []
        for dir in self.legal_directions:
            if self._direction_check(row, col, dir):
                directions.append(dir)
        return directions

    # Finding if a move will turn stones in one direction
    def _direction_check(self, row, col, direction):
        if not (0 <= row + direction[0] < 8) or not (0 <= col + direction[1] < 8):
            return False
        # False if the first step in that direction is not opponents piece
        if not self.board[row + direction[0], col + direction[1], (self.turn + 1) % 2] == 1:
            return False

        # Continues moving in that direction
        for dir in range(2, 8):
            curr_row = row + dir * direction[0]
            curr_col = col + dir * direction[1]

            # Square outside the board
            if not (0 <= curr_row < 8) or not (0 <= curr_col < 8):
                return False

            # True if it reaches a piece of its kind
            if self.board[curr_row, curr_col, self.turn] == 1:
                return True
            # False if it reaches a empty square
            elif self.board[curr_row, curr_col, (self.turn + 1) % 2] == 0:
                return False
        return False

    def execute_move(self, move):
        # Input validation
        if self.inp_move_validation and not self._pos_check(move[0], move[1]):
            print("Not a legal move!")
            return None

        # Turning stones on the board
        self._turn_stones(move, self.turn)

        # Changing turn and finding new moves
        self.calculated_moves = False
        self.turn = (self.turn + 1) % 2
        self.get_moves()
        # print(self.turn, self.get_moves())

        # If the opponent can not move
        if len(self.legal_moves) == 0:
            print('changing turn')
            self.calculated_moves = False
            self.turn = (self.turn + 1) % 2
            self.get_moves()
            print('once', self.get_moves())

    # Turning the stones on the board, adjusting the points and updating the history
    def _turn_stones(self, move, turn, undoing=False, stop_points=[[-1, -1] for _ in range(8)], directions=None):
        if undoing:
            self.board[move[0], move[1], turn] = 0
            self.score[turn] -= 1

        # Laying down the piece
        else:
            self.board[move[0], move[1], turn] = 1
            self.score[turn] += 1

        # Turning stones in all affected directions
        if directions is None:
            directions = self._directions_affected(move[0], move[1])

        final_points = []  # Final locations where a piece was turned
        for num, dir in enumerate(directions):
            changed = []
            for step in range(1, 8):
                curr_row = move[0] + step * dir[0]
                curr_col = move[1] + step * dir[1]

                # Square outside the board
                if not (0 <= curr_row < 8) or not (0 <= curr_col < 8):
                    break

                # True if it reaches a piece of its kind
                if self.board[curr_row, curr_col, turn] == 1 and not undoing:
                    break

                changed = [curr_row, curr_col]

                # Adjust board and points
                if undoing:
                    self.board[curr_row, curr_col, (turn + 1) % 2] = 1
                    self.board[curr_row, curr_col, turn] = 0

                    self.score[turn] -= 1
                    self.score[(turn + 1) % 2] += 1

                else:
                    self.board[curr_row, curr_col, turn] = 1
                    self.board[curr_row, curr_col, (turn + 1) % 2] = 0

                    self.score[turn] += 1
                    self.score[(turn + 1) % 2] -= 1

                # Used when undoing: if it has reached the last stone it turned
                if changed == stop_points[num]:
                    break

            if not undoing:
                final_points.append(changed)

        if not undoing:
            self.history.append([move, turn, final_points, directions])

    def undo_move(self):
        to_undo = self.history.pop()
        self._turn_stones(to_undo[0], to_undo[1], undoing=True, stop_points=to_undo[2], directions=to_undo[3])

        # Resetting turn and calculating moves
        self.turn = to_undo[1]
        self.calculated_moves = False

    # If the game is over
    def is_final(self):
        return True if len(self.get_moves()) == 0 else False

    # Get the score of the game
    def get_score(self):
        if not self.is_final():
            return None
        if self.score[0] == self.score[1]:
            return 1
        if self.turn == 0:
            return 2 if self.score[0] > self.score[1] else 0
        return 2 if self.score[0] < self.score[1] else 0

    def get_state(self):
        # return [str(self.board)]
        return str(self.history)+str(self.turn)


# game = Othello()
#
# while True:
#     print("Down")
#     game.print_game()
#     while not game.is_final() and len(game.history) < 100:
#         game.print_game()
#         print(game.get_moves())
#         print(game.get_legal_moves_matrix())
#         move = [int(x) for x in input("Move: ").split()]
#         # moves = game.get_moves()
#         # move = moves[random.randint(0, len(moves)) - 1]
#         if move[0] == -1:
#             game.undo_move()
#         else:
#             game.execute_move(move)
#     print("Up")
#     game.print_game()
#     while len(game.history) > 0:
#         game.undo_move()
#         game.print_game()
