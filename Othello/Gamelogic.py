# game.get_moves()
# game.execute_move()
# game.undo_move()
# game.is_final()
# game.get_score()
# game.get_states()
# game.get_turn()


class Othello:

    def __init__(self):
        self._reset_board()
        self.history = []
        self.turn = 0
        self.score = [2, 2]
        self.calculated_moves = False
        self.legal_moves = []
        self.legal_directions = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]]
        self.inp_move_validation = False

        self.get_moves()

    def _reset_board(self):
        self.board = [[-1 for _ in range(8)] for _ in range(8)]
        self.board[3][3] = 0
        self.board[4][4] = 0
        self.board[3][4] = 1
        self.board[4][3] = 1

    def get_turn(self):
        return self.turn

    def get_moves(self):
        if self.calculated_moves:
            return self.legal_moves
        moves = []
        for row in range(8):
            for col in range(8):
                if self._pos_check(row, col):
                    moves.append((row, col))
        self.legal_moves = moves
        self.calculated_moves = True

    # Checking if a square is possible move
    def _pos_check(self, row, col):
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
        # False if the first step in that direction is not opponents piece
        if not self.board[row + direction[0]][col + direction[1]] == (self.turn + 1) % 2:
            return False

        # Continues moving in that direction
        for dir in range(2, 8):
            curr_row = row + dir * direction[0]
            curr_col = col + dir * direction[1]

            # Square outside the board
            if not (0 <= curr_row < 8) or not (0 <= curr_col < 8):
                return False

            # True if it reaches a piece of its kind
            if self.board[curr_row][curr_col] == self.turn:
                return True
            # False if it reaches a empty square
            elif self.board[curr_row][curr_col] == -1:
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

        # If the opponent can not move
        if len(self.legal_moves) == 0:
            self.calculated_moves = False
            self.turn = (self.turn + 1) % 2
            self.get_moves()

    # Turning the stones on the board, adjusting the points and updating the history
    def _turn_stones(self, move, turn, undoing=False, stop_point=(-1, -1)):
        if undoing:
            self.board[move[0]][move[1]] = -1
            self.score[turn] -= 1

        # Laying down the piece
        self.board[move[0]][move[1]] = turn
        self.score[turn] += 1

        # Turning stones all affected directions
        direction = self._directions_affected(move[0], move[1])
        changed = ()
        for dir in range(1, 8):
            curr_row = move[0] + dir * direction[0]
            curr_col = move[1] + dir * direction[1]

            # Square outside the board
            if not (0 <= curr_row < 8) or not (0 <= curr_col < 8):
                break

            # True if it reaches a piece of its kind
            if self.board[curr_row][curr_col] == turn:
                break

            changed = (curr_row, curr_col)

            # Adjust board and points
            if undoing:
                self.board[curr_row][curr_col] = (turn + 1) % 2

                self.score[turn] -= 1
                self.score[(turn + 1) % 2] += 1
            else:
                self.board[curr_row][curr_col] = turn

                self.score[turn] += 1
                self.score[(turn + 1) % 2] -= 1

            # Used when undoing: if it has reached the ast stone it turned
            if changed == stop_point:
                return

        if not undoing:
            self.history.append([turn, move, changed])

    def undo_move(self):
        to_undo = self.history.pop()
        self._turn_stones(to_undo[1], to_undo[1], undoing=True, stop_point=to_undo[2])

        # Resetting turn and calculating moves
        self.turn = to_undo[0]
        self.calculated_moves = False
        self.get_moves()

    # If the game is over
    def is_final(self):
        return True if len(self.legal_moves) == 0 else False

    # Get the score of the game
    def get_score(self):
        if not self.is_final():
            return None
        if self.score[0] == self.score[1]:
            return 1
        if self.turn == 0:
            return 2 if self.score[0] > self.score[1] else 0
        return 2 if self.score[0] < self.score[1] else 0

    def get_states(self):
        return [str(self.board)]
