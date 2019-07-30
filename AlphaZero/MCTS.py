import math
import time
import numpy as np
from Othello import Gamelogic


class MCTS:
    def __init__(self):
        # #######DICTIONARIES#######
        self.search_dict = {}  # [visit cnt, sum. act. val., avg. act val, prior prob.] for each state-act
        self.pos_move_dict = {}  # Possible actions for each state
        self.state_visits = {}  # total visits for each state

        # #######PARAMETERS#######
        self.c_puct = 4  # Used for exploration (larger=>less long term exploration)
        self.c_init = 1  # Used for exploration (larger=>more exploration)
        self.dirichlet_noise = True  # Add dirichlet noise to the prior probabilities of the root
        self.alpha = 0.3  # Dirichlet noise variable

        # #######I/O shape for eval.#######
        self.NN_input_dim = (1, 8, 8, 2)
        self.policy_output_dim = (64, 1)

        self.tree_children = [0 for _ in range(61)]

        self.time_1 = 0
        self.time_2 = 0
        self.time_3 = 0
        self.time_4 = 0

    # Setting the game the MCTS will be used on
    def set_game(self, game):
        self.game = game

    # Setting the evaluation algorithm used by the MCTS
    def set_evaluation(self, eval):
        self.eval = eval

    # Returning a dictionary with action as key and visit number as value
    def get_action_numbers(self):
        state = self._return_one_state(self.game.get_states())
        actions = self.pos_move_dict.get(state)
        return {str(action): self.search_dict.get(action)[0] for action in actions}

    # Number of consecutive searches in the tree
    def search_series(self, number):
        for _ in range(number):
            self.search()

    # The main method for the MCTS-tree search
    def search(self):
        # Selection
        now = time.time()
        state, action = self._selection()
        self.time_1 += time.time() - now

        # The search traversed an internal node
        if action is not None:
            backp_value = self.search()
            now = time.time()
            # Negating the back-propagated value if it is the opponent to move
            to_move = self.game.get_turn()
            self.game.undo_move()
            moved = self.game.get_turn()

            if to_move is not moved:
                backp_value = -backp_value

            # Backward pass
            self._backward_pass(state, str(state) + '-' + str(action), backp_value)
            self.time_2 += time.time() - now
            return backp_value

        # Evaluation
        now = time.time()
        value, priors = self._evaluate(self.game.get_board())

        # Expansion
        self._expansion(state, priors)
        self.tree_children[len(self.game.history)] += 1
        self.time_3 += time.time() - now
        return value

    # Selecting the path from the root node to the leaf node and returning the new state and the last action executed
    def _selection(self):

        # Current state of the game
        state = self._return_one_state(self.game.get_states())

        if state not in self.pos_move_dict:  # New state encountered
            return state, None
        if self.game.is_final():  # Game is over
            return state, None

        best_action = ''
        best_value = None

        # Iterating through all actions to find the best
        for action in self.pos_move_dict.get(state):
            state_action_value = self.PUCT(self.search_dict.get(str(state) + '-' + str(action)),
                                           self.state_visits.get(state))
            if best_value is None or state_action_value > best_value:  # If new best action is found
                best_value = state_action_value
                best_action = action

        now = time.time()
        # Executing action and appending state-action pair to path
        self.game.execute_move(best_action)

        self.time_4 += time.time() - now
        # self.time_4 += time.time() - now
        return state, best_action

    # Returning the matching state from a set of states
    def _return_one_state(self, states):
        return states[0]

    # Calculating the value for a state-action pair
    def PUCT(self, args, parent_visits):
        exploration = math.log((1 + parent_visits + self.c_puct) / self.c_puct) + self.c_init
        Q = args[2]
        U = exploration * args[3] * math.sqrt(parent_visits) / (1 + args[0])
        return Q + U

    # Evaluate a state using the evaluation algorithm
    def _evaluate(self, state):
        state = state.reshape(1, 8, 8, 2)
        prior_prob = self.eval.predict(state)
        # print(self.game.get_moves())
        # print([prior_prob[0][0, pos[0] * 8 + pos[1]] for pos in self.game.get_moves()])
        # TODO: must be generalized by adding a set method for mapping NN probabilities to actions
        prior_porb_norm = np.array([prior_prob[0][0, pos[0] * 8 + pos[1]] for pos in self.game.get_moves()])
        prior_porb_norm = prior_porb_norm / sum(prior_porb_norm)
        prior_porb_norm = prior_porb_norm.reshape(prior_porb_norm.shape[0], 1)

        if len(self.state_visits) == 0 and self.dirichlet_noise:
            noise = np.random.dirichlet(np.array([self.alpha for _ in range(len(self.game.get_moves()))]), (1))
            noise = noise.reshape(noise.shape[1], 1)
            prior_porb_norm = (prior_porb_norm + noise) / 2
        return 0, {str(act): prior_porb_norm[num] for num, act in enumerate(self.game.get_moves())}

    # Initializing a new leaf node
    def _expansion(self, state, priors):
        # Finding all actions
        actions = self.game.get_moves()
        self.pos_move_dict[state] = actions
        self.state_visits[state] = 0

        # Initializing each state action pair
        for action in actions:
            # print(str(action))
            # print(str(state))
            self.search_dict[str(state) + '-' + str(action)] = [0, 0, 0, priors[str(action)]]

    def get_search_probabilities(self, state):
        prob = np.array([0 for _ in range(64)], dtype=float).reshape(self.policy_output_dim)
        total_visits = self.state_visits[state]
        for action in self.game.get_moves():
            prob[action[0] * 8 + action[1]] = self.search_dict[str(state) + '-' + str(action)][0] / total_visits
        return prob

    # Updating the tree
    def _backward_pass(self, state, state_action, value):
        state_action_values = self.search_dict.get(state_action)
        self.search_dict[state_action] = [state_action_values[0] + 1,
                                          state_action_values[1] + value,
                                          (state_action_values[1] + value) / (state_action_values[0] + 1),
                                          state_action_values[3]]
        self.state_visits[state] = self.state_visits.get(state) + 1
        # self.game.undo_move()

# game = Gamelogic.Othello()
# tree = MCTS()
# tree.set_game(game)
#
# for _ in range(8000):
#     print(tree.tree_children)
#     tree.search()
#     print(tree.time_1, tree.time_2, tree.time_3, tree.time_4)
