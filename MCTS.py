# TODO: must add "negamax backprop" and possiblilty to do two (or more) consecutive moves


import math


class MCTS:
    def __init__(self):
        # #######DICTIONARIES#######
        self.search_dict = {}  # [visit cnt, sum. act. val., avg. act val, prior prob.] for each state-act
        self.pos_move_dict = {}  # Possible actions for each state
        self.state_visits = {}  # total visits for each state

        # #######PARAMETERS#######
        self.c_puct = 4  # Used for exploration (larger=>less long term exploration)
        self.c_init = 1  # Used for exploration (larger=>more exploration)

    # Setting the game the MCTS will be used on
    def set_game(self, game):
        self.game = game

    # Setting the evaluation algorithm used by the MCTS
    def set_evaluation(self, eval):
        self.eval = eval

    # Returning a dictionary with action as key and visit number as value
    def get_action_numbers(self):
        state = self._return_one_state(self.game.get_states)
        actions = self.pos_move_dict.get(state)
        return {str(action): self.search_dict.get(action)[0] for action in actions}

    # Number of consecutive searches in the tree
    def search_series(self, number):
        for _ in range(number):
            self.search()

    # The main method for the MCTS-tree search
    def search(self):
        # Selection
        new_state, history = self._selection()

        # Evaluation
        value, priors = self._evaluate(new_state)

        # Expansion
        self._expansion(new_state, priors)

        # Backward pass
        self._backward_pass(history, value)

    # Selecting the path from the root node to the leaf node and returning the new state
    def _selection(self):
        path = []  # The state-action pairs chosen

        # Executing actions until leaf node reached
        while True:
            # Current state of the game
            state = self._return_one_state(self.game.get_states())

            if state not in self.pos_move_dict:  # New state encountered
                return state, path
            elif self.game.is_final():  # Game is over
                return state, path

            best_action = ''
            best_value = None

            # Iterating through all actions to find the best
            for action in self.pos_move_dict.get(state):
                state_action_value = self.PUCT(self.search_dict.get(str(state) + '-' + str(action)),
                                               self.state_visits.get(state))
                if state_action_value > best_value or best_value is None:  # If new best action is found
                    best_value = state_action_value
                    best_action = action

            # Executing action and appending state-action pair to path
            self.game.execute_action(best_action)
            path.append((state, best_action))

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
        return -1, {}

    # Initializing a new leaf node
    def _expansion(self, state, priors):
        # Finding all actions
        actions = self.game.get_moves()
        self.pos_move_dict[state] = actions
        self.state_visits[state] = 1

        # Initializing each state action pair
        for action in actions:
            self.search_dict[str(state) + '-' + str(action)] = [0, 0, 0, priors[action]]

    # Updating the tree
    def _backward_pass(self, path, value):
        for step in path:
            state_action = str(step[0]) + '-' + str(step([1]))
            state_action_values = self.search_dict.get(state_action)
            self.search_dict[state_action] = [state_action_values[0] + 1,
                                              state_action_values[1] + value,
                                              (state_action[1] + value) / (state_action_values[0] + 1),
                                              state_action_values[3]]
            self.state_visits[str(step[0])] = self.state_visits.get(str(step[0])) + 1
            self.game.undo_move()
