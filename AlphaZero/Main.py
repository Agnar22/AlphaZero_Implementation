import ResNet
import MCTS
import time
import numpy as np
from Othello import Gamerendering
from Othello import Gamelogic
from keras.optimizers import Adam

# Setting up game, ResNet and MCTS
game = Gamelogic.Othello()
height, width, depth = game.get_board().shape
agent = ResNet.ResNet.build(height, width, depth, 256)
tree = MCTS.MCTS()
tree.set_evaluation(agent)
tree.set_game(game)
print(agent.summary())

now = time.time()

for x in range(800):
    tree.search()
    print(tree.tree_children)

print(time.time() - now)
print(agent.predict(game.get_board().reshape(1, 8, 8, 2)))
print(tree.get_search_probabilities(game.get_states()[0]))
y_targ = tree.get_search_probabilities(game.get_states()[0])
print(y_targ.shape)
y_targ = y_targ.reshape(64)
x = np.array([game.get_board().reshape(8, 8, 2) for _ in range(100)])
agent.compile(loss=['categorical_crossentropy', 'mean_squared_error'], optimizer=Adam(lr=0.002, epsilon=10E-8))
agent.fit(x, [np.array([y_targ for _ in range(100)]),
              np.array([[1] for _ in range(100)])],
          batch_size=32, epochs=15)
print(agent.predict(game.get_board().reshape(1, 8, 8, 2)))
