import gym
import numpy as np
import keras
from keras import layers
env = gym.make('CartPole-v1')


def action_selection():
    if random.random() < explorationrate:
        action = env.action_space.sample()
    else:
        action = np.argmax(q)


class q_function(state,action):
# dnn approximater of physical function regarding pole balancing
# input = cart position, cart velocity, pole angle, pole angular velocity
# output = expected value of each action
    def __init__(self):

        state_dim = 4
        action_dim = 2
        hidden_dim = 64
        lr = 0.05

        inputs = keras.Input(shape=(4,))
        layer1 = layers.Dense(hidden_dim, activation=layers.LeakyReLU(alpha=lr))(inputs)
        layer2 = layers.Dense(hidden_dim*2, activation=layers.LeakyReLU(alpha=lr))(layer1)
        outputs = layers.Dense(2, name="predictions")(layer2)
        self.practicemodel = keras.Model(inputs=inputs, outputs=outputs)

        self.loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.optimizer = keras.optimizers.Adam(learning_rate = 0.05)
        #practicemodel.compile(loss=keras.losses.MeanSquaredError(), optimizer='adam')

    def update(self,state,action,reward):
        alpha = 0.2;
        q_function(state,action)+alpha*()

        y_pred = self.practicemodel.predict(state)
        loss = self.loss_fn()
    def predict(self,state):



iteration = 10000

score = []
rewardseq = np.zeros((1))

env.reset()
while not done:
    action = action_selection()
    next_state, reward, done, _ = env.step(action)
    rewardseq = np.append((rewardseq),reward)



score = np.sum(rewardseq)


# agent - environment
# action - state - reward
# current state is open: markov decision
# action -> environment -> reward, state -> action
# action is selected by calculating the most expected reward of a state
# Expected reward is learned, b/c complex pattern (q-learning)
# but how? by comparing expected return and actual return and modifying weights
# accordingly, the expectation becomes closer and closer to the actual value
#

# trying to do a q-learning on a cart pole problem with dnn in keras package
# currently done: setting up the model
# next issue: setting up update method
