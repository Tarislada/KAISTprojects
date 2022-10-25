import gym
import numpy as np
import keras
from keras import layers
import tensorflow as tf
import random


env = gym.make('CartPole-v1')

class q_function():
# dnn approximater of physical function regarding pole balancing
# input = cart position, cart velocity, pole angle, pole angular velocity
# output = expected value of each action
    def __init__(self):

        state_dim = 4
        action_dim = 2
        hidden_dim = 64
        lr = 0.05
        # deep neural network settings
        # 4 inputs, 2 intermediate layers with leakyrelu, final output layer
        inputs = keras.Input(shape=(4,))
        layer1 = layers.Dense(hidden_dim, activation=layers.LeakyReLU(alpha=lr))(inputs)
        layer2 = layers.Dense(hidden_dim*2, activation=layers.LeakyReLU(alpha=lr))(layer1)
        outputs = layers.Dense(2, name="predictions")(layer2)
        # compile model
        self.practicemodel = keras.Model(inputs=inputs, outputs=outputs)

        self.loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.optimizer = keras.optimizers.Adam(learning_rate = lr)
        #practicemodel.compile(loss=keras.losses.MeanSquaredError(), optimizer='adam')

    def update(self,state,reward):
        # Update the weights of the network given a training sample
        #alpha = 0.2;
        #self.practicemodel(state)+alpha*()
        with tf.GradientTape() as tape:
            y_pred = self.practicemodel(state,training=True)
            loss = self.loss_fn(reward,y_pred)
        grads = tape.gradient(loss,self.practicemodel.trainable_weights)
        self.optimizer.apply_gradients(grads,self.practicemodel.trainable_weights)

    def predict(self,state):
        return self.practicemodel(state)


    def action_selection(self,state):
        explorationrate = 0.3
        if random.random() < explorationrate:
            action = env.action_space.sample()
        else:
            q_value = self.predict(state)
            action = np.argmax(q_value)
        return action


# iteration = 10000
#
# score = []
# rewardseq = np.zeros((1))
#
# env.reset()
# while not done:
#     action = action_selection()
#     next_state, reward, done, _ = env.step(action)
#     rewardseq = np.append((rewardseq),reward)
#
#
#
# score = np.sum(rewardseq)
episodes = 150
final = []
memory = []
for episode in range(episodes):
    state = env.reset()
    done = False
    total = 0
    firsttry = q_function()

    while not done:
        action = firsttry.actionselection(state)

        next_state,reward,done,_ = env.step(action)

        total += reward
        memory.append((state,action,next_state,reward,done))
        q_values = firsttry.predict(state).tolist()

        if done:
            q_values[action] = reward
            firsttry.update(state,q_values)

state = next_state

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
