import gym
import numpy as np
import keras
from keras import layers
import tensorflow as tf
import random
import matplotlib.pyplot as plt

def plot_res(values, title=''):
    ''' Plot the reward curve and histogram of results over time.'''
    # Update the window after each episode

    # Define the figure
    f, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    f.suptitle(title)
    ax[0].plot(values, label='score per run')
    ax[0].axhline(195, c='red', ls='--', label='goal')
    ax[0].set_xlabel('Episodes')
    ax[0].set_ylabel('Reward')
    x = range(len(values))
    ax[0].legend()
    # Calculate the trend
    try:
        z = np.polyfit(x, values, 1)
        p = np.poly1d(z)
        ax[0].plot(x, p(x), "--", label='trend')
    except:
        print('')

    # Plot the histogram of results
    ax[1].hist(values[-50:])
    ax[1].axvline(195, c='red', label='goal')
    ax[1].set_xlabel('Scores per Last 50 Episodes')
    ax[1].set_ylabel('Frequency')
    ax[1].legend()
    plt.show()


env = gym.make('CartPole-v1')

class q_function:
# dnn approximater of physical function regarding pole balancing
# input = cart position, cart velocity, pole angle, pole angular velocity
# output = expected value of each action
    def __init__(self):

        state_dim = 4
        action_dim = 2
        hidden_dim = 64
        lr = 0.05
        self.practicemodel = keras.models.Sequential([
            keras.layers.Dense(hidden_dim,input_dim=4, activation=layers.LeakyReLU(alpha=lr)),
            keras.layers.Dense(hidden_dim, activation=layers.LeakyReLU(alpha=lr)),
            keras.layers.Dense(2,activation='sigmoid')
        ])

        self.practicemodel.compile(optimizer='adam',loss='mean_squared_error',learning_rate=lr)
        # # deep neural network settings
        # # 4 inputs, 2 intermediate layers with leakyrelu, final output layer
        # # self.practicemodel = keras.models.Sequential()
        # # self.practicemodel.add(layers.Dense(16,activation='relu',input_shape=(4,)))
        # # self.practicemodel.add(layers.Dense(64,activation='relu'))
        # # self.practicemodel.add(layers.Dense(2))
        #
        # # inputs = keras.layers.Input(shape=(4,))
        # layer1 = layers.Dense(hidden_dim, input_shape=(4,)activation=layers.LeakyReLU(alpha=lr))
        # # layer1 = layers.Dense(hidden_dim, activation=layers.LeakyReLU(alpha=lr))(inputs)
        # # layer2 = layers.Dense(hidden_dim*2, activation=layers.LeakyReLU(alpha=lr))(layer1)
        # outputs = layers.Dense(units=2, activation='sigmoid', name="predictions")(layer1)
        # # outputs = layers.Dense(units=2, activation='sigmoid',name="predictions")(layer2)
        #
        # # compile model
        # self.practicemodel = keras.Model(inputs=inputs, outputs=outputs)
        # # self.practicemodel.compile(loss=keras.losses.MeanSquaredError(), optimizer='adam')
        #
        # self.loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        # self.optimizer = keras.optimizers.Adam(learning_rate = lr)

    def model_summary(self):
        self.practicemodel.summary()

    def update(self,state,reward):
        # Update the weights of the network given a training sample

        self.practicemodel.fit(state,)
        # #alpha = 0.2;
        # #self.practicemodel(state)+alpha*()
        # with tf.GradientTape() as tape:
        #     y_pred = self.practicemodel(state,training=True)
        #     loss = self.loss_fn(reward,y_pred)
        # grads = tape.gradient(loss,self.practicemodel.trainable_weights)
        # self.optimizer.apply_gradients(grads,self.practicemodel.trainable_weights)

    def predict(self,state):
        return self.practicemodel(state)


    def action_selection(self,state,explorationrate):

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
firsttry = q_function()

for episode in range(episodes):
    state = env.reset()
    done = False
    total = 0
    epsilon = 0.3
    decay = 0.99
    while not done:
        action = firsttry.action_selection(state,epsilon)

        next_state,reward,done,_ = env.step(action)
        next_state = np.reshape(next_state,[1,4])

        total += reward
        memory.append((state,action,next_state,reward,done))
        q_values = firsttry.predict(state.reshape(1,4)).tolist()

        if done:
            q_values[action] = reward
            firsttry.update(state,q_values)

        state = next_state
        epsilon = max(epsilon*decay,0.01)
    final.append(total)
    plot_res(final,'firsttry')
    plt.show()
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
