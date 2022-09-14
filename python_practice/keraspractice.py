import keras
from keras import layers
import tensorflow as tf
import gym

env = gym.make('CartPole-v1')

state_dim = 4
action_dim = 2
hidden_dim = 64
lr = 0.05

batch_size = 64
##
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size=64)

#practicemodel = keras.Sequential(
#    [
#        layers.Dense(hidden_dim,input_shape=(4,),activation=layers.LeakyReLU(alpha=lr)),
#        layers.Dense(hidden_dim*2,activation=layers.LeakyReLU(alpha=lr)),
#        layers.Dense(1,activation=layers.LeakyReLU(alpha=lr))
#    ]
#)

inputs = keras.Input(shape=(4,),name='digits')
layer1 = layers.Dense(hidden_dim,activation=layers.LeakyReLU(alpha=lr))(inputs)
layer2 = layers.Dense(hidden_dim,activation=layers.LeakyReLU(alpha=lr))(layer1)
outputs = layers.Dense(2,name="predictions")(layer2)
practicemodel = keras.Model(inputs=inputs,outputs=outputs)


practicemodel.compile(loss=keras.losses.MeanSquaredError(),optimizer='adam')


#future_rewards = practicemodel.predict(next_state)

#updated_q_values = reward + gamma * tf.reduce_max(future_rewards, axis=1)

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)