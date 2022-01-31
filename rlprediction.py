# Date: 7/16/2020
#Author: Bikram Karki
# Procedure:This file contains logic to import and convert gps coordinates in  gpx format into utm coordinates

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import threading
import gym
import multiprocessing
from queue import Queue
import argparse
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import gpxpy
import pandas as pd
import utm
import numpy as np
import matplotlib.pyplot as plt
import math
import copy

parser = argparse.ArgumentParser(description='Run A3C algorithm on the goal '
                                             'gps localization.')
parser.add_argument('--algorithm', default='a3c', type=str,
                    help='Choose between \'a3c\' and \'random\'.')
parser.add_argument('--train', dest='train', action='store_true',
                    help='Train our model.')
parser.add_argument('--lr', default=0.05,
                    help='Learning rate for the shared optimizer.')
parser.add_argument('--update-freq', default=31, type=int,
                    help='How often to update the global model.')
parser.add_argument('--max-eps', default=30, type=int,
                    help='Global maximum number of episodes to run.')
parser.add_argument('--gamma', default=0.9,
                    help='Discount factor of rewards.')
parser.add_argument('--save-dir', default='/tmp/model_cs/', type=str,
                    help='Directory in which you desire to save the model.')
args = parser.parse_args()


GPS_NOISE = np.diag([1612, 2]) ** 2
prediction_z = []
action_taken_z = []
regularization_param = 5.0 * pow(10,-16)
confidence_store = []
data_count_store = []
reward_store = []
episode_reward_save = []
episodic_counter_save = []



def record(episode,
           episode_reward,
           worker_idx,
           global_ep_reward,
           result_queue,
           total_loss,
           num_steps):
    if global_ep_reward == 0:
        global_ep_reward = episode_reward
    else:
        global_ep_reward = global_ep_reward * 0.99 + episode_reward * 0.01
    print(
        f"Episode: {episode} | "
        f"Moving Average Reward: {int(global_ep_reward)} | "
        f"Episode Reward: {int(episode_reward)} | "
        f"Loss: {int(total_loss / float(num_steps) * 1000) / 1000} | "
        f"Steps: {num_steps} | "
        f"Worker: {worker_idx}"
    )
    result_queue.put(global_ep_reward)
    return global_ep_reward

"Import GPS Data"

def import_gps():
    with open('3296026.gpx') as fh:
        gpx_file = gpxpy.parse(fh)
        segment = gpx_file.tracks[0].segments[0]
        coords = pd.DataFrame([  #imported data is saved in this variable
        {'lat': p.latitude,
        'lon': p.longitude,
        } for p in segment.points])
    return coords

# convert long/lat into UTM coordinates


def convertcartesian(coords):
    numpy_gps_coord = np.zeros((len(coords.lat - 1),2))
    noisy_gps_utm_coord = np.zeros((len(coords.lat - 1),2))
    all_latitudes  = []
    all_longitudes = []

    for i in range(len(coords.lat) - 1 ):
        utm_conversion = utm.from_latlon(coords.lat[i], coords.lon[i])
        numpy_gps_coord[i][0] = utm_conversion[0]
        numpy_gps_coord[i][1] = utm_conversion[1]
        noisy_gps_utm_coord[i][0] = utm_conversion[0] + (5 * np.random.randn())    # get noisy gaussian lat
        noisy_gps_utm_coord[i][1] = utm_conversion[1] + (5 * np.random.randn())    # get noisy gaussian lon
        all_latitudes.append(utm_conversion[0])
        all_longitudes.append(utm_conversion[1])
    return numpy_gps_coord,noisy_gps_utm_coord,all_latitudes, all_longitudes


def get_action_space():  # get all set of action space
    count_action = 0
    scalling_mat_S = [-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Scaling matrix
    unit_op_latitude_ux = 1  # unit operation on latitude
    unit_op_longitude_uy = 1  # unit operation on longitude
    action_space_lat = []
    action_space_lon = []
    action_space = np.zeros(shape=(441, 2))

    for x in scalling_mat_S:                                 # Lx and Ly operation on lat and long
        action_space_lat.append(x * unit_op_latitude_ux)
        action_space_lon.append(x * unit_op_longitude_uy)

    for x in range(len(action_space_lat)):                   # get total 441 action sets
        for y in range(len(action_space_lon)):
            action_space[count_action] = [action_space_lat[x],action_space_lon[y]]
            count_action = count_action + 1
    return action_space


def reward_function(predition_confidence_measure,predicted_gps_points, groundtruth_gps_points):
    predicted_gps_lat = predicted_gps_points[0]
    predicted_gps_lon = predicted_gps_points[1]
    groundtruth_gps_lan = groundtruth_gps_points[0]
    groundtruth_gps_lon = groundtruth_gps_points[1]
    reward = (groundtruth_gps_lan - predicted_gps_lat) ** 2 + (groundtruth_gps_lon -  predicted_gps_lon) ** 2  # Eucledean distance based reward
    #reward =  predition_confidence_measure + regularization_param * reward * -1   # replace -1 when returning reward after using this line to calculate reward
    return -reward

def error_cal(predicted_gps_points, groundtruth_gps_points):
    predicted_gps_lat = predicted_gps_points[0]
    predicted_gps_lon = predicted_gps_points[1]
    groundtruth_gps_lan = groundtruth_gps_points[0]
    groundtruth_gps_lon = groundtruth_gps_points[1]
    error = math.hypot(groundtruth_gps_lan - predicted_gps_lat, groundtruth_gps_lon -  predicted_gps_lon)  # Eucledean distance based reward
    return error


#compute eigen values
def compute_eigen_values(p):
    eigenvalues = np.linalg.eigvals(p)
    # print(eigenvalues)
    return eigenvalues

# confidence_measure
def calculate_confidence_measure(a,b):
    if a < 0.0000001 or b < 0.0000001:
        confidence_measure = a + b
    else:
        confidence_measure = np.pi * a * b
    #print(confidence_measure)
    return confidence_measure


class ActorCriticModel(keras.Model):
    def __init__(self, state_size, action_size):
        super(ActorCriticModel, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.densea = layers.Dense(100, activation='relu')
        self.dense1 = layers.Dense(100, activation='relu')
        self.policy_logits = layers.Dense(action_size)
        self.dense2 = layers.Dense(100, activation='relu')
        self.values = layers.Dense(1)

    def call(self, inputs):
        # Forward pass
        x = self.densea(inputs)
        y = self.dense1(x)
        logits = self.policy_logits(y)
        v1 = self.dense2(y)
        values = self.values(v1)
        return logits, values


class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []

    def store(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []

    def delete(self,datarow):
        del self.states[datarow]
        del self.rewards[datarow]
        del self.actions[datarow]


class StoreResults:
    def __init__(self):
        self.data = []
        self.confidence_m = []
        self.rewards = []

    def store(self, datanum,confidence,reward):
        self.data = datanum
        self.confidence_m = confidence
        self.rewards = reward

    def clear(self):
        self.data = []
        self.confidence_m = []
        self.rewards = []


class MasterAgent():
    def __init__(self, gps_utm_coord, noisy_gps_utm_coord,action_space,observation_test):
        self.goal_name = 'cs20m'
        save_dir = args.save_dir
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        self.state_size = 2
        self.action_size = 441
        self.opt = tf.optimizers.Adam(args.lr)
        self.gps_utm_coord = gps_utm_coord
        self.noisy_gps_utm_coord = noisy_gps_utm_coord
        self.action_space = action_space
        self.observation_test = observation_test
        print(self.state_size, self.action_size)

        self.global_model = ActorCriticModel(self.state_size, self.action_size)  # global network
        self.global_model(tf.convert_to_tensor(np.random.random((1,self.state_size)), dtype=tf.float32))

    def train(self):

        ep_reward = 0
        done = False
        time_count = 0
        count_episode = 0
        hello = []

        mem = Memory()
        total_loss = 0

        for j in range(0,len(self.noisy_gps_utm_coord)):
            ep_steps = 0
            time_count += 1
            current_state_qt = noisy_gps_utm_coord[j]
            #print("current state", current_state_qt)
            Worker.global_episode = 0
            action_train_taken = 0

            res_queue = Queue()
            workers = [Worker(self.state_size,
                            self.action_size,
                            self.global_model,
                            self.noisy_gps_utm_coord,
                            self.gps_utm_coord,
                            self.action_space,
                            current_state_qt,
                            self.opt, res_queue,
                            i, goal_name=self.goal_name,
                              save_dir=self.save_dir) for i in range(multiprocessing.cpu_count())]

            for i, worker in enumerate(workers):
                #print("Starting worker {}".format(i))
                worker.start()

            [w.join() for w in workers]

            prediction_train = np.mean(prediction_z,axis=0)
            calculate_action = prediction_train - current_state_qt
            round_action = np.round(calculate_action, 1)
            action_distance = 100000
            for y in range(0,self.action_size):
                temp_action_distance = math.hypot(round_action[0] - self.action_space[y][0], round_action[1] - self.action_space[y][1])
                if action_distance > temp_action_distance:
                    action_distance = temp_action_distance
                    action_train_taken = y

            #print("prediction",prediction_train)

            prediction_covariance = np.cov(prediction_z, rowvar=False)
            prediction_eigen_val = compute_eigen_values(prediction_covariance)
            prediction_confidence_measure = calculate_confidence_measure(prediction_eigen_val[0], prediction_eigen_val[1])

            reward = reward_function(prediction_confidence_measure,prediction_train, gps_utm_coord[j])
            episodic_counter_save.append((prediction_train))

            #print(j,prediction_confidence_measure,reward)
            #print(j)
            data_count_store.append(j)
            confidence_store.append(prediction_confidence_measure)
            reward_store.append(reward)

            ep_reward += reward

            error = error_cal(prediction_train, gps_utm_coord[j])
            hello.append(error)
            print("action : ",j,action_train_taken, error)

            mem.store(current_state_qt,action_train_taken,reward)
            if(len(mem.states) == 6):
                mem.delete(0)

            if ( j == (len(self.noisy_gps_utm_coord))):
                done = True

            if (time_count == 5) or done:
                # calculate gradient wrt to local model. We do so by tracking the variables
                # involved in computing the loss by using tf.
                if done:
                    reward_sum = 0
                else:
                    reward_sum = self.global_model(
                        tf.convert_to_tensor(prediction_train[None, :],
                                            dtype=tf.float32))[-1].numpy()[0]

                with tf.GradientTape() as tape:
                    total_loss = self.compute_loss(done,
                                                   prediction_train,
                                                   mem,
                                                   reward_sum,
                                                   args.gamma
                                                   )

                #self.ep_loss += total_loss
                #print("loss : ", total_loss.numpy())
                #hello.append(total_loss.numpy())
                # Calculate local gradients
                grads = tape.gradient(total_loss, self.global_model.trainable_weights)
                # Push local gradients to global model
                self.opt.apply_gradients(zip(grads,
                                             self.global_model.trainable_weights))
                # Update local model with new weights
                self.global_model.set_weights(self.global_model.get_weights())

                #mem.clear()
                time_count = 0
                count_episode = count_episode + 1
                episode_reward_save.append(ep_reward)
                ep_reward = 0

            prediction_z.clear()
            action_taken_z.clear()
            #print("byeee")

        #plt.plot(data_count_store, confidence_store)

        f = open("/tmp/model_cs/cs20r.csv", "a")
        for i in range(len(reward_store) -1):
            f.write(str(i) + "," + str(reward_store[i]) + "\n")
        f.close()

        f = open("/tmp/model_cs/cs20d.csv", "a")
        for i in range(len(episodic_counter_save) -1):
            error = error_cal(episodic_counter_save[i],gps_utm_coord[i] )
            f.write(str(i) + "," + str(error) + "," + str(episodic_counter_save[i]) + "," +  str(noisy_gps_utm_coord[i]) + "," + str(gps_utm_coord[i]) + "\n")
        f.close()


        hello.pop()
        reward_store.pop()
        plt.plot(hello)
        plt.xlabel('loop step')
        plt.ylabel('error')
        plt.title(' error measure vs each step')
        plt.legend([" error measure"])
        plt.show()

    def compute_loss(self,
                     done,
                     new_state,
                     memory,
                    reward_sum,
                     gamma=0.9):


        # Get discounted rewards
        discounted_rewards = []
        for reward in memory.rewards[::-1]:  # reverse buffer r
            reward_sum = reward + gamma * reward_sum
            discounted_rewards.append(reward_sum)
        discounted_rewards.reverse()

        logits, values = self.global_model(
            tf.convert_to_tensor(np.vstack(memory.states),
                                 dtype=tf.float32))
        # Get our advantages
        advantage = tf.convert_to_tensor(np.array(discounted_rewards)[:, None],
                                         dtype=tf.float32) - values
        # Value loss
        value_loss = advantage ** 2

        # Calculate our policy loss
        policy = tf.nn.softmax(logits)
        entropy = tf.nn.softmax_cross_entropy_with_logits(labels=policy, logits=logits)

        policy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=memory.actions,
                                                                     logits=logits)
        policy_loss *= tf.stop_gradient(advantage)
        policy_loss -= 0.01 * entropy
        total_loss = tf.reduce_mean((0.5 * value_loss + policy_loss))
        return total_loss


class Worker(threading.Thread):
    global_episode = 0
    # Moving average reward
    global_moving_average_reward = 0
    best_score = 0
    save_lock = threading.Lock()
    prediction = []

    def __init__(self,
                 state_size,
                 action_size,
                 global_model,
                 noisy_gps_utm_coord,
                 gps_utm_coord,
                 action_space,
                 current_state_qt,
                 opt,
                 result_queue,
                 idx,
                 goal_name='cs20l',
                 save_dir='/tmp/model_cs/cs19m'):
        super(Worker,self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.result_queue = result_queue
        self.global_model = global_model
        self.noisy_gps_utm_coord = noisy_gps_utm_coord
        self.gps_utm_coord = gps_utm_coord
        self.action_space = action_space
        self.current_state_qt = current_state_qt
        self.opt = opt
        self.local_model = global_model
        self.worker_idx = idx
        self.goal_name = goal_name
        self.save_dir = save_dir
        self.ep_loss = 0.0

    def run(self):
        total_step = 0
        mem = Memory()

        while Worker.global_episode < args.max_eps:   # total episode to
            Worker.global_episode += 1
            current_state = self.current_state_qt
            logits, _ = self.local_model(
                tf.convert_to_tensor(current_state[None, :],
                                     dtype=tf.float32))

            probs = tf.nn.softmax(logits)
            action = np.argmax(probs)
            new_state = current_state + action_space[action]
            prediction_z.append([new_state[0], new_state[1]])
            action_taken_z.append((action))

        self.result_queue.put(None)


if __name__ == '__main__':

    state = [] # current reported gps points at time t

    cords = import_gps()                                                   # import gpx data
    gps_utm_coord,noisy_gps_utm_coord,all_latitudes,all_longitudes = convertcartesian(cords)   # convert gpx data into utm coordinates
    observation_test = np.zeros(shape=(50, 2))
    action_space = get_action_space()                                      # get action space

    master = MasterAgent(gps_utm_coord,noisy_gps_utm_coord,action_space,observation_test)
    master.train()

    #plot gps points on graph
    plt.plot(all_latitudes, all_longitudes)
    plt.xlabel('latitude')
    plt.ylabel('longitude')
    plt.title('  Plot of gps points ')
    plt.legend([" gps trajectory"])
    #plt.show()


