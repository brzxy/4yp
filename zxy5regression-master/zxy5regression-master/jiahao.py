import gym
import numpy as np
import os
from pyglet.window import key
import matplotlib.pyplot as plt
import pickle, os, gzip
from datetime import datetime
from PIL import Image
from gym.envs.box2d.car_racing import CarRacing
import utils
from skimage.color import rgb2gray
import torch


class User_Input:
    # car
    steering = 0
    gas = 0
    breaking = 0
    action = np.zeros(3, dtype=np.float32)
    # simulation
    escape = False
    record = False
    save = False
    reset = False
    # key press
    up = False
    down = False
    right = False
    left = False

    # setter from key press
    def on_key_press(self, k, mod):
        # car
        if k == key.UP:
            self.up = True
            return
        if k == key.LEFT:
            self.left = True
            return
        if k == key.RIGHT:
            self.right = True
            return
        if k == key.DOWN:
            self.down = True
            return

        # simulation
        if k == key.SPACE:
            self.reset = True
            return
        if k == key.ESCAPE:
            self.escape = True
            return
        if k == key.R and self.record == False:
            self.record = True
            print("start recording")
            return
        if k == key.R and self.record == True:
            self.record = False
            self.save = True
            print("paused recording, start saving")
            return

    def on_key_release(self, k, mod):
        # car
        if k == key.UP:
            self.up = False
            return
        if k == key.LEFT:
            self.left = False
            return
        if k == key.RIGHT:
            self.right = False
            return
        if k == key.DOWN:
            self.down = False
            return

    def check_arrow_keys(self):

        if (self.left): self.steering -= 0.02
        if (self.right): self.steering += 0.02
        if (self.up): self.gas += 0.05
        if (self.down): self.breaking += 0.02
        if ((not self.left) and (not self.right)): self.steering = 0
        if (not self.up): self.gas = 0
        if (not self.down): self.breaking = 0

        return

    # setter from action
    def on_save(self):
        self.save = False

    # getter
    def get_action(self):
        self.action[0] = self.steering
        self.action[1] = self.gas
        self.action[2] = self.breaking
        return self.action

    def escape_pressed(self):
        return self.escape

    def record_pressed(self):
        return self.record

    def save_pressed(self):
        return self.save

    def reset_pressed(self):
        return self.reset


class Data:
    data = {"state": [], "info": [], "action": []}
    dir_folder = "./data2"

    def record(self, current_state, info, action):
        self.data["state"].append(current_state)
        self.data["info"].append(info)
        self.data["action"].append(action)

    def vstack(self, arr):
        stack = np.array(arr[0], dtype=np.float32)
        for i in range(1, len(arr)):
            stack = np.vstack((stack, arr[i]))
        return stack

    def read_data(self):
        """Reads the states and actions recorded in all files inside the given directory"""

        # directory_path = os.path.join(os.getcwd(), "data2")
        directory_path = "./data2"
        print("Reading data from..." + directory_path)

        #initialise the list
        state = []
        action = []
        info = []

        for filename in os.listdir(directory_path):
            if filename.endswith(".pkl.gzip"):
                with gzip.open(os.path.join(directory_path, filename),
                               'rb') as f:
                    print(filename + " loaded")
                    data = pickle.load(f)
                    #append array into list
                    state.append(data["state"])
                    action.append(data["action"])
                    info.append(data["info"])

        s = self.vstack(state)
        a = self.vstack(action)
        i = self.vstack(info)

        print("All files added")
        print(s.shape)
        print(a.shape)
        print(i.shape)
        return s, a, i

    def save(self):

        # create folder if doesn't already exist
        if not os.path.exists(self.dir_folder):
            os.mkdir(self.dir_folder)

        # get file name
        file_name = str(
            datetime.now().strftime("%Y_%m_%d_%H-%M-%S")) + ".pkl.gzip"
        dir_file = os.path.join(self.dir_folder, file_name)

        # save
        self.data["state"] = Data.preprocess_state(self.data["state"])
        print("image processed")
        f = gzip.open(dir_file, 'wb')
        pickle.dump(self.data, f)
        print("saved data to " + dir_file)

        # reset data
        self.data = {"state": [], "info": [], "action": []}

    def plot_safety(self, info, i):
        info = np.array(info)
        speed = info[:, 0]
        gyro = info[:, 6]

        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        # Plot first graph in first subplot
        ax1.plot(i, speed)
        ax1.set_title('Speed over time')

        # Plot second graph in second subplot
        ax2.plot(i, gyro)
        ax2.set_title('Angular Velocity over time')

        # Add labels and title for the entire figure
        fig.suptitle('Safety')
        fig.tight_layout()

        # Display the plot
        plt.show()


class Simulation:
    #env = gym.make('CarRacing-v0').unwrapped
    env = CarRacing()
    user_input = User_Input()
    current_state = np.ndarray((96, 96, 3))
    data = Data()
    step = 0

    def __init__(self):
        # initialize window
        self.current_state = self.env.reset()
        # assign key action
        self.env.viewer.window.on_key_press = self.user_input.on_key_press
        self.env.viewer.window.on_key_release = self.user_input.on_key_release

    def run_simulation(self):
        while True:

            # act
            next_state, reward, done, info = self.env.step(
                self.user_input.get_action())

            # render
            isopen = self.env.render()
            self.step += 1
            if self.step % 1000 == 0:
                print(f'step number has been {self.step}')

            # update control
            self.user_input.check_arrow_keys()

            # ---------- ACTIONS ----------
            # close window
            if (self.user_input.escape_pressed()):
                self.env.close()
                break

            # record data
            if (self.user_input.record_pressed()):
                self.data.record(self.current_state, self.env.read_info(),
                                 self.user_input.get_action())
                #print(self.env.read_info())
            if (self.user_input.save_pressed()):
                self.data.save()
                self.user_input.on_save()

            # --------- NEXT LOOP ---------
            self.current_state = next_state
            if done or self.user_input.reset == True or isopen == False:
                self.env.reset()
                self.user_input.reset = False

    def test_simulation(self, model):
        all_info = []
        frames = []
        while True:
            # Define a state to test the model on
            state = preprocess_state(self.current_state)
            info = self.env.read_info()

            all_info.append(info)
            frames.append(self.step)

            # Wrap the state and info in Tensors and add a batch dimension
            state = torch.tensor(state).unsqueeze(0).unsqueeze(0).to(
                torch.float32)
            info = torch.tensor(info).unsqueeze(0).to(torch.float32)

            # Pass the state and info through the model and get the output action
            action = model((state, info))
            # Convert tensor to tuple
            action = tuple(action.squeeze().tolist())

            # if self.step < 200:
            #     action = list(action)
            #     action[1] = 1
            #     action= tuple(action)

            #print(f"output action is: {action}")

            # act
            next_state, reward, done, i = self.env.step(action)

            # render
            isopen = self.env.render()
            self.step += 1
            if self.step % 1000 == 0:
                print(f'step number has been {self.step}')

            # update control
            self.user_input.check_arrow_keys()

            # ---------- ACTIONS ----------
            # close window
            if (self.user_input.escape_pressed()):
                self.env.close()
                break

            # --------- NEXT LOOP ---------
            self.current_state = next_state
            if done or self.user_input.reset == True or isopen == False:
                self.step = 0
                self.env.reset()
                self.user_input.reset = False
        self.data.plot_safety(all_info, frames)

    def dagger(self, model):
        max_timesteps = 1000
        n_test_episodes = 10
        episode_reward = 0
        episode_rewards = []  #To record the average reward
        good_expert = 0

        for i in range(n_test_episodes):
            self.data.data["state"] = []
            self.data.data["info"] = []
            self.data.data["action"] = []
            state = self.env.reset()
            episode_reward = 0
            for _ in range(max_timesteps):
                # Define a state to test the model on
                state = preprocess_state(self.current_state)
                info = self.env.read_info()  #

                # Wrap the state and info in Tensors and add a batch dimension
                state = torch.tensor(state).unsqueeze(0).unsqueeze(0).to(
                    torch.float32)
                info = torch.tensor(info).unsqueeze(0).to(torch.float32)

                # Pass the state and info through the model and get the output action
                action = model((state, info))
                # Convert tensor to tuple
                action = tuple(action.squeeze().tolist())

                # act
                next_state, reward, done, info = self.env.step(action)
                # record
                episode_reward += reward
                self.data.record(self.current_state, info, action)

                # render
                isopen = self.env.render()
                self.current_state = next_state
                self.step += 1

                if done:
                    break

                # ---------- ACTIONS ----------
                # close window
                if (self.user_input.escape_pressed()):
                    self.env.close()
                    break

                # --------- NEXT LOOP ---------
                if self.user_input.reset == True or isopen == False:
                    self.env.reset()
                    self.user_input.reset = False
            if (i + 1) % 10 == 0:
                print(f'Episode {i+1}')
            episode_rewards.append(episode_reward)

            if episode_reward > 1000:
                good_expert += 1
                print(f'GOOD EXPERT with reward {episode_reward}')
                self.data.save()
        self.env.close()
        average = sum(episode_rewards) / len(episode_rewards)
        print(f"-------------AVERAGE REWARD: {average}")
        print(
            f"-------------No.GOOD EXPERT: {good_expert} out of {n_test_episodes} experts"
        )


def preprocess_state(states):
    """ Preprocess the images (states) of the expert dataset before feeding them to agent """
    #states_pp = np.copy(states)
    states_pp = np.array(states, dtype=np.float32)
    # Paint black over the sum of rewards
    states_pp[:, 85:, :15] = [0.0, 0.0, 0.0]

    # Replace the colors defined bellow
    def replace_color(old_color, new_color, states_pp):
        mask = np.all(states_pp == old_color, axis=-1)
        states_pp[mask] = new_color

    # Black bar
    replace_color([0., 0., 0.], [120.0, 120.0, 120.0], states_pp)
    #print("black bar replaced")
    # Road
    new_road_color = [102.0, 102.0, 102.0]
    replace_color([102., 102., 102.], new_road_color, states_pp)
    replace_color([105., 105., 105.], new_road_color, states_pp)
    replace_color([107., 107., 107.], new_road_color, states_pp)
    #print("road replaced")
    # Curbs
    replace_color([255., 0., 0.], new_road_color, states_pp)
    replace_color([255., 255., 255.], new_road_color, states_pp)
    #print("curb replaced")

    # Grass
    #new_grass_color = [0.0, 0.0, 0.0]
    new_grass_color = [102., 229., 102.]
    replace_color([102., 229., 102.], new_grass_color, states_pp)
    replace_color([102., 204., 102.], new_grass_color, states_pp)
    #print("grass replaced")
    # Float RGB represenattion
    states_pp /= 255.

    # Converting to gray scale
    states_pp = rgb2gray(states_pp)

    return states_pp