import rospy
import pygame
import yaml
import time
import gym
from argparse import Namespace
import numpy as np
# from f110_gym import F110Env
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import LaserScan
from auxiliary.parse_settings import parse_settings
from algorithms.ManeuverAutomaton import ManeuverAutomaton
from algorithms.MPC_Linear import MPC_Linear
from algorithms.GapFollower import GapFollower
from algorithms.DisparityExtender import DisparityExtender
from algorithms.SwitchingDriver import SwitchingDriver
from algorithms.PurePersuit import PurePersuit

# CONTROLLER = ['PurePersuit']
CONTROLLER = ['GapFollower']
RACETRACK = 'Spielberg'
# RACETRACK = 'ICRA'
VISUALIZE = False

class KeyboardControlF1TenthEnv:
    def __init__(self, *args, **kwargs):
        super(KeyboardControlF1TenthEnv, self).__init__(*args, **kwargs)
        
        # Initialize pygame for capturing keyboard input
        pygame.init()
        self.screen = pygame.display.set_mode((800, 600))   # Create a dummy screen for pygame
        pygame.key.set_repeat(50,50)
        self.clock = pygame.time.Clock()

        # Initialize car control variables
        self.speed = 0.0
        self.steering_angle = 0.0

        # Initialize ROS publishers
        rospy.init_node('f110gym_keyboard_control', anonymous=True)
        self.lidar_pub = rospy.Publisher('/gym_scan', LaserScan, queue_size=10)
        self.cmd_pub = rospy.Publisher('/gym_ackermann_cmd', AckermannDriveStamped, queue_size=10)

        # Initialize a list to record the data
        self.data_log = []

        # load the configuration for the desired Racetrack
        self.path = 'racetracks/' + RACETRACK + '/config_' + RACETRACK + '.yaml'
        with open(self.path) as file:
            conf_dict = yaml.load(file, Loader=yaml.FullLoader)
        self.conf = Namespace(**conf_dict)

        # create the simulation environment and initialize it
        self.env = gym.make('f110_gym:f110-v0', map=self.conf.map_path, map_ext=self.conf.map_ext, num_agents=len(CONTROLLER))
        

    def get_keyboard_input(self):
        """
        Capture keyboard input for controlling speed and steering angle.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
    
            if event.type == pygame.KEYDOWN:  #when key is pressed
                
                # Control steering angle with left and right arrow keys
                if event.key == pygame.K_LEFT:
                    self.steering_angle += 0.01  # Turn left

                elif event.key == pygame.K_RIGHT:
                    self.steering_angle -= 0.01  # Turn right
                
                elif event.key == pygame.K_SPACE:
                    self.steering_angle = 0.00  # Turn right

                # Control speed with up and down arrow keys
                elif event.key == pygame.K_UP:
                    self.speed += 0.1  # Increase speed

                elif event.key == pygame.K_DOWN:
                    self.speed -= 0.1  # Decrease speed

        # Limit the steering and speed values
        self.steering_angle = np.clip(self.steering_angle, -1.0, 1.0)  # Steering angle between -1 and 1
        self.speed = np.clip(self.speed, -3.0, 3.0)  # Speed between -3 and 3

    def step(self):
        """
        Overriding the step function to control the car using the keyboard.
        """
        # Capture keyboard input for car control
        # self.get_keyboard_input()
        
        # Use the keyboard-controlled speed and steering angle to simulate the car's movement
        # action = [self.steering_angle, self.speed]

        #Get input for controller
        speed, steer = controller[i].plan(obs['poses_x'][i], obs['poses_y'][i], obs['poses_theta'][i],
                                              obs['linear_vels_x'][i], obs['scans'][i])

        #Use the controller's speed and steering anle to simulate the car's movement  
        action = [steer, speed]

        action = np.asarray([action])

        # Call the original step function with the updated action
        obs, reward, done, info = self.env.step(action)
        self.env.render()

        # Publish LiDAR data (for example, simulate a random LiDAR scan)
        lidar_msg = LaserScan()
        lidar_msg.header.stamp = rospy.Time.now()
        obs['scans'] = obs['scans'][0]

        lidar_msg.ranges = list(map(float, obs['scans']))  # Use the LiDAR scan from the observation
        self.lidar_pub.publish(lidar_msg)

        # Publish Ackermann command (steering and speed)
        cmd_msg = AckermannDriveStamped()
        cmd_msg.header.stamp = rospy.Time.now()
        cmd_msg.drive.speed = self.speed  # Speed command
        cmd_msg.drive.steering_angle = self.steering_angle  # Steering command
        self.cmd_pub.publish(cmd_msg)

        # Log the data to save later
        self.data_log.append({
            'timestamp': rospy.Time.now().to_sec(),
            'lidar': lidar_msg.ranges,
            'speed': cmd_msg.drive.speed,
            'steering_angle': cmd_msg.drive.steering_angle
        })

        return obs, reward, done, info

    def reset(self):
        """
        Reset the environment and control variables.
        """
        print("reset")
        # obs = super(KeyboardControlF1TenthEnv, self).reset()
        obs, _, done, _ = self.env.reset(np.array([[self.conf.sx, self.conf.sy, self.conf.stheta]]))

        self.env.render()

        self.speed = 0.0
        self.steering_angle = 0.0
        return obs

    def save_log(self, filename='recorded_data.csv'):
        """
        Save the recorded data to a CSV file.
        """
        import pandas as pd
        df = pd.DataFrame(self.data_log)
        df.to_csv(filename, index=False)
        print(f"Data saved to {filename}")

    def render(self):
        """
        Optionally, render the environment (show the simulation in a window).
        """
        self.env.render()
        pass  # You can render the simulation, or keep it as is if not needed.

    def close(self):
        """
        Clean up and close the pygame window when done.
        """
        pygame.quit()
        self.env.close()


# Example of running the environment with keyboard control
if __name__ == "__main__":

    env = KeyboardControlF1TenthEnv()

    done = False
    
    # Reset the environment
    obs = env.reset()

    # Loop to drive the car
    while not done:
        obs, reward, done, info = env.step()  # Pass an empty action because it's controlled via keyboard

        env.render()
        pygame.event.pump()  # Keep the pygame window responsive (required)
        # done = env.is_done()  # Define an appropriate condition to stop
        env.clock.tick(30)  # Control the loop rate (e.g., 30Hz)

        # check if lap is finished
        if np.max(obs['lap_counts']) == 1:
            break
    
    pygame.quit()

    # Save the recorded data after the simulation
    env.save_log(filename='keyboard_controlled_data2.csv')
    env.close()
