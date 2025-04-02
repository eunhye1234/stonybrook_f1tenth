import cvxpy as cp
import time
import yaml
import gym
import numpy as np
from argparse import Namespace
import warnings
from auxiliary.parse_settings import parse_settings
from algorithms.ManeuverAutomaton import ManeuverAutomaton
from algorithms.MPC_Linear import MPC_Linear
from algorithms.GapFollower import GapFollower
from algorithms.DisparityExtender import DisparityExtender
from algorithms.SwitchingDriver import SwitchingDriver
from algorithms.PurePersuit import PurePersuit
import os
import sys

# from TinyLidarNet.Benchmark.f1tenth_benchmarks.zarrar.tiny_lidarnet import TinyLidarNet
# import sys
# sys.path.append("/home/ccri-batch2-car3/TinyLidarNet/Benchmark/f1tenth_benchmarks/zarrar")  # Add the external directory to sys.path

print("Hiiii")
import pyglet.gl
print("Bye")

import tiny_lidarnet

# CONTROLLER = ['PurePersuit']
# CONTROLLER = ['GapFollower']
CONTROLLER = ['TinyLidarNet']
RACETRACK = 'Spielberg'
# RACETRACK = 'ICRA'
VISUALIZE = False

if __name__ == '__main__':

    # parse users arguments
    if len(CONTROLLER) > 2:
        raise Exception('At most two motion planners are supported for head-to-head racing!')

    for s in CONTROLLER:
        if s not in ['ManeuverAutomaton', 'MPC_Linear', 'GapFollower', 'DissparityExtender', 'SwitchingDriver',
                     'PurePersuit', 'TinyLidarNet']:
            raise Exception('Specified controller not available!')

    if len(CONTROLLER) > 1 and VISUALIZE:
        VISUALIZE = False
        warnings.warn('Visualization is only supported for single agent racing!')

    # load the configuration for the desired Racetrack
    path = 'racetracks/' + RACETRACK + '/config_' + RACETRACK + '.yaml'
    with open(path) as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)

    # create the simulation environment and initialize it
    
    env = gym.make('f110_gym:f110-v0', map=conf.map_path, map_ext=conf.map_ext, num_agents=len(CONTROLLER))

    if len(CONTROLLER) == 1:
        obs, _, done, _ = env.reset(np.array([[conf.sx, conf.sy, conf.stheta]]))
    else:
        obs, _, done, _ = env.reset(np.array([[conf.sx, conf.sy, conf.stheta], [conf.sx2, conf.sy2, conf.stheta2]]))
    env.render()

    # initialize the motion planner
    controller = []
    params = env.params
    params['lidar'] = 0.0

    # for i in range(len(CONTROLLER)):
        # settings = parse_settings(CONTROLLER[i], RACETRACK, VISUALIZE)
        # exec('tmp = ' + CONTROLLER[i] + '(params, settings)')
        # controller.append(tmp)
        # print(type(tmp))  # <class 'algorithms.GapFollower.GapFollower'>


    test_id = "benchmark_tiny_il_m"
    # print(test_id)
    
    # tmp = TinyLidarNet(test_id,1, 0,'/home/ccri-batch2-car3/TinyLidarNet/Benchmark/f1tenth_benchmarks/zarrar/f1_tenth_model_diff_main_noquantized.tflite') #'shape': array([  1, 1081,   1], dtype=int32)
    # controller = TinyLidarNet(test_id,2, 0,'/home/ccri-batch2-car3/TinyLidarNet/Benchmark/f1tenth_benchmarks/zarrar/f1_tenth_model_small_noquantized.tflite')
    controller = tiny_lidarnet.TinyLidarNet(test_id,2, 0,'/home/ccri-batch2-car3/TinyLidarNet/Benchmark/f1tenth_benchmarks/zarrar/f1_tenth_model_small_noquantized.tflite')
    # print(type(tmp))
    
    # initialize auxiliary variables
    laptime = 0.0
    start = time.time()

    # main control loop
    while not done:

        # re-plan trajectory
        actions = []

        # for i in range(len(controller)):
            # speed, steer = controller[i].plan(obs['poses_x'][i], obs['poses_y'][i], obs['poses_theta'][i],
                                            #    obs['linear_vels_x'][i], obs['scans'][i])
            # speed, steer = controller.plan(obs['scans'][i])
            # actions.append([steer, speed])
        obs['scans'] = np.array(obs['scans'])     
        speed, steer = controller.plan(obs['scans'])
        print(f"speed, steer: {speed, steer}")
        # speed = 0.5
        # actions.append([steer, speed])
        actions.append([speed, steer])

        # update the simulation environment
        obs, step_reward, done, _ = env.step(np.asarray(actions))
        laptime += step_reward

        env.render(mode='human')

        # check if lap is finished
        if np.max(obs['lap_counts']) == 2:
            break

    # print results
    print('Sim elapsed time:', laptime, 'Real elapsed time:', time.time() - start)
