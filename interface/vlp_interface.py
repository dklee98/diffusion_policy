if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
    print(ROOT_DIR)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import rospy
import torch
import cv2
import numpy as np
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped, Twist
from cv_bridge import CvBridge, CvBridgeError
from scipy import spatial
from collections import deque
from typing import Dict, Callable
from message_filters import ApproximateTimeSynchronizer, Subscriber
import tf
import signal
import sys
import dill
import hydra
import time
from math import pi as M_PI
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy

def signal_handler(signal, frame): # ctrl + c -> exit program
        print('You pressed Ctrl+C!')
        sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

# Helper function to find nearest waypoint index for given position
def find_nearest_waypoint(waypoints, position):
    dist_list = [spatial.distance.euclidean(waypoint, position) for waypoint in waypoints]
    nearest_idx = np.argmin(dist_list)
    nearest_dist = dist_list[nearest_idx]
    return nearest_dist, nearest_idx

class InferenceNode:
    def __init__(self, model_path):
        self.debug = True
        # Initialize the ROS node
        rospy.init_node('inference_node', anonymous=True)
        
        # Initialize the CvBridge
        self.bridge = CvBridge()

        # Load the trained model policy
        self.policy = self.load_policy(model_path)
        self.policy.eval()
        
        # Observation
        self.img_odom_buffer = deque(maxlen = self.policy.n_obs_steps)
        self.state_obs = None

        # Subscribers for sync
        self.img_sub = Subscriber('/camera/left/rgb_img', Image)
        self.odom_sub = Subscriber('/ground_truth_pose', Odometry)

        # ApproximateTimeSynchronizer
        self.ats = ApproximateTimeSynchronizer([self.img_sub, self.odom_sub], queue_size=10, slop=0.1)
        self.ats.registerCallback(self.sync_callback)

        # Subscriber for goal point
        self.goal_sub = rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goalpoint_callback) 
        self.isGoal = False

        # Publisher for action
        self.action_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.action_path_debug = rospy.Publisher('/action_path', Path, queue_size=10)
        
        # Main loop
        self.run_inference()

    def load_policy(self, model_path):
        # Load the model (replace with your model loading code)
        payload = torch.load(open(model_path, 'rb'), pickle_module=dill)
        cfg = payload['cfg']
        cls = hydra.utils.get_class(cfg._target_)
        workspace = cls(cfg)
        workspace : BaseWorkspace
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)
        policy: BaseImagePolicy
        policy = workspace.model
        if cfg.training.use_ema:
            policy = workspace.ema_model
        policy.to('cuda' if torch.cuda.is_available() else 'cpu')

        # set inference params
        policy.num_inference_steps = 16 # DDIM inference iterations
        policy.n_action_steps = policy.horizon - policy.n_obs_steps + 1
        return policy
    
    def sync_callback(self, img_msg, odom_msg):
        # print('asdfasd')
        synced_time = img_msg.header.stamp.to_sec()
        ## Img
        try:
            # Convert the ROS Image message to OpenCV2
            cv_image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
            # Convert to the required format
            cv_image = cv2.resize(cv_image, (640, 480)).astype(np.float32) / 255.0    # (H, W, C)
            cv_image = np.moveaxis(cv_image, -1, 0)  # Move channel to first dimension (C, H, W)
            # if self.debug: print(f"Image shape: {cv_image.shape}")
        except CvBridgeError as e:
            rospy.logerr(f"Failed to convert image: {e}")

        ## Odom
        # get R_t rotation matrix
        curr_pose = np.array([odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y, odom_msg.pose.pose.position.z], dtype=np.float32)
        curr_quat = np.array([odom_msg.pose.pose.orientation.x, odom_msg.pose.pose.orientation.y, odom_msg.pose.pose.orientation.z, odom_msg.pose.pose.orientation.w], dtype=np.float32)
        curr_rot = tf.transformations.quaternion_matrix(curr_quat)[:3,:3]
        curr_euler = tf.transformations.euler_from_quaternion(curr_quat)
        curr_vel = np.array([odom_msg.twist.twist.linear.x, odom_msg.twist.twist.linear.y, odom_msg.twist.twist.angular.z], dtype=np.float32)
        curr_vel = np.dot(curr_rot.T, curr_vel)

        self.img_odom_buffer.append({
            'image': cv_image, # (C, H, W)
            'pose': curr_pose, # (3,)
            'euler': curr_euler, # (3,)
            'velocity': curr_vel, # (3,)
            'time': synced_time # (1,)
        })

        # if self.debug: print(f"buffer length: {len(self.img_odom_buffer)}")

    def goalpoint_callback(self, msg):
        if self.img_odom_buffer is None:
            print("Goal: Waiting for state and image data...")
            return
        self.goal = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z], dtype=np.float32)
        self.isGoal = True

    def get_obs(self) -> dict:
        if len(self.img_odom_buffer) == 0:
            return None
        ## Make img dict (T, C, H, W)
        img_dict = np.stack([item['image'] for item in self.img_odom_buffer], axis=0)

        ## Make state dict (T, 3)
        state_dict = np.stack([item['velocity'] for item in self.img_odom_buffer], axis=0)

        ## cal relative goal
        if self.isGoal:
            goal_dict = np.stack([self.get_relative_goal(item['pose'], item['euler'], self.goal) for item in self.img_odom_buffer], axis=0)
        else:
            goal_dict = np.zeros((len(self.img_odom_buffer), 3), dtype=np.float32)
                
        ## Make timestamp dict (T,)
        timestamp_dict = np.array([item['time'] for item in self.img_odom_buffer])
        
        ## get obs
        obs_dict = dict()
        obs_dict['image'] = img_dict
        obs_dict['state'] = state_dict
        obs_dict['goal'] = goal_dict
        # obs_dict['timestamp'] = timestamp_dict

        # print(f"Image shape: {img_dict.shape}")
        # print(f"State shape: {state_dict.shape}")
        # print(f"Goal shape: {goal_dict.shape}")
        # print(f"Timestamp shape: {timestamp_dict.shape}")
        # print("Obs dict: ", obs_dict)
        return obs_dict

    def get_relative_goal(self, pose, euler, goal):
        R_curr = tf.transformations.euler_matrix(euler[0], euler[1], euler[2])[:3,:3]
        relative_goal = np.dot(R_curr.T, goal - pose)
        return relative_goal

    def dict_apply(self,
        x: Dict[str, torch.Tensor], 
        func: Callable[[torch.Tensor], torch.Tensor]
        ) -> Dict[str, torch.Tensor]:
        result = dict()
        for key, value in x.items():
            if isinstance(value, dict):
                result[key] = dict_apply(value, func)
            else:
                result[key] = func(value)
        return result
    
    def cal_future_positions(self,initial_pose, linear_velocity, angular_velocity, dt=0.05, steps=30):
        positions = [initial_pose]
        current_pose = initial_pose

        for _ in range(steps):
            x, y, theta = current_pose
            theta += angular_velocity * dt
            x += linear_velocity * np.cos(theta) * dt
            y += linear_velocity * np.sin(theta) * dt
            current_pose = (x, y, theta)
            positions.append(current_pose)

        return positions
          
    def run_inference(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if len(self.img_odom_buffer) == 0:
                print("Run: Waiting for state and image data...")
                rate.sleep()
                continue
            # if self.isGoal:
            if len(self.img_odom_buffer) == self.policy.n_obs_steps:
                s = time.time()
                obs_dict = self.get_obs()
                
                # self.img_odom_buffer.clear()
                if obs_dict is None:
                    rate.sleep()
                    continue
                # obs_dict['image'] = torch.from_numpy(obs_dict['image']).float().to('cuda')
                # obs_dict['state'] = torch.from_numpy(obs_dict['state']).float().to('cuda')
                # obs_dict['goal'] = torch.from_numpy(obs_dict['goal']).float().to('cuda')
                # obs_dict['timestamp'] = torch.from_numpy(obs_dict['timestamp']).float().to('cuda')
                obs_dict = self.dict_apply(obs_dict, lambda x: torch.from_numpy(x).unsqueeze(0).to('cuda'))
                result = self.policy.predict_action(obs_dict)
                action = result['action'][0].detach().to('cpu').numpy()
                print(result)
                # print(action)
                print('Inference latency[s]:', time.time() - s)

                action_1_step = action[0]

                action_1_step[0] = np.clip(action_1_step[0], -1.0, 1.0).item()
                action_1_step[1] = np.clip(action_1_step[1], -0.5, 0.5).item()
                action_1_step[2] = np.clip(action_1_step[2], -90.0 * M_PI / 180.0, 90.0 * M_PI / 180.0).item()

                # is action float?


                action_msg = Twist()
                action_msg.linear.x = action_1_step[0]
                action_msg.linear.y = action_1_step[1]
                action_msg.angular.z = action_1_step[2]
                self.action_pub.publish(action_msg)

                # Publish action path for debug
                future_positions = self.cal_future_positions((0.0, 0.0, 0.0), action_1_step[0], action_1_step[2])

                action_path_msg = Path()
                action_path_msg.header.stamp = rospy.Time.now()
                action_path_msg.header.frame_id = 'map'
                
                for pose in future_positions:
                    pose_msg = PoseStamped()
                    pose_msg.pose.position.x = pose[0]
                    pose_msg.pose.position.y = pose[1]
                    action_path_msg.poses.append(pose_msg)
                self.action_path_debug.publish(action_path_msg)

            rate.sleep()


if __name__ == '__main__':
    model_path = '/home/dklee98/git/diffusion_ws/diffusion_policy/data/outputs/2024.09.11/11.57.17_train_diffusion_unet_hybrid_velocity_local_planner/checkpoints/latest.ckpt'
    try:
        node = InferenceNode(model_path)
    except rospy.ROSInterruptException:
        pass
