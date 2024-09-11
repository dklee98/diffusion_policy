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
from cv_bridge import CvBridge, CvBridgeError
from scipy import spatial
import tf
import signal
import sys
import dill
import hydra
import time
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
        
        # Initialize placeholders for the data
        self.curr_pose = None
        # Observation
        self.image = None
        self.agent_vel = None
        self.waypoints = np.zeros((5, 3))

        # Subscribers
        rospy.Subscriber('/camera/left/rgb_img', Image, self.image_callback)
        rospy.Subscriber('/ground_truth_pose', Odometry, self.odometry_callback)
        rospy.Subscriber('/waypoints', Path, self.waypoint_callback)

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
        
    def image_callback(self, msg):
        try:
            # Convert the ROS Image message to OpenCV2
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            # Convert to the required format
            self.image = cv2.resize(cv_image, (640, 480)).astype(np.float32) / 255.0
            self.image = np.moveaxis(self.image, -1, 0)  # Move channel to first dimension
            if self.debug: print(f"Image shape: {self.image.shape}")
        except CvBridgeError as e:
            rospy.logerr(f"Failed to convert image: {e}")

    def odometry_callback(self, msg):
        # get R_t rotation matrix
        self.curr_pose = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z], dtype=np.float32)
        curr_quat = np.array([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w], dtype=np.float32)
        curr_rot = tf.transformations.quaternion_matrix(curr_quat)[:3,:3]
        curr_vel = np.array([msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.angular.z], dtype=np.float32)
        curr_vel = np.dot(curr_rot.T, curr_vel)
        # Extract linear and angular velocity
        self.agent_vel = np.array([curr_vel[0], curr_vel[2]], dtype=np.float32)

    def waypoint_callback(self, msg):
        if self.curr_pose is None:
            return
        # Find closest waypoint
        input_wpts = []
        for i, pose in enumerate(msg.poses):
            input_wpts.append([pose.pose.position.x, pose.pose.position.y, pose.pose.position.z])
        closest_dist, closest_idx = find_nearest_waypoint(input_wpts, self.curr_pose)
        closest_idx = closest_idx + 1 if closest_idx < len(input_wpts) - 1 else closest_idx
        if self.debug: print(f"Closest idx: {closest_idx}")
        # Extract waypoints
        idx = closest_idx
        for i in range(5):
            self.waypoints[i] = input_wpts[idx]
            idx = idx + 1 if idx < len(input_wpts) - 1 else idx
        if self.debug: print(f"Waypoints: {self.waypoints}")
          
    def run_inference(self):
        rate = rospy.Rate(10)  # 10 Hz
        self.policy.reset()
        while not rospy.is_shutdown():
            if self.image is not None and self.agent_vel is not None:
                s = time.time()
                # Prepare the input tensor
                obs_dict = self.prepare_input(self.image, self.agent_vel, self.waypoints)

                # Perform inference
                result = self.policy.predict_action(obs_dict)
                action = result['action'][0].detach().cpu().numpy()
                if self.debug: print(f"Action: {action}")
                print(f"Inference time: {time.time() - s}")

            rate.sleep()

    def prepare_input(self, image, agent_vel, waypoints):
        # Stack the inputs into a single tensor
        image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        agent_vel_tensor = torch.tensor(agent_vel, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        waypoints_tensor = torch.tensor(waypoints.flatten(), dtype=torch.float32).unsqueeze(0)  # Add batch dimension

        # Concatenate all inputs along the appropriate dimension
        input_tensor = torch.cat([image_tensor, agent_vel_tensor, waypoints_tensor], dim=1)
        input_tensor = input_tensor.to('cuda' if torch.cuda.is_available() else 'cpu')
        return input_tensor

if __name__ == '__main__':
    model_path = 'path/to/your/model.pth'
    try:
        node = InferenceNode(model_path)
    except rospy.ROSInterruptException:
        pass
