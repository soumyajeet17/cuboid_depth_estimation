#!/usr/bin/env python3
# ^ Shebang line to ensure this script is executed using the python3 interpreter

# --- Import necessary libraries ---
import rclpy  # ROS 2 client library for Python
from rclpy.node import Node  # Base class for creating a ROS 2 node
import numpy as np  # Library for numerical operations
import open3d as o3d  # Library for 3D data processing (point clouds)
from std_msgs.msg import Header, Float64  # Standard ROS 2 message types
from sensor_msgs.msg import Image, PointCloud2, PointField  # ROS 2 message types for sensor data
from geometry_msgs.msg import Point  # ROS 2 message type for geometric points
import struct  # For packing/unpacking binary data (used for PointCloud2 colors)
import pandas as pd  # Library for data analysis and manipulation (saving results to a table)
import os  # Library for interacting with the operating system (creating directories, paths)

# Helper library for converting between ROS and NumPy.
# If you don't have it, run: sudo apt install ros-humble-sensor-msgs-py
from sensor_msgs_py import point_cloud2  # Utility functions for PointCloud2 messages

class DepthProcessorNode(Node):
    """
    This class defines the ROS 2 node that processes depth images.
    It subscribes to a depth image topic, performs 3D reconstruction,
    estimates plane properties (normal, area), estimates the axis of rotation,
    and publishes visualization and results.
    """
    def __init__(self):
        """Node constructor: initializes parameters, subscribers, and publishers."""
        super().__init__('depth_processor_node')  # Initialize the parent Node class
        self.get_logger().info('Depth Processor Node started.')  # Log a startup message

        # --- Parameters & Assumptions ---
        # ASSUMED Camera Intrinsics (Pinhole camera model parameters)
        # These values are placeholders and will be updated by the first message,
        # assuming the principal point is the image center.
        self.INTRINSICS = {
            'fx': 525.0,  # Focal length in x
            'fy': 525.0,  # Focal length in y
            'cx': 320.0,  # Principal point x-coordinate (optical center)
            'cy': 240.0,  # Principal point y-coordinate (optical center)
            'width': 640, # Image width in pixels
            'height': 480 # Image height in pixels
        }
        
        # --- Result storage ---
        self.centroid_list = []  # Stores the centroid of the object from each frame (for Task 2)
        self.results_table = []  # Stores a dictionary for each frame's results (for Task 1)
        self.frame_counter = 1  # Counter for numbering the processed frames
        self.processed_timestamps = set() # Stores timestamps to avoid reprocessing frames from a looped bag file

        # --- ROS Subscribers & Publishers ---
        # Create a subscriber to the depth image topic
        self.subscription = self.create_subscription(
            Image,  # The message type to receive
            '/depth',  # The topic name to subscribe to
            self.depth_callback,  # The function to call when a message is received
            10)  # Quality of Service (QoS) profile depth
        
        # Publisher for the 3D visualization (for RVIZ)
        self.vis_publisher = self.create_publisher(
            PointCloud2,  # The message type to publish
            '/perception/vis_cloud',  # The topic name for RVIZ
            10)  # QoS profile depth
        
        # Publishers for Task 1 Results (Plane properties)
        # Publishes the angle of the detected plane's normal vector
        self.angle_pub = self.create_publisher(Float64, '/perception/normal_angle', 10)
        # Publishes the visible area of the detected plane
        self.area_pub = self.create_publisher(Float64, '/perception/visible_area', 10)
        
        # Publisher for Task 2 Result (Rotation axis)
        # Publishes the estimated point on the axis of rotation
        self.axis_pub = self.create_publisher(Point, '/perception/rotation_axis_point', 10)


    def depth_callback(self, msg):
        """Main callback function, runs every time a new depth image is received."""
        
        # --- Timestamp Check (for looping bag files) ---
        # Check if we already processed this frame
        msg_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9  # Get unique timestamp
        if msg_time in self.processed_timestamps:
            return  # Skip this frame, we already have it
        self.processed_timestamps.add(msg_time)  # Add new timestamp to the set
        
        self.get_logger().info(f'Processing new frame (Timestamp: {msg_time})')
        
        # 1. Convert ROS Image message to NumPy array
        try:
            # msg_to_numpy handles different encodings (e.g., 16UC1, 32FC1)
            # and converts depth values to meters.
            depth_image, _, _ = self.msg_to_numpy(msg)
            
            # Update intrinsics based on the actual image message,
            # overriding the initial assumed values.
            self.INTRINSICS['width'] = msg.width
            self.INTRINSICS['height'] = msg.height
            self.INTRINSICS['cx'] = msg.width / 2.0  # Assume principal point is image center
            self.INTRINSICS['cy'] = msg.height / 2.0  # Assume principal point is image center
        except Exception as e:
            self.get_logger().error(f'Failed to convert image: {e}')
            return  # Stop processing this frame

        # 2. Convert depth image to 3D Point Cloud
        # This function projects each pixel (u, v, Z) to (X, Y, Z) in 3D space.
        points = self.depth_to_point_cloud(depth_image, self.INTRINSICS)
        
        # Basic check to ensure we have a valid point cloud
        if len(points) < 100:  # Threshold for minimum number of points
            self.get_logger().warn('Point cloud has too few points, skipping frame.')
            return

        # 3. Task 1: Estimate Face Properties
        # Create an Open3D PointCloud object
        pcd = o3d.geometry.PointCloud()
        # Set its points from the NumPy array
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Run RANSAC plane segmentation and compute properties
        normal, angle, area, inliers_indices = self.estimate_face_properties(pcd)
        
        # Check if RANSAC was successful
        if normal is not None:
            # --- Publish Task 1 results ---
            self.angle_pub.publish(Float64(data=angle))
            self.area_pub.publish(Float64(data=area))
            
            # --- Store results for final report ---
            self.results_table.append({
                'Image Number': self.frame_counter,
                'Estimated Normal Angle (deg)': round(angle, 2),
                'Visible Area (m^2)': round(area, 4)
            })
            self.frame_counter += 1  # Increment frame counter *only* on success

            # 4. Task 2: Estimate Rotation Axis
            # Calculate the centroid (average point) of the *entire* point cloud for this frame
            current_centroid = np.mean(points, axis=0)
            self.centroid_list.append(current_centroid)  # Add to our running list
            
            # Calculate the running average of all centroids seen so far
            # This provides a stable estimate of a point on the rotation axis.
            average_centroid = np.mean(np.array(self.centroid_list), axis=0)
            
            # --- Publish Task 2 result ---
            axis_msg = Point()  # Create a new Point message
            axis_msg.x = average_centroid[0]
            axis_msg.y = average_centroid[1]
            axis_msg.z = average_centroid[2]
            self.axis_pub.publish(axis_msg)  # Publish the average centroid
            
            # 5. Create Visualization Point Cloud (for RVIZ)
            # Color the RANSAC inliers (plane) red and outliers gray
            vis_pcd = self.color_pcd(pcd, inliers_indices)
            # Convert the colored Open3D PCD to a ROS PointCloud2 message
            ros_cloud = self.o3d_to_ros_pointcloud(vis_pcd, msg.header)
            # Publish the colored cloud
            self.vis_publisher.publish(ros_cloud)

        else:
            # RANSAC failed to find a plane
            self.get_logger().warn('RANSAC plane fitting failed for this frame.')

    def save_results_on_shutdown(self):
        """Saves the final results to text files upon node shutdown (e.g., Ctrl+C)."""
        self.get_logger().info('Shutting down and saving results...')
        
        # Check if we have any results to save
        if not self.results_table:
            self.get_logger().warn('No results to save.')
            return
            
        # --- Set save directory to your specific path ---
        # os.path.expanduser expands the '~' to your home directory
        # (e.g., '/home/username/Downloads/New_assesment/depth')
        save_dir = os.path.expanduser('~/Downloads/New_assesment/depth') 
        # Create the directory if it doesn't already exist
        os.makedirs(save_dir, exist_ok=True) 

        # --- Save Deliverable 2: Results Table (Task 1) ---
        # Convert the list of dictionaries into a pandas DataFrame
        df_results = pd.DataFrame(self.results_table)
        results_table_path = os.path.join(save_dir, 'results_table.txt')
        
        # Write the DataFrame to a formatted text file
        with open(results_table_path, 'w') as f:
            f.write("Task 1 Results: Normal Angle and Visible Area\n")
            f.write("="*50 + "\n")
            f.write(df_results.to_string(index=False)) # .to_string() formats it nicely
        self.get_logger().info(f'✅ Results table saved to: {os.path.abspath(results_table_path)}')

        # --- Save Deliverable 3: Rotation Axis (Task 2) ---
        # Calculate final average centroid
        if not self.centroid_list:
            self.get_logger().warn('No centroid data to calculate rotation axis.')
            return
            
        # Final calculation of the average point on the axis
        axis_point = np.mean(np.array(self.centroid_list), axis=0)
        # This is an *assumption* based on the problem setup (top-down rotation)
        axis_direction = np.array([0.0, 1.0, 0.0]) # Assumed to be the Y-axis
        
        rotation_axis_path = os.path.join(save_dir, 'rotation_axis.txt')
        
        # Write the axis information to a text file
        with open(rotation_axis_path, 'w') as f:
            f.write("Task 2 Result: Axis of Rotation\n")
            f.write("="*50 + "\n")
            f.write("Axis of Rotation Vector (Direction) [X, Y, Z]:\n")
            f.write(f"{axis_direction}\n\n")
            f.write("Point on the Axis (Average Centroid) [X, Y, Z]:\n")
            f.write(f"{np.round(axis_point, 4)}\n\n")
            f.write("Note: Direction vector [0, 1, 0] is assumed based on the\n")
            f.write("problem diagram (Top View) showing vertical rotation.\n")
        self.get_logger().info(f'✅ Rotation axis saved to: {os.path.abspath(rotation_axis_path)}')

    # --- Helper Functions ---

    def msg_to_numpy(self, msg):
        """
        Converts a ROS sensor_msgs/Image message to a NumPy array.
        Handles different depth encodings (e.g., 16-bit unsigned int, 32-bit float)
        and scales depth values to meters.
        """
        # Check the encoding of the depth image
        if msg.encoding in ['16UC1', 'mono16']:
            # 16-bit unsigned integer (common for depth cameras)
            dtype = np.uint16
            scale = 0.001  # ASSUMPTION: Depth is in millimeters, convert to meters
        elif msg.encoding in ['32FC1']:
            # 32-bit float
            dtype = np.float32
            scale = 1.0  # ASSUMPTION: Depth is already in meters
        else:
            # Unsupported encoding
            raise TypeError(f'Unsupported encoding: {msg.encoding}')

        # Create a NumPy array from the raw message data buffer
        depth_image = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, msg.width)
        
        # Return the depth image as float32 in meters, along with dimensions
        return depth_image.astype(np.float32) * scale, msg.width, msg.height

    def depth_to_point_cloud(self, depth_image, intrinsics):
        """
        Converts a depth image into a 3D point cloud using camera intrinsics.
        This process is called "unprojection" or "back-projection".
        """
        H, W = depth_image.shape  # Image Height and Width
        # Get intrinsic parameters
        fx, fy, cx, cy = intrinsics['fx'], intrinsics['fy'], intrinsics['cx'], intrinsics['cy']
        
        # Create a grid of pixel coordinates (u, v)
        u_grid, v_grid = np.meshgrid(np.arange(W), np.arange(H))
        
        # Get the depth (Z) value for each pixel
        Z = depth_image.copy()
        
        # --- Filtering ---
        # Filter out depth values that are too close or too far, setting them to NaN
        Z[(Z < 0.2) | (Z > 3.0)] = np.nan # Keep points between 0.2m and 3.0m
        
        # --- Unprojection Equations ---
        # X = (u - cx) * Z / fx
        X = (u_grid - cx) * Z / fx
        # Y = (v - cy) * Z / fy
        Y = (v_grid - cy) * Z / fy
        
        # Stack X, Y, and Z to create (H, W, 3) array
        # Then reshape to (N, 3) where N is the number of pixels
        points = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
        
        # Filter out any points containing NaN (from the depth filtering)
        return points[np.isfinite(points).all(axis=1)]

    def estimate_face_properties(self, pcd):
        """
        Runs RANSAC plane segmentation on the point cloud.
        If a plane is found, it computes the plane's normal vector,
        the angle of the normal w.r.t. the camera, and the visible area.
        """
        distance_threshold = 0.01  # 1 cm threshold for RANSAC inliers
        
        # Use Open3D's RANSAC plane segmentation
        # ransac_n=3: Minimum 3 points to define a plane
        # num_iterations: Number of RANSAC iterations
        plane_model, inliers_indices = pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=3,
            num_iterations=1000
        )
        
        # If RANSAC fails (no plane found)
        if len(inliers_indices) == 0:
            return None, 0.0, 0.0, []  # Return empty/default values

        # --- Calculate Normal Vector & Angle ---
        # plane_model is [A, B, C, D] from the plane equation Ax + By + Cz + D = 0
        A, B, C, D = plane_model
        face_normal = np.array([A, B, C])  # The normal vector is [A, B, C]
        
        # Camera's view direction is along the +Z-axis in the camera frame
        camera_normal = np.array([0, 0, 1.0])
        
        # Ensure the normal points *towards* the camera (negative Z direction)
        # If the dot product is positive, both vectors are in the same
        # general +Z direction (pointing away from camera). We flip the
        # face_normal so it points back towards the camera's origin.
        if np.dot(face_normal, camera_normal) > 0:
            face_normal = -face_normal
            
        # Calculate the angle between the plane normal (now pointing at camera)
        # and the camera's viewing axis (+Z).
        dot_product = np.dot(face_normal, camera_normal)
        norm_product = np.linalg.norm(face_normal) * np.linalg.norm(camera_normal)
        dot_product_norm = np.clip(dot_product / norm_product, -1.0, 1.0) # Avoid numerical errors
        
        # arccos will give an angle between 90 (grazing) and 180 (head-on)
        normal_angle_deg = np.degrees(np.arccos(dot_product_norm))

        # --- Calculate Area ---
        # Get only the inlier points (the plane)
        inlier_pcd = pcd.select_by_index(inliers_indices)
        
        # Compute the Oriented Bounding Box (OBB) of the inliers
        bbox = inlier_pcd.get_oriented_bounding_box()
        
        # Get the 3D dimensions (length, width, height) of the OBB
        extents = bbox.extent
        
        # The plane is 2D, so one of these extents will be very small (the "thickness")
        # We sort the extents to find the smallest one.
        sorted_extents = np.sort(extents)
        
        # The area is the product of the two *larger* extents.
        # sorted_extents[0] = smallest (thickness)
        # sorted_extents[1] = medium (width)
        # sorted_extents[2] = largest (length)
        visible_area = sorted_extents[1] * sorted_extents[2]
        
        return face_normal, normal_angle_deg, visible_area, inliers_indices

    def color_pcd(self, pcd, inliers_indices):
        """
        Creates a new Open3D PointCloud with colored points for visualization.
        RANSAC inliers (the plane) are colored red.
        Outliers are colored gray.
        """
        vis_pcd = o3d.geometry.PointCloud()  # Create a new, empty point cloud
        points = np.asarray(pcd.points)  # Get all points as a NumPy array
        
        # Create a color array, default to gray [0.5, 0.5, 0.5]
        colors = np.full((len(points), 3), [0.5, 0.5, 0.5]) # Gray
        
        # Set the color of the inlier points to red [1.0, 0.0, 0.0]
        colors[inliers_indices] = [1.0, 0.0, 0.0] # Red
        
        # Assign points and colors to the new point cloud
        vis_pcd.points = o3d.utility.Vector3dVector(points)
        vis_pcd.colors = o3d.utility.Vector3dVector(colors)
        return vis_pcd

    def o3d_to_ros_pointcloud(self, pcd, header):
        """
        Converts an Open3D PointCloud object (with colors)
        to a ROS sensor_msgs/PointCloud2 message.
        """
        points = np.asarray(pcd.points)  # Get (N, 3) xyz points
        colors = np.asarray(pcd.colors)  # Get (N, 3) rgb colors (as 0.0-1.0 float)
        
        # --- Pack Colors ---
        # RVIZ PointCloud2 'rgb' field is a single 4-byte float.
        # The color is packed into these 4 bytes (RGBA).
        
        # Convert 0.0-1.0 float colors to 0-255 uint8
        r = (colors[:, 0] * 255).astype(np.uint8)
        g = (colors[:, 1] * 255).astype(np.uint8)
        b = (colors[:, 2] * 255).astype(np.uint8)
        a = np.full_like(r, 255) # Alpha channel (fully opaque)
        
        # Bit-shift to pack R, G, B, A into a single 32-bit integer (uint32)
        # Layout in memory (little-endian): BBBB GGGG RRRR AAAA
        # The integer value will be: AAAA... RRRR... GGGG... BBBB...
        rgba_packed = (a.astype(np.uint32) << 24) | \
                      (r.astype(np.uint32) << 16) | \
                      (g.astype(np.uint32) << 8)  | \
                      b.astype(np.uint32)
                      
        # Re-interpret the bits of the 32-bit integer as a 32-bit float.
        # This is a common (though tricky) way to store RGB data in a float field.
        rgb_float = [struct.unpack('f', struct.pack('I', rgba))[0] for rgba in rgba_packed]

        # --- Create Structured Array ---
        # Combine xyz (float32) and rgb (float32) into a structured NumPy array.
        # This matches the memory layout required by PointCloud2.
        points_with_color = np.zeros(len(points), dtype=[
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32),
            ('rgb', np.float32)
        ])
        points_with_color['x'] = points[:, 0]
        points_with_color['y'] = points[:, 1]
        points_with_color['z'] = points[:, 2]
        points_with_color['rgb'] = rgb_float

        # --- Define PointCloud2 Fields ---
        # Describe the memory layout of the structured array to ROS.
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
        ]

        # --- Create the Message ---
        # Use the sensor_msgs_py helper to create the PointCloud2 message
        return point_cloud2.create_cloud(header, fields, points_with_color)

def main(args=None):
    """The main entry point for the ROS 2 node."""
    rclpy.init(args=args)  # Initialize the ROS 2 client library
    
    node = DepthProcessorNode()  # Create an instance of the node
    
    try:
        # "Spin" the node, which means it will enter a loop and
        # wait for callbacks (like new depth images) to occur.
        rclpy.spin(node)
    except KeyboardInterrupt:
        # This block executes if the user presses Ctrl+C
        # Call save function on Ctrl+C
        node.get_logger().info('Keyboard interrupt detected, shutting down.')
    finally:
        # This block runs *always* on shutdown (Ctrl+C or other exit)
        # This is the crucial part for saving data.
        node.save_results_on_shutdown()  # Save the results
        node.destroy_node()  # Clean up the node
        rclpy.shutdown()  # Shut down the ROS 2 client library

if __name__ == '__main__':
    # This ensures that main() is called only when the script is executed directly
    # (not when imported as a module).
    main()
