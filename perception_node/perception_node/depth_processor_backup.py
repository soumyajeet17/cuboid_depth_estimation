#!/usr/bin/env python3
"""
ROS 2 Node for processing depth images to estimate cuboid rotation.

This node subscribes to a depth image topic, converts each image to a 3D
point cloud, uses RANSAC to find the largest planar face, and then
calculates that face's normal angle and visible area. It also estimates
the cuboid's axis of rotation. Results are published to ROS topics and
saved to text files on shutdown.
"""

# Standard Library Imports
import os
import struct
from typing import List, Optional, Tuple, Set

# Third-Party Imports
import numpy as np
import o3d
import pandas as pd
from rclpy.node import Node
from rclpy.parameter import Parameter

# ROS 2 Imports
import rclpy
from geometry_msgs.msg import Point
from sensor_msgs.msg import Image, PointCloud2, PointField
from sensor_msgs_py import point_cloud2  # sudo apt install ros-humble-sensor-msgs-py
from std_msgs.msg import Float64


class DepthProcessorNode(Node):
    """
    A ROS 2 node to process depth images and estimate cuboid properties.

    Subscribes:
        /depth (sensor_msgs/msg/Image): Raw depth images.

    Publishes:
        /perception/vis_cloud (sensor_msgs/msg/PointCloud2):
            A 3D point cloud for RVIZ, with the detected face colored red.
        /perception/normal_angle (std_msgs/msg/Float64):
            The angle (in degrees) between the largest face normal and the camera.
        /perception/visible_area (std_msgs/msg/Float64):
            The estimated visible area (in m^2) of the largest face.
        /perception/rotation_axis_point (geometry_msgs/msg/Point):
            The estimated center point of the cuboid (a point on the axis).
    """

    def __init__(self):
        """Initializes the node, parameters, publishers, and subscribers."""
        super().__init__('depth_processor_node')
        self.get_logger().info('Depth Processor Node started.')

        # --- Declare ROS 2 Parameters with default values ---
        self.declare_parameters(
            namespace='',
            parameters=[
                ('intrinsics.fx', 525.0),
                ('intrinsics.fy', 525.0),
                ('intrinsics.cx', 320.0),
                ('intrinsics.cy', 240.0),
                ('depth_filter.min_m', 0.2),
                ('depth_filter.max_m', 3.0),
                ('ransac.threshold_m', 0.01),
                ('ransac.min_points', 100),
                ('save_dir', '~/Downloads/New_assesment/depth'),
                ('rotation_axis_vector', [0.0, 1.0, 0.0])
            ]
        )

        # --- Get Parameters ---
        self.fx: float = self.get_parameter('intrinsics.fx').value
        self.fy: float = self.get_parameter('intrinsics.fy').value
        self.cx: float = self.get_parameter('intrinsics.cx').value
        self.cy: float = self.get_parameter('intrinsics.cy').value
        self.depth_min: float = self.get_parameter('depth_filter.min_m').value
        self.depth_max: float = self.get_parameter('depth_filter.max_m').value
        self.ransac_threshold: float = self.get_parameter('ransac.threshold_m').value
        self.ransac_min_points: int = self.get_parameter('ransac.min_points').value
        self.save_dir: str = os.path.expanduser(self.get_parameter('save_dir').value)
        self.rot_axis_vec: np.ndarray = np.array(
            self.get_parameter('rotation_axis_vector').value
        )

        # --- Internal State for Storing Results ---
        self.centroid_list: List[np.ndarray] = []
        self.results_table: List[dict] = []
        self.frame_counter: int = 1
        self.processed_timestamps: Set[float] = set()

        # --- ROS 2 Publishers ---
        self.vis_publisher = self.create_publisher(
            PointCloud2, '/perception/vis_cloud', 10
        )
        self.angle_pub = self.create_publisher(
            Float64, '/perception/normal_angle', 10
        )
        self.area_pub = self.create_publisher(
            Float64, '/perception/visible_area', 10
        )
        self.axis_pub = self.create_publisher(
            Point, '/perception/rotation_axis_point', 10
        )

        # --- ROS 2 Subscribers ---
        self.subscription = self.create_subscription(
            Image, '/depth', self.depth_callback, 10
        )

    def depth_callback(self, msg: Image):
        """
        Run every time a new depth image is received.

        Orchestrates the conversion, processing, and publishing of results.
        """
        # Prevent re-processing frames if the bag is looping
        msg_time: float = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if msg_time in self.processed_timestamps:
            return
        self.processed_timestamps.add(msg_time)

        self.get_logger().info(f'Processing new frame (Timestamp: {msg_time})')

        # --- 1. Convert ROS Image to NumPy Array ---
        try:
            depth_image, width, height = self.msg_to_numpy(msg)
            # Update intrinsic center based on actual image size, just in case
            self.cx = width / 2.0
            self.cy = height / 2.0
        except TypeError as e:
            self.get_logger().error(f'Failed to convert image: {e}')
            return

        # --- 2. Convert Depth Image to 3D Point Cloud ---
        points = self.depth_to_point_cloud(depth_image)
        if len(points) < self.ransac_min_points:
            self.get_logger().warn('Point cloud has too few points, skipping frame.')
            return

        # --- 3. Estimate Face Properties (Task 1) ---
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        result = self.estimate_face_properties(pcd)
        if result is None:
            self.get_logger().warn('RANSAC plane fitting failed for this frame.')
            return

        face_normal, angle_deg, area, inliers_indices = result

        # --- 4. Publish Results and Store for Saving ---

        # Publish Task 1 results
        self.angle_pub.publish(Float64(data=angle_deg))
        self.area_pub.publish(Float64(data=area))

        # Store results for final text file
        self.results_table.append({
            'Image Number': self.frame_counter,
            'Estimated Normal Angle (deg)': round(angle_deg, 2),
            'Visible Area (m^2)': round(area, 4)
        })
        self.frame_counter += 1

        # --- 5. Estimate Rotation Axis (Task 2) ---
        current_centroid = np.mean(points, axis=0)
        self.centroid_list.append(current_centroid)
        average_centroid = np.mean(np.array(self.centroid_list), axis=0)

        # Publish Task 2 result
        axis_msg = Point()
        axis_msg.x, axis_msg.y, axis_msg.z = average_centroid
        self.axis_pub.publish(axis_msg)

        # --- 6. Create and Publish RVIZ Visualization ---
        vis_pcd = self.color_pcd(pcd, inliers_indices)
        ros_cloud = self.o3d_to_ros_pointcloud(vis_pcd, msg.header)
        self.vis_publisher.publish(ros_cloud)

    def save_results_on_shutdown(self):
        """Saves the final results to text files upon node shutdown."""
        self.get_logger().info('Shutting down and saving results...')

        if not self.results_table:
            self.get_logger().warn('No results to save.')
            return

        # Ensure the target save directory exists
        os.makedirs(self.save_dir, exist_ok=True)

        # --- Save Deliverable 1: Results Table ---
        df_results = pd.DataFrame(self.results_table)
        results_table_path = os.path.join(self.save_dir, 'results_table.txt')
        try:
            with open(results_table_path, 'w') as f:
                f.write("Task 1 Results: Normal Angle and Visible Area\n")
                f.write("=" * 50 + "\n")
                f.write(df_results.to_string(index=False))
            self.get_logger().info(
                f'✅ Results table saved to: {os.path.abspath(results_table_path)}'
            )
        except IOError as e:
            self.get_logger().error(f'Failed to save results table: {e}')

        # --- Save Deliverable 2: Rotation Axis ---
        if not self.centroid_list:
            self.get_logger().warn('No centroid data to calculate rotation axis.')
            return

        axis_point = np.mean(np.array(self.centroid_list), axis=0)
        rotation_axis_path = os.path.join(self.save_dir, 'rotation_axis.txt')
        try:
            with open(rotation_axis_path, 'w') as f:
                f.write("Task 2 Result: Axis of Rotation\n")
                f.write("=" * 50 + "\n")
                f.write(f"Axis of Rotation Vector (Direction) [X, Y, Z]:\n")
                f.write(f"{self.rot_axis_vec}\n\n")
                f.write("Point on the Axis (Average Centroid) [X, Y, Z]:\n")
                f.write(f"{np.round(axis_point, 4)}\n\n")
                f.write("Note: Direction vector is an assumed parameter.\n")
            self.get_logger().info(
                f'✅ Rotation axis saved to: {os.path.abspath(rotation_axis_path)}'
            )
        except IOError as e:
            self.get_logger().error(f'Failed to save rotation axis: {e}')

    # --- Helper Functions ---

    def msg_to_numpy(self, msg: Image) -> Tuple[np.ndarray, int, int]:
        """
        Converts a ROS sensor_msgs/msg/Image into a NumPy array.

        Args:
            msg: The ROS Image message.

        Returns:
            A tuple containing:
                - The depth image as a NumPy array (in meters).
                - The image width.
                - The image height.

        Raises:
            TypeError: If the image encoding is not supported.
        """
        if msg.encoding in ['16UC1', 'mono16']:
            dtype = np.uint16
            scale = 0.001  # mm to meters
        elif msg.encoding == '32FC1':
            dtype = np.float32
            scale = 1.0  # Already in meters
        else:
            raise TypeError(
                f'Unsupported image encoding: {msg.encoding}. '
                "Expected '16UC1' or '32FC1'."
            )

        depth_image = np.frombuffer(msg.data, dtype=dtype).reshape(
            msg.height, msg.width
        )
        return depth_image.astype(np.float32) * scale, msg.width, msg.height

    def depth_to_point_cloud(self, depth_image: np.ndarray) -> np.ndarray:
        """
        Converts a depth image (in meters) to a 3D point cloud.

        Uses the "un-projection" pinhole camera model.
        Assumes intrinsics (fx, fy, cx, cy) are stored as class members.

        Args:
            depth_image: The depth image as a NumPy array.

        Returns:
            An (N, 3) NumPy array of 3D points (X, Y, Z).
        """
        H, W = depth_image.shape
        u_grid, v_grid = np.meshgrid(np.arange(W), np.arange(H))

        Z = depth_image.copy()
        # Filter out invalid depth values
        Z[(Z < self.depth_min) | (Z > self.depth_max)] = np.nan

        # Un-project 2D pixels + Z depth to 3D points
        X = (u_grid - self.cx) * Z / self.fx
        Y = (v_grid - self.cy) * Z / self.fy

        # Stack into an (N, 3) array and remove all NaN points
        points = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
        return points[~np.isnan(points).any(axis=1)]

    def estimate_face_properties(
        self, pcd: o3d.geometry.PointCloud
    ) -> Optional[Tuple[np.ndarray, float, float, np.ndarray]]:
        """
        Runs RANSAC on the point cloud to find the largest face and its properties.

        Args:
            pcd: The input Open3D PointCloud object.

        Returns:
            A tuple (face_normal, angle_deg, area, inliers_indices) if a
            plane is found, otherwise None.
        """
        # 1. Fit plane using RANSAC
        plane_model, inliers_indices = pcd.segment_plane(
            distance_threshold=self.ransac_threshold,
            ransac_n=3,
            num_iterations=1000
        )
        if len(inliers_indices) < self.ransac_min_points:
            return None  # Not enough points to be a valid plane

        # 2. Calculate Normal Vector and Angle
        A, B, C, _ = plane_model
        face_normal = np.array([A, B, C])

        # Camera viewing vector (positive Z-axis)
        camera_normal = np.array([0.0, 0.0, 1.0])

        # Ensure normal points "out" of the face, towards the camera
        # (dot product should be negative)
        if np.dot(face_normal, camera_normal) > 0:
            face_normal = -face_normal

        # Calculate angle using dot product
        dot_product = np.dot(face_normal, camera_normal)
        norm_product = np.linalg.norm(face_normal) * np.linalg.norm(camera_normal)
        angle_rad = np.arccos(np.clip(dot_product / norm_product, -1.0, 1.0))
        angle_deg = np.degrees(angle_rad)

        # 3. Calculate Visible Area
        inlier_pcd = pcd.select_by_index(inliers_indices)
        # Use Oriented Bounding Box (OBB) to estimate 2D area of the plane
        bbox = inlier_pcd.get_oriented_bounding_box()
        extents = bbox.extent
        # The area is the product of the two largest dimensions of the OBB
        sorted_extents = np.sort(extents)
        visible_area = sorted_extents[1] * sorted_extents[2]

        return face_normal, angle_deg, visible_area, np.array(inliers_indices)

    def color_pcd(
        self, pcd: o3d.geometry.PointCloud, inliers_indices: np.ndarray
    ) -> o3d.geometry.PointCloud:
        """
        Colors the RANSAC inlier points red and outliers gray.

        Args:
            pcd: The full Open3D PointCloud.
            inliers_indices: The indices of the points belonging to the plane.

        Returns:
            A new Open3D PointCloud with 'colors' attribute set.
        """
        points = np.asarray(pcd.points)
        # Initialize all points to gray
        colors = np.full((len(points), 3), [0.5, 0.5, 0.5])
        # Set inliers to red
        colors[inliers_indices] = [1.0, 0.0, 0.0]

        vis_pcd = o3d.geometry.PointCloud()
        vis_pcd.points = o3d.utility.Vector3dVector(points)
        vis_pcd.colors = o3d.utility.Vector3dVector(colors)
        return vis_pcd

    def o3d_to_ros_pointcloud(
        self, pcd: o3d.geometry.PointCloud, header: Header
    ) -> PointCloud2:
        """
        Converts an Open3D PointCloud (with colors) to a ROS PointCloud2 message.

        Args:
            pcd: The Open3D PointCloud (must have 'points' and 'colors').
            header: The ROS message header to use (for timestamp and frame_id).

        Returns:
            A sensor_msgs/msg/PointCloud2 message.
        """
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)

        # --- Pack RGB data into a single float32 ---
        # This is a common, if hacky, way to pack RGB for ROS 1/2
        # 1. Convert 0-1 float colors to 0-255 uint8
        r = (colors[:, 0] * 255).astype(np.uint8)
        g = (colors[:, 1] * 255).astype(np.uint8)
        b = (colors[:, 2] * 255).astype(np.uint8)
        a = np.full_like(r, 255)  # Alpha

        # 2. Bit-shift R, G, B, A into one 32-bit unsigned integer
        rgba_packed: np.ndarray = (
            (a.astype(np.uint32) << 24) |
            (r.astype(np.uint32) << 16) |
            (g.astype(np.uint32) << 8) |
            b.astype(np.uint32)
        )

        # 3. Re-interpret the bits of the uint32 as a float32
        # 'I' = unsigned int, 'f' = float
        rgb_float = [
            struct.unpack('f', struct.pack('I', rgba))[0] for rgba in rgba_packed
        ]

        # --- Create structured NumPy array for point_cloud2.create_cloud ---
        # This array must match the 'fields' definition
        points_with_color = np.zeros(
            len(points),
            dtype=[
                ('x', np.float32),
                ('y', np.float32),
                ('z', np.float32),
                ('rgb', np.float32),
            ]
        )
        points_with_color['x'] = points[:, 0]
        points_with_color['y'] = points[:, 1]
        points_with_color['z'] = points[:, 2]
        points_with_color['rgb'] = rgb_float

        # --- Define the PointCloud2 fields ---
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
        ]

        return point_cloud2.create_cloud(header, fields, points_with_color)


def main(args: Optional[List[str]] = None):
    """Entry point for the ROS 2 node."""
    rclpy.init(args=args)
    node = DepthProcessorNode()
    try:
        # rclpy.spin() keeps the node alive until shutdown
        rclpy.spin(node)
    except KeyboardInterrupt:
        # This block executes when you press Ctrl+C
        node.get_logger().info('Keyboard interrupt detected, shutting down.')
    finally:
        # This block executes regardless of how spin() exits
        # (e.g., Ctrl+C or rclpy.shutdown())
        node.save_results_on_shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
