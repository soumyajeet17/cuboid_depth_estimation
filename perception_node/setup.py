import os  # <-- ADDED
from glob import glob  # <-- ADDED
from setuptools import find_packages, setup

package_name = 'perception_node'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        
        # --- ADDED THIS SECTION ---
        # Include all launch files from the 'launch' directory
        (os.path.join('share', package_name, 'launch'), 
            glob(os.path.join('launch', '*.launch.py'))),
        # Include all config files from the 'config' directory
        (os.path.join('share', package_name, 'config'), 
            glob(os.path.join('config', '*.rviz'))),
        # --------------------------
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='soumyajeet',
    maintainer_email='soumyajeet17@github.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'depth_processor = perception_node.depth_processor:main',
        ],
    },
)
