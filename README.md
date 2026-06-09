# Naex

A package for planning in a traversability grid using AStar. Fork for Taros Mule @ ELROB 2026.

## Taros Mule Planner

#### Planner loop

The planner loop goes off on each received traversability point cloud:
  1. The planner grid/graph is built.
  2. If latest received path is obstacle free, it is republished and the loop ends.
  3. Else, we will be replanning:
      1. The start node of the graph is selected (has to be traversable and has to be close to the robot, if this cannot be satisfied, return).
      2. Run AStar on the graph (no early stopping).
      3. Choose goal of the path as the last reachable point (i.e. AStar could get there) on the original path (e.g. if there is an obstacle in the middle of the path and we can see behind it (tree, red&white tape), then choose the point behind it, if the obstacles is huge, like a building, then we will likely only select the last point on the path before the building).
      4. Based on the AStar results get the optimal path from start to goal.
      5. If the tail of the path (the part of the original path from the planner goal to its original goal) is traverable, append it to the planned path. Else don't.
      6. Publish path.

#### Subscribed topics

- `points` -> `/terrain_map` (sensor_msgs/PointCloud2)
  Single scan traversability map in the form of a square grid point cloud with 'traversability' score for each point. The planner builds a graph out of this to run AStar on.
- `path` -> `/path` (nav_msgs/Path)
  The proposed odometry path. As long as it is traversable this node only republishes it. Once there is an obstacle on the path, the planner plans an alternative traversable path based on the original one.

#### Published Topics

- `~/planner_grid` -> `/mule_planner/planner_grid` (sensor_msgs/PointCloud2)  
  The graph created out of the single scan traversability that we plan in. Should basically mirror the subscribed `points` and is here just for debug.
- `~/path` -> `/mule_planner/path` (nav_msgs/Path)
  Planned path. Either original if traversable or in the opposite case a traversable alternative. 


## Usage

Launch: `ros2 launch mule_planner.launch.py use_sim_time:=false`
