<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

## Corrected Implementation Plan: ArUco Marker-Based Localization System

### Camera Calibration and Global Reference

Camera calibration will be performed using a ChArUco board to obtain the intrinsic camera matrix and distortion coefficients. The ChArUco board provides superior corner detection accuracy compared to standard ArUco markers due to its hybrid chessboard-marker design. This board will serve as the home marker, establishing the origin (0,0) of the global coordinate system.[^1][^2][^3][^4][^5]

### Pose Estimation and Spatial Relationship Calculation

Pose estimation for each ArUco marker will be performed using the `cv::solvePnP()` algorithm with the known physical marker dimensions and previously calibrated camera parameters. This yields rotation vectors (rvecs) and translation vectors (tvecs) for each detected marker.[^6][^7][^8][^9][^10]

The spatial relationship between markers is characterized by calculating:

- **θ_ab and θ_ba**: Angular differences using the yaw equations from the paper[^5]
- **φ_ab**: Phase difference calculated as φ = arctan(Δy/Δx) using the y and x distance components between markers[^5]
- **d_ab**: Euclidean distance extracted from the translation vector magnitude[^8][^11][^12]


### Graph-Based Mapping Architecture

The system implements a node-edge graph structure using three classes:

**Graph Class**: Stores all marker nodes in an `std::unordered_map<int, MarkerNode>` indexed by ArUco marker ID for O(1) lookup.[^13][^14]

**Node Class**: Represents each ArUco marker, storing its global (x, y) coordinates and maintaining an adjacency list (`std::vector<Edge>`) of connections to other visible markers.[^15][^16][^13][^5]

**Edge Class**: Encapsulates the spatial relationship E_ab = {φ_ab, θ_ab, θ_ba, d_ab} between two markers.[^17][^5]

### Multi-Hop Navigation with Edge Composition

The M function will be implemented to compose edges for indirect navigation paths. When navigating from marker A to marker C through intermediate marker B, the composite edge E_ac = M(E_ab, E_bc) is calculated using:[^5]

$$
\begin{align}
\phi_{ac} &= \phi_{ab} + \phi_{bc} \\
\theta_{ac} &= \theta_{ab} + \theta_{bc} - \phi_{bc} \\
\theta_{ca} &= -(\theta_{ab} + \theta_{bc} - \phi_{bc}) \\
d_{ac} &= \sqrt{d_{ab}^2 + d_{bc}^2 - 2 \cdot d_{ab} \cdot d_{bc} \cdot \cos(\phi_{bc})}
\end{align}
$$

This enables pathfinding and target heading calculation through the marker graph.[^5]

<div align="center">⁂</div>

[^1]: https://docs.opencv.org/4.x/da/d13/tutorial_aruco_calibration.html

[^2]: https://github.com/jamiemilsom/Fisheye_ChArUco_Calibration

[^3]: https://docs.opencv.org/4.x/d4/d94/tutorial_camera_calibration.html

[^4]: https://stackoverflow.com/questions/79084728/how-do-to-camera-calibration-using-charuco-board-for-opencv-4-10-0

[^5]: 2208.09355v1.pdf

[^6]: https://www.diva-portal.org/smash/get/diva2:1505194/FULLTEXT01.pdf

[^7]: https://github.com/tentone/aruco

[^8]: https://pmc.ncbi.nlm.nih.gov/articles/PMC12196723/

[^9]: https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html

[^10]: https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html

[^11]: https://forum.opencv.org/t/aruco-marker-distance-calculation-off/13026

[^12]: https://www.reddit.com/r/opencv/comments/1333jv5/question_aruco_marker_detection_distance/

[^13]: https://www.geeksforgeeks.org/cpp/implementation-of-graph-in-cpp/

[^14]: https://stackoverflow.com/questions/5493474/graph-implementation-c

[^15]: https://www.programiz.com/dsa/graph-adjacency-list

[^16]: https://www.geeksforgeeks.org/dsa/adjacency-list-meaning-definition-in-dsa/

[^17]: https://www.geeksforgeeks.org/competitive-programming/graph-implementation-using-stl-for-competitive-programming-set-2-weighted-graph/

