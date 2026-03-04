# Logic of our control 
Note upfront: The control code mainly presents two parts: Line keeping using Stanley steering

1. Vision-based line detection and projection
2. Lane keeping using Stanley steering control

---
## Line Detection:
Two approaches are tried during the developement:
### 1 RGB to Depth alignment:
Originally, our idea is that if we can map the line detected from RGB to the depth image using global 3D coordinate and extrinsic matrix given from the original document, we can know where the lines are, and from the pixel and depth combination, we can project the bird-eye view of lines w.r.t vehicle coordinate frame to furthher finish the line keeping control. Yet, the difficulty is that the scale makes the depth alignemnt kind of far off. 
### 2 Find camera height and use the assumptions that the line is on the ground. 
From the hardware document, we found the height of the camera mounting, further based on the assusmption that the lines arer on the ground, we further estimate the Z (scale depth) for each pixel of the detected lines, then project to 2D and planed the trajectory to follow. Please refer to our video demonstration in real time. 
