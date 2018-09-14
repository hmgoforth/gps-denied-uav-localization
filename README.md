# GPS-Denied UAV Localization using Pre-existing Satellite Imagery

This repository contains the code for our paper here.

### Dependencies

Download dataset folders from this Google Drive link and add to top level of repo after downloading.

Python version, PyTorch version, OpenCV version

### Training and Testing Deep Features

Explain

### Testing Alignment on UAV Datasets

Explain

<!-- 

# BPVO

### Dependencies

- [bitplanes](https://github.com/halismai/bitplanes)

### Compiling

./build_vo_solver_video_test.sh

### Running

./vo_solver_video_test config/config_cust.cfg data/flight1.mp4 1

Expected output:
```
BitPlanes Parameters:
MultiChannelFunction = BitPlanes
ParameterTolerance = 0.00015
FunctionTolerance = 0.0001
NumLevels = 4
sigma = 1.618
verbose = 0
subsampling = 2

.
.
.
.
[ vo_solver_video_test.cc:0075 ]: Frame 00134 @ 19.49 Hz
pose = 
	 x: -335.664
	 y: 339.097
	 z: 1.16251
	 h: 0
.
.
.
.
.
```

### More Details

Instantiating a BPVO module:

BPVO bpvo_module(config_file, K);

Where std::string::config_file points to a .cfg for Bitplane tracker parameters (one of
these is provided in the config/ folder), and cv::Mat K is a 3x3 camera instrinsic matrix.

[More information on calibrating a camera to get the intrinsic matrix](https://www.mathworks.com/help/vision/ug/camera-calibration.html)

One way to get the intrinsic matrix is by performing proper camera calibration. There are
simpler ways to construct a slightly inaccurate but sufficient intrinsic matrix by
just knowing the focal length (in pixels) of a camera, and the height and width (in pixels) of
the images returned from the camera.

The bpvo_module.solver(global_x, global_y, alt, comp_heading, I) function will
use the telemetry (global_x, global_y, alt, comp_heading) and the current
camera image (cv::Mat I) to compute a refined telemetry estimate. The estimate is
returned as a pointer to a 1D array containing refined global_x, global_y, alt, comp_heading.

The input image to bpvo_module.solver must be non-null.

Any of the telemetry inputs can be specified as INFINITY. In this case, the function will ignore these
inputs, but still use the current image I to compute a refined pose.

### Simulation Test

Compilation: ./build_vo_solver_dir.sh

Running: ./vo_solver_dir config/config_cust.cfg path/to/frames/directory/ data/sm_telem.txt

Expected Output:

```
reading images ... 
reading csv ... 
BitPlanes Parameters:
MultiChannelFunction = BitPlanes
ParameterTolerance = 0.00015
FunctionTolerance = 0.0001
NumLevels = 4
sigma = 1.618
verbose = 0
subsampling = 2

Starting loop
frame 0 input telem = 
	x: -1510.65
	y: -2268.43
	alt: 436.928
	ch: 0
refined pose = 
	 x: -1510.65
	 y: -2268.43
	 z: 436.928
	 h: 0

.
.
.
.
.

frame 169 input telem = 
	x: -1509.08
	y: -461.968
	alt: 489.428
	ch: inf
refined pose = 
	 x: -1519.42
	 y: -474.938
	 z: 489.314
	 h: 96.8461

.
.
.
.
.
.

``` -->