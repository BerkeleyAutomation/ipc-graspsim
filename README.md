### Introduction
Grasp simulator based on IPC/ipc (put link here), contains modifications from it.

1. unzip data.zip, will create `dexgrasp_data/` directory populated with mesh + json related data
2. build using docker/build.sh
3. run using docker/run.sh `obj_name` `grasp_ind`

So, for example, (including setup) you should be able to simulate a cube grasp with:
```
unzip data.zip
./docker/build.sh
./docker/run.sh cube_3cm 0
```
This will create grasp state files (.obj files including gripper + object) in `output/cube_3cm_0_(time)/`.

### Visualization:
The grasps can also be visualized at each timestep using any mesh visualizer, but it might be finicky to play through the various timesteps. 

If you install IPC-GraspSim's dependencies and build this package locally, 
you can use the --online flag on `run.sh` to  enable IPC's default visualization method.

If you have blender, then you can
1) use https://github.com/neverhood311/Stop-motion-OBJ to load in the object files in order 
(the `test` prefix can be used to load in the objects). 
2) use the existing `render_grasp.py` file to create high-quality images of each grasp 
```
python augment_objs.py <output_dir> # render gripper head + etc for improved vis
blender -b -P render_grasp.py <output_dir> # actual render vis, assumes augment_objs already run
```
In that case, images will be generated in `<output_dir>/images`. 
Can use ffmpeg to stitch them together:
```
ffmpeg -f image2 -framerate 30 -pattern_type glob -i '<output_dir>/images/*.png' ipc.mp4
ffmpeg -i ipc.mp4 -framerate 5 -vf "split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" ipc.gif
```

