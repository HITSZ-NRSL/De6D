# Utilities

1. sloped terrain generation: synthesize a scene with slopes from a flat point cloud.

   `--dist_*`:  mean or variance of the distance to the slope. `--angle_*`: slope of the sloped terrain. `--viz`: visualization. `--smooth` : smooth the gap between the flat and sloped terrain. `--save`: save to disk. `--postprocess`: run the script of database building.

   * single frame data:

     ```bash
     python experiments/make_slope.py --cfg_file cfgs/kitti_models/det6d_car.yaml --dist_mean 22 --dist_var 8 --angle_mean 15 --angle_var 5 --smooth --viz --data_path ../data/kitti/training/velodyne/000021.bin
     ```

   * the whole datasets: as shown in `Getting Started`. `gt_box(x,y,z,dx,dy,dz,Θ)` will be converted into `gt_box(x,y,z,dx,dy,dz,rz,ry,rx)`

     ```bash
     python experiments/make_slope.py --cfg_file cfgs/kitti_models/det6d_car.yaml --dist_mean 22 --dist_var 8 --angle_mean 15 --angle_var 5 --viz --smooth
     ```

2. visualize a point cloud from given file (in bin/npy format) or directory.

   * for given directory:

     ```bash
     python experiments/viz/bin.py --file experiments/demo_data/slopedkitti/
     ```

   * for given file:

     ```
     python experiments/viz/bin.py --file experiments/demo_data/slopedkitti/000021.bin 
     ```

3. convert image sequence into gif animation.

   ```bash
   python experiments/utils/image2gif.py --dir experiments/demo_data/gazebo/upslope/image_2 --out experiments/results/gazebo/upslope --fps 30
   ```

4. data format conversion for point cloud: `ros topic` $\leftrightharpoons$ `bin` $\leftrightharpoons$ `pcd` . 

   example:

   ```bash
   python pcvt.py --source topic --topic /rslidar/points --dest pcd --output pcd_data_dir
   ```

   

# Exploration

1. visualize all model comparison results for the given data one by one.

   ```bash
   python experiments/demo_all.py experiments/demo_data/slopedkitti/000021.bin 
   ```

2. evaluate all models (ckpts in [utils/settings.py](utils/settings.py)) in various datasets (datasets in [utils/settings.py](utils/settings.py)) and plot performances. result will be stored in [results/distribution.pdf](results/distribution.pdf).

   ```bash
   python experiments/eval_all.py eval
   ```

   you can also plot performance from cache (log) after running cmd above.

   ```bash
   python experiments/eval_all.py
   ```

3. generate the Fig 6. of the paper. visualization result will be stored in [results/slopedkitti](results/slopedkitti)

   `--pause`: enable press `space` to continue.

   ```bash
   python experiments/viz/results.py --cfg_file cfgs/slopedkitti_models/det6d_car.yaml --ckpt models/det6d_car_slopeaug01_80.pth --pause
   ```

   or you can run all model at once: the visualization windows will be blocked in this case. 

   ```bash
   python experiments/viz/results_all.py
   ```

4. generate the Fig. 1 of the paper. result will be stored in [results](results/)

   ```bash
   python experiments/viz/cover.py --cfg_file cfgs/slopedkitti_models/det6d_car.yaml --ckpt models/det6d_car_slopeaug01_80.pth --data_path experiments/demo_data/gtav/collection1/points --load experiments/results/PointRCNN.npy --pause --save
   ```

5. visualize the process of Slope-Aug: press `space` to continue.

   ```bash
   python experiments/viz/make_slope.py --cfg_file cfgs/kitti_models/det6d_car.yaml
   ```

6. visualize the pipeline of networks: press `space` to continue. results will be stored in [results/pipeline](results/pipeline).

   ```bash
   python experiments/viz/seg_and_head.py --cfg_file cfgs/slopedkitti_models/det6d_car.yaml --ckpt models/det6d_car_slopeaug01_80.pth --data_path experiments/demo_data/slopedkitti/000021.bin
   ```

7. record gif animation as shown in our video in KITTI and SlopedKITTI.

   ```bash
   python experiments/viz/record_gif_all.py
   python experiments/viz/record_gif_gz_all.py
   ```

8. plot the Fig .3 dataset properties' distribution of the paper.

   ```bash
   python experiments/distribution.py 
   ```

9. visualize the sampling results of the backbone network.

   ```bash
   python experiments/viz/backbone_sampling.py --cfg cfgs/slopedkitti_models/det6d_car.yaml --ckpt models/det6d_car_slopeaug01_80.pth --data_path experiments/demo_data/slopedkitti/000021.bin 
   ```

10. visualize the datasets.

    `--rand`: (optional) randomly select n frames point cloud for visualization.

    ```bash
    python experiments/viz/slopedkitti.py --cfg_file cfgs/slopedkitti_models/det6d_car.yaml --rand 3
    ```
    
    
