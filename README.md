# [Simple_Splatofacto](https://github.com/Naka1313-bacon/Simple_Splatfacto/)
This is the code simplified Splatofacto by nerfstudio and added an implementation creating mesh. 

# Install
```bash
pip install plyfile pymeshlab trimesh vdbfusion gsplat==1.0.0 torch==2.2.1 torchmetrics torchvision scikit-learn pytorch-msssim ninja open3d
```
# Run
```bash
python main.py --camera_data_path 'your_colmap_folder/cameras.bin' \
                --image_data_path 'your_colmap_folder/images.bin' \
                --image_path 'your_image_folder/*' \
                --sparse_model_path 'your_colmap_folder/model.ply' \
                --max_steps 4000 \
                --cull_alpha_thresh 0.005 \
                --n_split_samples 2 \
                --sh_degree 3 \
                --stop_split_at 3000
                
```
You can customize the other parameters, please refer the file opt.py.
If you just want to extract mesh, add the argument '--only_mesh'. 

# Guidline

The file 'process.py' is expected camera model of PINHOLE or SIMPLE RADIAL.If you want to use an other camera model parameter,you can change code in the file. 
