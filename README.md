# [Simple_Splatofacto](https://github.com/Naka1313-bacon/Simple_Splatfacto/)
This is the code simplified Splatofacto by nerfstudio and added an implementation creating mesh. 

# Install
```bash
pip install plyfile gsplat torchmetrics torchvision scikit-learn pytorch-msssim ninja open3d
```
# Run
```bash
python main.py --camera_data_path 'your_colmap_folder/cameras.bin' \
                --image_data_path 'your_colmap_folder/images.bin' \
                --image_path 'your_image_folder/*' \
                --sparse_model_path 'your_colmap_folder/filter_model.ply' \
                --max_steps 30000 \
                
```
You can customize the other parameters, please refer the file opt.py.
If you just want to extract mesh, add the argument '--only_mesh'. 

# Guidline

The file 'process.py' is expected camera model of PINHOLE or SIMPLE RADIAL.If you want to use an other camera model parameter,you can change code in the file. 
