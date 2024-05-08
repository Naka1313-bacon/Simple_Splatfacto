
import torch
import json
from process_data import read_cameras_binary,read_images_binary,camera_params2,camera_params
from plyfile import PlyData, PlyElement
import numpy as np
from splat import SplatModel
from splat import SplatModelConfig
from opt import create_parser
from run_train import training
import imageio
from multiprocessing import Pool
from glob import glob
from pprint import pprint
from mesh import convert_mesh
def read_image(file_path):

    return imageio.imread(file_path) / 255

def load_imgs(path):

    img_file_names = sorted(glob(path))
    images = []
    for f in img_file_names:
        image = read_image(f)
        np_img = np.array(image)
        images.append(image)

    images = np.array(images)
    return images

def load_ply_to_tensors(ply_file):
    ply_data = PlyData.read(ply_file)
    points = ply_data['vertex']
    positions = [[point['x'], point['y'], point['z']] for point in points]
    colors = [[point['red'], point['green'], point['blue']] for point in points]
    position_tensor = torch.tensor(positions, dtype=torch.float32)
    color_tensor = torch.tensor(colors, dtype=torch.float32)
    return position_tensor, color_tensor

def rotate_points(points):
    points[:, 1] *= -1  
    points[:, 2] *= -1 
    print(5)
    
    
    return points

class scheduler_func:
      
    def __init__(self,max_steps,lr_final):
        
        self.max_steps = max_steps
        self.lr_final = lr_final
    
    def func(self,step):
        
        lr_init = 1.6e-4
        lr_pre_warmup = 1e-8
        warmup_steps = 0
    
    
        t = np.clip(
            (step - warmup_steps) / (self.max_steps - warmup_steps), 0, 1
        )
        lr = np.exp(np.log(lr_init) * (1 - t) + np.log(self.lr_final) * t)
        return lr / lr_init  # divided by lr_init because the multiplier is with the initial learning rate


def setup_optimizers(model,func):
    optimizers = {
        'xyz': torch.optim.Adam([{'params': model.means}], lr=1.6e-4, eps=1e-15),
        'features_dc': torch.optim.Adam([{'params': model.features_dc}], lr=0.0025, eps=1e-15),
        'features_rest': torch.optim.Adam([{'params': model.features_rest}], lr=0.0025 / 20, eps=1e-15),
        'opacity': torch.optim.Adam([{'params': model.opacities}], lr=0.05, eps=1e-15),
        'scaling': torch.optim.Adam([{'params': model.scales}], lr=0.005, eps=1e-15),
        'rotation': torch.optim.Adam([{'params': model.quats}], lr=0.001, eps=1e-15),
        
    }

    schedulers = {
        'xyz': torch.optim.lr_scheduler.LambdaLR(optimizers['xyz'], lr_lambda=func),
    }

    return optimizers, schedulers

def reconstruction():
    
    parser = create_parser()  # 上記のcreate_parser関数を使用
    args = parser.parse_args()
    camera_data_path = args.camera_data_path
    image_data_path = args.image_data_path

    

    out = camera_params(image_data_path,camera_data_path)
    pos,color = load_ply_to_tensors(args.sparse_model_path)
    images = load_imgs(args.image_path)
    
    images_data = torch.from_numpy(images).reshape(-1, int(images.shape[2]), int(images.shape[1]), 3).type(torch.float)
    images_data = images_data.to('cuda')
    c2w = torch.tensor(out["c2w"],device='cuda',dtype=torch.float32)
    fx = out["fx"]
    fy = out["fy"]
    cx = out["cx"]
    cy = out["cy"]
    w = out["w"]
    h = out["h"]
    w = torch.tensor(images.shape[2])
    h = torch.tensor(images.shape[1])
    if args.resize:
       fx = torch.tensor(fx * (w / w))
       fy = torch.tensor(fy * (h / h))
       cx = torch.tensor(cx * (w / w))
       cy = torch.tensor(cy * (h / h))
       print(w,h,fx,fy,cx,cy,cy)
    camera_data = {
            'fx': fx, 
            'fy': fy,
            'cx': cx,
            'cy': cy,
            'width':w,
            'height':h }
    print(w,h,fx,fy,cx,cy,cy)

    assert c2w.shape[0] == images_data.shape[0]
    assert torch.isnan(c2w).any() == True

    points = (pos,color)

    config = SplatModelConfig(
        warmup_length=args.warmup_length,
        refine_every=args.refine_every,
        resolution_schedule=args.resolution_schedule,
        background_color=args.background_color,
        num_downscales=args.num_downscales,
        cull_alpha_thresh=args.cull_alpha_thresh,
        cull_scale_thresh=args.cull_scale_thresh,
        continue_cull_post_densification=args.continue_cull_post_densification,
        reset_alpha_every=args.reset_alpha_every,
        densify_grad_thresh=args.densify_grad_thresh,
        densify_size_thresh=args.densify_size_thresh,
        n_split_samples=args.n_split_samples,
        sh_degree_interval=args.sh_degree_interval,
        cull_screen_size=args.cull_screen_size,
        split_screen_size=args.split_screen_size,
        stop_screen_size_at=args.stop_screen_size_at,
        random_init=args.random_init,
        num_random=args.num_random,
        random_scale=args.random_scale,
        ssim_lambda=args.ssim_lambda,
        stop_split_at=args.stop_split_at,
        sh_degree=args.sh_degree,
        use_scale_regularization=args.use_scale_regularization,
        max_gauss_ratio=args.max_gauss_ratio,
        training = not args.only_mesh
    )
    model = SplatModel(seed_points=points,config=config)
   
    xyzfunc = scheduler_func(max_steps=args.max_steps,lr_final=1.6e-6)
    optimizers, schedulers = setup_optimizers(model, xyzfunc.func)
    
    if args.only_mesh:
        convert_mesh(model,camera_data,c2w)
    else:
       training(model,camera_data,optimizers,schedulers,c2w,images_data,args)
       convert_mesh(model,camera_data,c2w)

    



if __name__ == '__main__' :

    reconstruction() 

    