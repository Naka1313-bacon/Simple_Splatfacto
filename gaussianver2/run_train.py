from gc import enable
import torch
from pytorch_msssim import SSIM
from tqdm import tqdm 
from torch.cuda.amp.grad_scaler import GradScaler
from torchmetrics.image import PeakSignalNoiseRatio
import functools
import numpy as np
from torch.nn import functional as F 

def edge_aware_normal_loss(I, D):
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).float().unsqueeze(0).unsqueeze(0).to(I.device)/4
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).float().unsqueeze(0).unsqueeze(0).to(I.device)/4
    dD_dx = torch.cat([F.conv2d(D[i].unsqueeze(0), sobel_x, padding=1) for i in range(D.shape[0])])
    dD_dy = torch.cat([F.conv2d(D[i].unsqueeze(0), sobel_y, padding=1) for i in range(D.shape[0])])
    
    dI_dx = torch.cat([F.conv2d(I[i].unsqueeze(0), sobel_x, padding=1) for i in range(I.shape[0])])
    dI_dx = torch.mean(torch.abs(dI_dx), 0, keepdim=True)
    dI_dy = torch.cat([F.conv2d(I[i].unsqueeze(0), sobel_y, padding=1) for i in range(I.shape[0])])
    dI_dy = torch.mean(torch.abs(dI_dy), 0, keepdim=True)

    weights_x = (dI_dx-1)**500
    weights_y = (dI_dy-1)**500
    loss_x = abs(dD_dx) * weights_x.T
    loss_y = abs(dD_dy) * weights_y.T
    loss = (loss_x + loss_y).norm(dim=0, keepdim=True)
    return loss.mean()


def ms_ssim(x, y, scales=3, window_size=11):
    ssim_loss = 0
    ssim = SSIM(data_range=1.0, size_average=True, channel=3)
    for i in range(scales):
        ssim_loss += 1/scales * ssim(x, y)
        x = F.avg_pool2d(x, 2, stride=2)
        y = F.avg_pool2d(y, 2, stride=2)

    return ssim_loss

def depth_to_normal(depth_map, c2w,camera_data):

    depth_map = depth_map.squeeze()
    height, width = depth_map.shape
    points_world = torch.zeros((height + 1, width + 1, 3)).to(depth_map.device)
    points_world[:height, :width, :] = unproject_depth_map(depth_map, c2w,camera_data)

    p1 = points_world[:-1, :-1, :]
    p2 = points_world[1:, :-1, :]
    p3 = points_world[:-1, 1:, :]

    v1 = p2 - p1
    v2 = p3 - p1

    normals = torch.cross(v1, v2, dim=-1)
    normals = normals / (torch.norm(normals, dim=-1, keepdim=True)+1e-8)

    return normals

def unproject_depth_map(depth_map, c2w,camera_data):
    depth_map = depth_map.squeeze()
    height, width = depth_map.shape
    x = torch.linspace(0, width - 1, width).cuda()
    y = torch.linspace(0, height - 1, height).cuda()
    Y, X = torch.meshgrid(y, x, indexing='ij')

    # Reshape the depth map and grid to N x 1
    depth_flat = depth_map.reshape(-1)
    X_flat = X.reshape(-1)
    Y_flat = Y.reshape(-1)

    # Normalize pixel coordinates to [-1, 1]
    X_norm = (X_flat / (width - 1)) * 2 - 1
    Y_norm = (Y_flat / (height - 1)) * 2 - 1

    # Create homogeneous coordinates in the camera space
    points_camera = torch.stack([X_norm, Y_norm, depth_flat], dim=-1)    

    # parse out f1, f2 from K_matrix
    f1 = camera_data['fx']
    f2 = camera_data['fy']

    # get the scaled depth
    sdepth = (f1 * points_camera[..., 2:3] + f2) / (points_camera[..., 2:3] + 1e-8)

    # concatenate xy + scaled depth
    points_camera = torch.cat((points_camera[..., 0:2], sdepth), dim=-1)
    points_camera = points_camera.view((height,width,3))
    points_camera = torch.cat([points_camera, torch.ones_like(points_camera[:, :, :1])], dim=-1)  
    points_world = torch.matmul(points_camera, c2w.inverse())

    # Discard the homogeneous coordinate
    points_world = points_world[:, :, :3] / points_world[:, :, 3:]
    points_world = points_world.view((height,width,3))

    return points_world
def training(model,camera_data,optimizers,schedulers,c2w_data,images_data,args):

    device='cuda'
    num_iterations = args.max_steps
    steps = 0
    start = 0
    ssim_lambda = 0.1
    ssim = SSIM(data_range=1.0, size_average=True, channel=3)
    scaler = GradScaler()
    model.to(device)
    c2w_data.to(device)
    psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
    random_indices_unique = np.random.choice(range(num_iterations), size=num_iterations, replace=False)
    data_num = int(images_data.shape[0])
    # print(c2w_data.shape,images_data.shape)
    for step in tqdm(range(start,start+num_iterations)):
           
         
          for optimizer in optimizers.values():
              optimizer.zero_grad()
        #   c2w = c2w_data[step % data_num]
        #   image = images_data[step % data_num]
          c2w = c2w_data[random_indices_unique[step] % data_num]
          image = images_data[random_indices_unique[step] % data_num]

          out = model(camera_data,c2w,step)
          pre_img = out['rgb']
          pre_img = pre_img.reshape(int(image.shape[0]),int(image.shape[1]),3)
          
          image = image.reshape(int(image.shape[0]),int(image.shape[1]),3)
       
          Ll1 = torch.abs(image - pre_img).mean()
          simloss = 1 - ms_ssim(image.permute(2, 0, 1)[None, ...], pre_img.permute(2, 0, 1)[None, ...])
          loss = (1 - ssim_lambda) * Ll1 + ssim_lambda * simloss
        #   if step < 4000:
        #        normal_loss = edge_aware_normal_loss(image,depth_to_normal(out["depth"], c2w,camera_data).permute(2,0,1))
        #        loss += ssim_lambda * normal_loss
          loss.backward()
          for optimizer in optimizers.values():
                optimizer.step()
  
          schedulers['xyz'].step()
          model.after_train(step)
          model.refinement_after(optimizers,step)
          


        #   needs_step = [
        #     group
        #     for group in optimizers.parameters.keys()
        #   ]
        #   optimizers.optimizer_step(needs_step)
          

          if step % 1000 == 0:
               print(loss)
    #   schedulers['camera_opt'].step() 
    ps = psnr(pre_img,image)
    print(loss)  
    print(ps)                  
    ply_path = '/content/drive/MyDrive/Colab Notebooks/gaussianver2/gus.ply'
    model.save_ply2(ply_path)
    torch.save(model.state_dict(), 'model_path.pth')
    print("MODEL saved")