from gc import enable
import torch
from pytorch_msssim import SSIM
from tqdm import tqdm 
from torchmetrics.image import PeakSignalNoiseRatio
import functools
import numpy as np

def training(model,camera_data,optimizers,schedulers,c2w_data,images_data,args):

    device='cuda'
    num_iterations = args.max_steps
    start = 0
    ssim_lambda = 0.2
    ssim = SSIM(data_range=1.0, size_average=True, channel=3)
    model.to(device)
    c2w_data.to(device)
    psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
    random_indices_unique = np.random.choice(range(num_iterations), size=num_iterations, replace=False)
    data_num = int(images_data.shape[0])

    for step in tqdm(range(start,start+num_iterations)):
           
          
          for optimizer in optimizers.values():
              optimizer.zero_grad()
          c2w = c2w_data[random_indices_unique[step] % data_num]
          image = images_data[random_indices_unique[step] % data_num]

          out = model(camera_data,c2w,step)
          pre_img = out['rgb']
          pre_img = pre_img.reshape(int(image.shape[0]),int(image.shape[1]),3)
          image = image.reshape(int(image.shape[0]),int(image.shape[1]),3)
       
          Ll1 = torch.abs(image - pre_img).mean()
          simloss = 1 - ssim(image.permute(2, 0, 1)[None, ...], pre_img.permute(2, 0, 1)[None, ...])
          loss = (1 - ssim_lambda) * Ll1 + ssim_lambda * simloss

          loss.backward()
          for optimizer in optimizers.values():
                optimizer.step()
  
          schedulers['xyz'].step()
          model.after_train(step)
          model.refinement_after(optimizers,step)
          


          if step % 1000 == 0:
               print(loss) 
    ps = psnr(pre_img,image)
    print(loss)  
    print(ps)                  
    ply_path = 'gaus.ply'
    model.save_ply2(ply_path)
    torch.save(model.state_dict(), 'model_path.pth')
    print("MODEL saved")