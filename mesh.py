import torchvision
from tqdm import tqdm
import trimesh
import torch
import pymeshlab
import os
from PIL import Image
import plyfile
import skimage.measure
import numpy as np
import open3d as o3d

@torch.no_grad()
def convert_mesh(model,camera_data,c2w):

    model.load_state_dict(torch.load('model_path.pth'))
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=0.008,   
        sdf_trunc=0.1,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )
    K = np.array([[camera_data['fx'], 0, camera_data['cx']],
                  [0, camera_data['fy'], camera_data['cy']],
                  [0, 0, 1]],dtype=np.float64)
    custom_intrinsic = o3d.camera.PinholeCameraIntrinsic(int(camera_data['width']), int(camera_data['height']), intrinsic_matrix=K)   
    data_num = int(c2w.shape[0])
    os.makedirs('images',exist_ok=True)
    for i in tqdm(range(data_num)):
        c = c2w[i]
        
        out = model.forward(camera_data,c,i)
  
        depth_im = out['depth'].reshape(out['rgb'].shape[0],out['rgb'].shape[1])
        rgb = out['rgb'] * 255
        
        invalid_mask = depth_im <= 0
        depth_im[invalid_mask] = 5.
        color_np = rgb.cpu().numpy().astype(np.uint8)
        depth_np = depth_im.cpu().numpy().astype(np.float32)
        c = c.cpu().numpy()
        c[0:3, 1:3] *= -1
        
        color_image = o3d.geometry.Image(color_np)
        depth_image = o3d.geometry.Image(depth_np)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_image, depth_image,depth_scale=1.0,depth_trunc=4.8, convert_rgb_to_intensity=False)
        volume.integrate(
        rgbd,
        custom_intrinsic,
        np.linalg.inv(c))
        
        # o3d.io.write_image(f"depths/depth{i}.png", depth_image)

    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()

    # if you want to scale this mesh by object's height
     
    # vertices = np.asarray(mesh.vertices)
    # min_y = np.min(vertices[:, 1])
    # max_y = np.max(vertices[:, 1])
    # current_height = max_y - min_y
    # target_height = 160
    # scale_factor = target_height / current_height
    # scale_matrix = np.diag([scale_factor, scale_factor, scale_factor])
    # mesh.vertices = o3d.utility.Vector3dVector(np.dot(vertices, scale_matrix))
    mesh = o3d.io.read_triangle_mesh("fused_mesh.ply")

    mesh.remove_unreferenced_vertices()
    mesh = mesh.remove_duplicated_triangles()
    mesh = mesh.remove_degenerate_triangles()
    number_of_triangles = 20000  
    mesh.remove_connected_components_by_triangle_count(number_of_triangles)
    o3d.io.write_triangle_mesh("clean_mesh.ply", mesh)
    
    # if you want to use pymeshlab

    # ms = pymeshlab.MeshSet()
    # ms.load_new_mesh('fused_mesh.ply')
    # ms.meshing_remove_unreferenced_vertices()
    # ms.meshing_remove_duplicate_faces()
    # ms.meshing_remove_null_faces()
    # ms.meshing_remove_connected_component_by_face_number(mincomponentsize=20000)
    # ms.save_current_mesh('clean_mesh.ply')
