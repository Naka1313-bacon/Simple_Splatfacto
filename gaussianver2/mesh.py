import torchvision
from tqdm import tqdm
import vdbfusion
import trimesh
import torch
import pymeshlab
import os
from PIL import Image
import plyfile
import skimage.measure
import numpy as np
import open3d as o3d

def convert_sdf_samples_to_ply(
    alpha,
    point,
    rgb_map,
    normals,
    ply_filename_out,
    bbox,
    level=3,
    offset=None,
    scale=None,
):
    """
    Convert sdf samples to .ply

    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to

    This function adapted from: https://github.com/RobotLocomotion/spartan
    # """

    
    voxel_size = list((bbox[1]-bbox[0]) / np.array(alpha.shape))
    print('voxel',voxel_size)
    verts, faces, normals, values = skimage.measure.marching_cubes(
        alpha, level=level, spacing=voxel_size
    )
    faces = faces[...,::-1] # inverse face orientation

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = bbox[0,0] + verts[:, 0]
    mesh_points[:, 1] = bbox[0,1] + verts[:, 1]
    mesh_points[:, 2] = bbox[0,2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    # try writing to the ply file

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(mesh_points[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])
    device = 'cuda'
    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    print("saving mesh to %s" % (ply_filename_out))

def ndc_2_cam(ndc_xyz, intrinsic, W, H):
    inv_scale = torch.tensor([[W - 1, H - 1]], device=ndc_xyz.device)
    cam_z = ndc_xyz[..., 2:3]
    cam_xy = ndc_xyz[..., :2] * inv_scale * cam_z
    cam_xyz = torch.cat([cam_xy, cam_z], dim=-1)
    # print(cam_xyz.shape)
    # print(torch.inverse(intrinsic[0, ...].t()))
    cam_xyz = cam_xyz @ torch.inverse(intrinsic[0, ...].t())
    # print(cam_xyz.shape)
    return cam_xyz

def depth2point_cam(sampled_depth, ref_intrinsic):
    B, N, C, H, W = sampled_depth.shape
    valid_z = sampled_depth
    valid_x = torch.arange(W, dtype=torch.float32, device=sampled_depth.device) / (W - 1)
    valid_y = torch.arange(H, dtype=torch.float32, device=sampled_depth.device) / (H - 1)
    valid_y, valid_x = torch.meshgrid(valid_y, valid_x)
    # B,N,H,W
    valid_x = valid_x[None, None, None, ...].expand(B, N, C, -1, -1)
    valid_y = valid_y[None, None, None, ...].expand(B, N, C, -1, -1)
    ndc_xyz = torch.stack([valid_x, valid_y, valid_z], dim=-1).view(B, N, C, H, W, 3)  # 1, 1, 5, 512, 640, 3
    cam_xyz = ndc_2_cam(ndc_xyz, ref_intrinsic, W, H) # 1, 1, 5, 512, 640, 3
    return ndc_xyz, cam_xyz

def depth2point_world(depth_image, intrinsic_matrix, extrinsic_matrix):
    # depth_image: (H, W), intrinsic_matrix: (3, 3), extrinsic_matrix: (4, 4)
    _, xyz_cam = depth2point_cam(depth_image[None,None,None,...], intrinsic_matrix[None,...])
    xyz_cam = xyz_cam.reshape(-1,3)
    xyz_world = torch.cat([xyz_cam, torch.ones_like(xyz_cam[...,0:1])], axis=-1) @ torch.inverse(extrinsic_matrix.t()).transpose(0,1)
    xyz_world = xyz_world[...,:3]

    return xyz_world
def depth2point(depth_image, intrinsic_matrix, extrinsic_matrix):
    # depth_image: (H, W), intrinsic_matrix: (3, 3), extrinsic_matrix: (4, 4)
    _, xyz_cam = depth2point_cam(depth_image[None,None,None,...], intrinsic_matrix[None,...])
    xyz_cam = xyz_cam.reshape(-1,3)
    xyz_world = (xyz_cam - extrinsic_matrix[:3,3]) @ (extrinsic_matrix[:3,:3].T)
    # xyz_world = xyz_world[...,:3]
    print(xyz_world.shape)
    return xyz_cam.reshape(*depth_image.shape, 3), xyz_world.reshape(*depth_image.shape, 3)

@torch.no_grad()
def convert_mesh(model,camera_data,c2w):

    model.load_state_dict(torch.load('model_path.pth'))
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=0.008,    # 解像度も適切に設定
        sdf_trunc=0.1,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )
    K = np.array([[camera_data['fx'], 0, camera_data['cx']],
                  [0, camera_data['fy'], camera_data['cy']],
                  [0, 0, 1]],dtype=np.float64)
    custom_intrinsic = o3d.camera.PinholeCameraIntrinsic(int(camera_data['width']), int(camera_data['height']), intrinsic_matrix=K)
    # vdb_volume = vdbfusion.VDBVolume(voxel_size=0.02, sdf_trunc=0.08, space_carving=True)
    
    
    data_num = int(c2w.shape[0])
    os.makedirs('images',exist_ok=True)
    for i in tqdm(range(data_num)):
        c = c2w[i]
        
        out = model.forward(camera_data,c,i)
  
        depth_im = out['depth'].reshape(out['rgb'].shape[1],out['rgb'].shape[2])
        print(2)
        print(depth_im)
        rgb = out['rgb'] * 255
        color = rgb.reshape(depth_im.shape[0],depth_im.shape[1],3)
        alpha = out['alpha']
        
        invalid_mask = depth_im >= 4.5
        # rgb[invalid_mask] = 0.
        # print('rgb',rgb * 255)
        # depth_im[invalid_mask] = 0.
        depth_im[invalid_mask] = 5.
        color_np = color.cpu().numpy().astype(np.uint8)
        
        # color_np = color_np * 255
        depth_np = depth_im.cpu().numpy().astype(np.float32)
        # print('color',depth_np)
        # rgb_normalized = color_np.clip(0, 1)  # 0から1の範囲にクリッピング
        # rgb_uint8 = (rgb_normalized * 255).astype('uint8')
        c = c.cpu().numpy()
        c[0:3, 1:3] *= -1
        
        # Create Open3D Image objects
        color_image = o3d.geometry.Image(color_np)
        depth_image = o3d.geometry.Image(depth_np)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_image, depth_image,depth_scale=1.0,depth_trunc=4.8, convert_rgb_to_intensity=False)
        volume.integrate(
        rgbd,
        custom_intrinsic,
        np.linalg.inv(c))
        
        # depth_in_mm = (depth_np * 1000).astype(np.uint16)
        # depth_image_mm = o3d.geometry.Image(depth_in_mm)
        # o3d.io.write_image(f"images/color{i}.png", color_image)
        # o3d.io.write_image(f"depths/depth{i}.png", depth_image)
        
        # rendered_pcd_cam, rendered_pcd_world = depth2point(depth_im, K.to(depth_im.device), 
                                                                    #   c.to(depth_im.device))
        # rendered_pcd_world = rendered_pcd_world[~invalid_mask]
        # rendered_pcd_world = rendered_pcd_world[..., [1, 0, 2]]
        # rendered_pcd_world[...,0] = -rendered_pcd_world[...,0]
        # print(rendered_pcd_world.shape)
        # c_inv = c.inverse()
        # cam_center = c[:3, 3]
        # print(depth_np)
        # vdb_volume.integrate(rendered_pcd_world.double().cpu().numpy(), extrinsic=cam_center.double().cpu().numpy())
        # rgb = depth_im.cpu().numpy()
        # rgb_normalized = depth_np.clip(0, 1)  # 0から1の範囲にクリッピング
        # rgb_uint8 = (rgb_normalized * 255).astype('uint8')
        # im = Image.fromarray(rgb_uint8)
        # im.save(f"images/{i}.png")
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals() 
    vertices = np.asarray(mesh.vertices)

    # Y座標の最大値と最小値を計算
    # min_y = np.min(vertices[:, 1])
    # max_y = np.max(vertices[:, 1])
    
    # 現在のYの範囲
    # current_height = max_y - min_y
    
    # # 目標の高さ
    # target_height = 160
    
    # # スケーリングファクターを計算
    # scale_factor = target_height / current_height
    
    # # 全軸にわたってスケーリングを適用（ここではY軸のみにスケーリングを適用したい場合は、scale_factorを他の軸には1を使う）
    # scale_matrix = np.diag([scale_factor, scale_factor, scale_factor])
    
    # メッシュの頂点にスケーリングを適用
    
    o3d.io.write_triangle_mesh('fused_mesh.ply', mesh)

    # vertices, faces = vdb_volume.extract_triangle_mesh(min_weight=5)
    # geo_mesh = trimesh.Trimesh(vertices, faces)
    # geo_mesh.export('fused_mesh.ply')

    
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh('fused_mesh.ply')
    ms.meshing_remove_unreferenced_vertices()
    ms.meshing_remove_duplicate_faces()
    ms.meshing_remove_null_faces()
    ms.meshing_remove_connected_component_by_face_number(mincomponentsize=20000)
    ms.save_current_mesh('clean_mesh.ply')