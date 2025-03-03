

import os
import shutil
from glob import glob
from multiprocessing import Pool
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import imageio
import math
from PIL import Image
import struct
import argparse
import collections
import random

"""#Read cameras data"""
CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])

CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12)
}
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model)
                         for camera_model in CAMERA_MODELS])
CAMERA_MODEL_NAMES = dict([(camera_model.model_name, camera_model)
                           for camera_model in CAMERA_MODELS])

def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
 
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

def read_cameras_binary(path_to_model_file):
 
  
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(
                fid, num_bytes=24, format_char_sequence="iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(fid, num_bytes=8*num_params,
                                     format_char_sequence="d"*num_params)
            cameras[camera_id] = Camera(id=camera_id,
                                        model=model_name,
                                        width=width,
                                        height=height,
                                        params=np.array(params))
        assert len(cameras) == num_cameras
    return cameras


class Image(BaseImage):
      def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)

def qvec2rotmat(qvec):
    w, x, y, z = qvec
    return np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
    ])

def read_images_binary(path_to_model_file):

    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":   # look for the ASCII 0 entry
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, num_bytes=8,
                                           format_char_sequence="Q")[0]
            x_y_id_s = read_next_bytes(fid, num_bytes=24*num_points2D,
                                       format_char_sequence="ddq"*num_points2D)
            xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])),
                                   tuple(map(float, x_y_id_s[1::3]))])
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = Image(
                id=image_id, qvec=qvec, tvec=tvec,
                camera_id=camera_id, name=image_name,
                xys=xys, point3D_ids=point3D_ids)
    return images
def load_intrinsics(camera_path):
    cameras = read_cameras_binary(camera_path)
    # for i in cameras.keys():
    #     print(cameras[i])
    
    fxs = []
    fys = []
    H = []
    W = []
    cxs = []
    cys = []
    print('cameras',cameras[1])
    for i in cameras.keys():
        width = cameras[i][2]
        height = cameras[i][3]
        cx = cameras[i][4][2]
        cy = cameras[i][4][3]
        fx = cameras[i][4][0]
        fy = cameras[i][4][1]
        W.append(width)
        H.append(height)
        cxs.append(cx)
        cys.append(cy)
        fxs.append(fx)
        fys.append(fy)
    W = np.mean(W)
    H = np.mean(H)
    
    cx = np.mean(cxs)
    cy = np.mean(cys)
    fx = np.mean(fxs)
    fy = np.mean(fys)
    return W,H,fx,fy,cx,cy
def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2 """
    a, b = vec1.reshape(3), vec2.reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix
# Define the qvec2rotmat function
def qvec2rotmat(qvec):
    w, x, y, z = qvec
    return np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
    ])

def camera_params(image_data,camera_data):
    # Initialize output dictionary

    W,H,fx,fy,cx,cy = load_intrinsics(camera_data)
    images = read_images_binary(image_data)
    # Process each image
    frames = []
    for image_id, image in sorted(images.items()):
        print(image_id)
        rotation = qvec2rotmat(image.qvec)

        translation = image.tvec.reshape(3, 1)
        w2c = np.concatenate([rotation, translation], 1)
        w2c = np.concatenate([w2c, np.array([[0, 0, 0, 1]])], 0)
        c2w = np.linalg.inv(w2c)
        # Convert from COLMAP's camera coordinate system (OpenCV) to ours (OpenGL)
        c2w[0:3, 1:3] *= -1
        frames.append(c2w)
    
    output_data = {
        "w": W,
        "h": H,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,   
        "c2w": frames
    }
    return output_data

def camera_params2(images,cameras):
    # Initialize output dictionary
 out = {'frames': []}
 up = np.zeros(3)
 
 # Process each image
 for image_id, image in images.items():
    qvec = image.qvec
    tvec = image.tvec
    R = qvec2rotmat(-qvec)
    t = tvec.reshape([3, 1])
    bottom = np.array([0.0, 0.0, 0.0, 1.0]).reshape([1, 4])
    m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
    c2w = np.linalg.inv(m)
    c2w[0:3, 1:3] *= -1
    # c2w = c2w[[1, 0, 2, 3], :]
    # c2w[2, :] *= -1
    # up += c2w[0:3, 1]
    frame = {'file_path': image.name, 'transform_matrix': c2w.tolist()}
    out['frames'].append(frame)

    # Reorient the scene and adjust camera positions
    # up = up / np.linalg.norm(up)
    # R = np.eye(4)
    # R_align = rotation_matrix_from_vectors(up, np.array([0, 0, 1]))
    # R[:3, :3] = R_align
    # for frame in out['frames']:
    #     frame['transform_matrix'] = (R @ np.array(frame['transform_matrix'])).tolist()
    
    # Center and scale camera positions
    camera_positions = [np.array(frame['transform_matrix'])[:3, 3] for frame in out['frames']]
    center = np.mean(camera_positions, axis=0)
    scale = 4.0 / np.mean(np.linalg.norm(camera_positions - center, axis=1))
    
    for frame in out['frames']:
        transform_matrix = np.array(frame['transform_matrix'])
        transform_matrix[:3, 3] = (transform_matrix[:3, 3] - center) * scale
        frame['transform_matrix'] = transform_matrix.tolist()
 transform_matrixies = []
 for l, frame in enumerate(out['frames']):
    if np.isnan(np.array(frame['transform_matrix']).any()):
       print('nan')
    transform_matrix = np.array(frame['transform_matrix'])
    transform_matrixies.append(transform_matrix)

#  print(transform_matrixies.shape)
 camera_angle_x = math.atan(W / (intrinsic * 2)) * 2
 camera_angle_y = math.atan(H / (intrinsic * 2)) * 2

 focal_x = 0.5 * W / np.tan(0.5 * camera_angle_x)  # original focal length  
 focal_y = 0.5 * H / np.tan(0.5 * camera_angle_y)  # original focal length
 output_data = {
      "w": W,
      "h": H,
      "fx": focal_x,
      "fy": focal_y,
      "cx": cx,
      "cy": cy,   
      "camera_angle_x": camera_angle_x,
      "camera_angle_y": camera_angle_y,
      "c2w": transform_matrixies
  }
 return output_data
# # JSONファイルとして保存
# output_path = 'NeuRBF1/data/nerf_synthetic/indians/transforms_test.json'  # 適切なパスに変更してください
# with open(output_path, 'w') as json_file:
#     json.dump(output_data, json_file, indent=4)

# print(len(out['frames']))







