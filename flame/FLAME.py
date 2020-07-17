"""
FLAME Layer: Implementation of the 3D Statistical Face model in PyTorch

It is designed in a way to directly plug in as a decoder layer in a 
Deep learning framework for training and testing

It can also be used for 2D or 3D optimisation applications

Author: Soubhik Sanyal
Copyright (c) 2019, Soubhik Sanyal
All rights reserved.

Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights on this
computer program.
You can only use this computer program if you have closed a license agreement with MPG or you get the right to use
the computer program from someone who is authorized to grant you that right.
Any use of the computer program without a valid license is prohibited and liable to prosecution.
Copyright 2019 Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V. (MPG). acting on behalf of its
Max Planck Institute for Intelligent Systems and the Max Planck Institute for Biological Cybernetics.
All rights reserved.

More information about FLAME is available at http://flame.is.tue.mpg.de.

For questions regarding the PyTorch implementation please contact soubhik.sanyal@tuebingen.mpg.de
"""
# Modified from smplx code [https://github.com/vchoutas/smplx] for FLAME

import numpy as np
import torch
import torch.nn as nn
import pickle
from smplx.lbs import lbs, batch_rodrigues, vertices2landmarks, find_dynamic_lmk_idx_and_bcoords
from smplx.utils import Struct, to_tensor, to_np, rot_mat_to_euler

from types import SimpleNamespace

from pytorch3d.structures import Meshes, Textures
from pytorch3d.renderer import OpenGLOrthographicCameras, OpenGLPerspectiveCameras, look_at_view_transform,\
    RasterizationSettings, BlendParams, MeshRenderer, MeshRasterizer, HardPhongShader, HardFlatShader,\
    SoftSilhouetteShader, TexturedSoftPhongShader, PointLights, DirectionalLights

from utils.train_eval_utils import generate_random

class FLAME(nn.Module):
    """
    Given flame parameters this class generates a differentiable FLAME function
    which outputs the a mesh and 3D facial landmarks
    """
    def __init__(self, config):
        super(FLAME, self).__init__()
        print("creating the FLAME Decoder")
        with open(config.flame_model_path, 'rb') as f:
            self.flame_model = Struct(**pickle.load(f, encoding='latin1'))
        self.NECK_IDX = 1
        self.batch_size = config.batch_size
        self.dtype = torch.float32
        self.use_face_contour = config.use_face_contour
        self.faces = self.flame_model.f
        self.register_buffer('faces_tensor',
                             to_tensor(to_np(self.faces, dtype=np.int64),
                                       dtype=torch.long))

        pose_params = torch.zeros([self.batch_size, 6],
                                            dtype=self.dtype, requires_grad=True)
        self.register_parameter('pose_params', nn.Parameter(pose_params, requires_grad=True))

        # Fixing remaining Shape betas
        # There are total 300 shape parameters to control FLAME; But one can use the first few parameters to express
        # the shape. For example 100 shape parameters are used for RingNet project 
        default_shape = torch.zeros([self.batch_size, 300-config.shape_params],
                                            dtype=self.dtype, requires_grad=False)
        self.register_parameter('shape_betas', nn.Parameter(default_shape,
                                                      requires_grad=False))
        shape_params = torch.zeros([self.batch_size, config.shape_params],
                                    dtype=self.dtype, requires_grad=True)
        self.register_parameter('shape_params', nn.Parameter(shape_params,
                                                             requires_grad=True))

        # Fixing remaining expression betas
        # There are total 100 shape expression parameters to control FLAME; But one can use the first few parameters to express
        # the expression. For example 50 expression parameters are used for RingNet project 
        default_exp = torch.zeros([self.batch_size, 100 - config.expression_params],
                                    dtype=self.dtype, requires_grad=False)
        self.register_parameter('expression_betas', nn.Parameter(default_exp,
                                                            requires_grad=False))
        espression_params = torch.zeros([self.batch_size, config.expression_params],
                                  dtype=self.dtype, requires_grad=True)
        self.register_parameter('expression_params', nn.Parameter(espression_params,
                                                                 requires_grad=True))

        # Eyeball and neck rotation
        default_eyball_pose = torch.zeros([self.batch_size, 6],
                                    dtype=self.dtype, requires_grad=True)
        self.register_parameter('eye_pose', nn.Parameter(default_eyball_pose,
                                                            requires_grad=True))

        default_neck_pose = torch.zeros([self.batch_size, 3],
                                    dtype=self.dtype, requires_grad=True)
        self.register_parameter('neck_pose', nn.Parameter(default_neck_pose,
                                                            requires_grad=True))

        # Fixing 3D translation since we use translation in the image plane

        self.use_3D_translation = config.use_3D_translation

        default_transl = torch.zeros([self.batch_size, 3],
                                     dtype=self.dtype, requires_grad=True)
        self.register_parameter(
            'transl',
            nn.Parameter(default_transl, requires_grad=True))

        # The vertices of the template model
        self.register_buffer('v_template',
                             to_tensor(to_np(self.flame_model.v_template),
                                       dtype=self.dtype))

        # The shape components
        shapedirs = self.flame_model.shapedirs
        # The shape components
        self.register_buffer(
            'shapedirs',
            to_tensor(to_np(shapedirs), dtype=self.dtype))

        j_regressor = to_tensor(to_np(
            self.flame_model.J_regressor), dtype=self.dtype)
        self.register_buffer('J_regressor', j_regressor)

        # Pose blend shape basis
        num_pose_basis = self.flame_model.posedirs.shape[-1]
        posedirs = np.reshape(self.flame_model.posedirs, [-1, num_pose_basis]).T
        self.register_buffer('posedirs',
                             to_tensor(to_np(posedirs), dtype=self.dtype))

        # indices of parents for each joints
        parents = to_tensor(to_np(self.flame_model.kintree_table[0])).long()
        parents[0] = -1
        self.register_buffer('parents', parents)

        self.register_buffer('lbs_weights',
                             to_tensor(to_np(self.flame_model.weights), dtype=self.dtype))

        # Static and Dynamic Landmark embeddings for FLAME

        with open(config.static_landmark_embedding_path, 'rb') as f:
            static_embeddings = Struct(**pickle.load(f, encoding='latin1'))

        lmk_faces_idx = (static_embeddings.lmk_face_idx).astype(np.int64)
        self.register_buffer('lmk_faces_idx',
                             torch.tensor(lmk_faces_idx, dtype=torch.long))
        lmk_bary_coords = static_embeddings.lmk_b_coords
        self.register_buffer('lmk_bary_coords',
                             torch.tensor(lmk_bary_coords, dtype=self.dtype))

        if self.use_face_contour:
            conture_embeddings = np.load(config.dynamic_landmark_embedding_path,
                allow_pickle=True, encoding='latin1')
            conture_embeddings = conture_embeddings[()]
            dynamic_lmk_faces_idx = np.array(conture_embeddings['lmk_face_idx']).astype(np.int64)
            dynamic_lmk_faces_idx = torch.tensor(
                dynamic_lmk_faces_idx,
                dtype=torch.long)
            self.register_buffer('dynamic_lmk_faces_idx',
                                 dynamic_lmk_faces_idx)

            dynamic_lmk_bary_coords = conture_embeddings['lmk_b_coords']
            dynamic_lmk_bary_coords = torch.tensor(
                dynamic_lmk_bary_coords, dtype=self.dtype)
            self.register_buffer('dynamic_lmk_bary_coords',
                                 dynamic_lmk_bary_coords)

            neck_kin_chain = []
            curr_idx = torch.tensor(self.NECK_IDX, dtype=torch.long)
            while curr_idx != -1:
                neck_kin_chain.append(curr_idx)
                curr_idx = self.parents[curr_idx]
            self.register_buffer('neck_kin_chain',
                                 torch.stack(neck_kin_chain))



    def forward(self, shape_params=None, expression_params=None, pose_params=None, neck_pose=None, eye_pose=None, transl=None):
        """
            Input:
                shape_params: N X number of shape parameters
                expression_params: N X number of expression parameters
                pose_params: N X number of pose parameters
            return:
                vertices: N X V X 3
                landmarks: N X number of landmarks X 3
        """

        if shape_params is not None:
            self.shape_params = nn.Parameter(shape_params.to(device=self.shape_params.device), requires_grad=True)
        if expression_params is not None:
            self.expression_params = nn.Parameter(expression_params.to(device=self.expression_params.device),
                                                  requires_grad=True)
        if pose_params is not None:
            self.pose_params = nn.Parameter(pose_params.to(device=self.pose_params.device), requires_grad=True)
        if neck_pose is not None:
            self.neck_pose = nn.Parameter(neck_pose.to(device=self.neck_pose.device), requires_grad=True)
        if eye_pose is not None:
            self.eye_pose = nn.Parameter(eye_pose.to(device=self.eye_pose.device), requires_grad=True)
        if transl is not None:
            self.transl = nn.Parameter(transl.to(device=self.transl.device), requires_grad=True)

        betas = torch.cat([self.shape_params, self.shape_betas, self.expression_params, self.expression_betas], dim=1)
        full_pose = torch.cat([self.pose_params[:,:3], self.neck_pose, self.pose_params[:,3:], self.eye_pose], dim=1)
        template_vertices = self.v_template.unsqueeze(0).repeat(self.batch_size, 1, 1)

        vertices, _ = lbs(betas, full_pose, template_vertices,
                               self.shapedirs, self.posedirs,
                               self.J_regressor, self.parents,
                               self.lbs_weights, dtype=self.dtype)

        lmk_faces_idx = self.lmk_faces_idx.unsqueeze(dim=0).repeat(
            self.batch_size, 1)
        lmk_bary_coords = self.lmk_bary_coords.unsqueeze(dim=0).repeat(
            self.batch_size, 1, 1)
        if self.use_face_contour:

            dyn_lmk_faces_idx, dyn_lmk_bary_coords = find_dynamic_lmk_idx_and_bcoords(
                vertices, full_pose, self.dynamic_lmk_faces_idx,
                self.dynamic_lmk_bary_coords,
                self.neck_kin_chain, dtype=self.dtype)

            lmk_faces_idx = torch.cat([dyn_lmk_faces_idx, lmk_faces_idx], 1)
            lmk_bary_coords = torch.cat(
                [dyn_lmk_bary_coords, lmk_bary_coords], 1)

        landmarks = vertices2landmarks(vertices, self.faces_tensor,
                                             lmk_faces_idx,
                                             lmk_bary_coords)

        if self.use_3D_translation:
            landmarks += self.transl.unsqueeze(dim=1)
            vertices += self.transl.unsqueeze(dim=1)

        return vertices, landmarks


def get_flame_layer(flame_model_path: str, static_landmark_embedding_path: str, dynamic_landmark_embedding_path: str,
                    batch_size: int=1, shape_params: int=100, expression_params: int=50, pose_params: int=7,
                    use_3D_translation: bool = False):
    flame_conf = SimpleNamespace()
    flame_conf.flame_model_path = flame_model_path
    flame_conf.use_face_contour = True
    flame_conf.batch_size = batch_size
    flame_conf.shape_params = shape_params
    flame_conf.expression_params = expression_params
    flame_conf.pose_params = pose_params
    flame_conf.use_3D_translation = False
    flame_conf.static_landmark_embedding_path = static_landmark_embedding_path
    flame_conf.dynamic_landmark_embedding_path = dynamic_landmark_embedding_path

    flame = FLAME(flame_conf)
    return flame


def random_shape_params():
    ranges = np.zeros((100, 2))
    ranges[:, 0] = -2.0
    ranges[:, 1] = 2.0
    return generate_random(ranges)


def random_expression_params():
    ranges = np.array([
        [-2.0, 2.0],
        [-2.0, 2.0],
        [-2.0, 2.0],
        [-2.0, 2.0],
        [-2.0, 2.0],
        [-2.0, 2.0],
        [-2.0, 2.0],
        [-2.0, 2.0],
        [-2.0, 2.0],
        [-2.0, 2.0],

        [-2.0, 2.0],
        [-2.0, 2.0],
        [-2.0, 2.0],
        [-2.0, 2.0],
        [-2.0, 2.0],
        [-2.0, 2.0],
        [-2.0, 2.0],
        [-2.0, 2.0],
        [-2.0, 2.0],
        [-2.0, 2.0],

        [-2.0, 2.0],
        [-2.0, 2.0],
        [-2.0, 2.0],
        [-2.0, 2.0],
        [-2.0, 2.0],
        [-2.0, 2.0],
        [-2.0, 2.0],
        [-2.0, 2.0],
        [-2.0, 2.0],
        [-2.0, 2.0],

        [-2.0, 2.0],
        [-2.0, 2.0],
        [-2.0, 2.0],
        [-2.0, 2.0],
        [-2.0, 2.0],
        [-2.0, 2.0],
        [-2.0, 2.0],
        [-2.0, 2.0],
        [-2.0, 2.0],
        [-2.0, 2.0],

        [-2.0, 2.0],
        [-2.0, 2.0],
        [-2.0, 2.0],
        [-2.0, 2.0],
        [-2.0, 2.0],
        [-2.0, 2.0],
        [-2.0, 2.0],
        [-2.0, 2.0],
        [-2.0, 2.0],
        [-2.0, 2.0]
    ])
    return generate_random(ranges)


def random_pose_params():
    radian = np.pi/180.0
    ranges = np.array([
        [-20*radian, 20*radian],
        [-30*radian, 30*radian],
        [0.0, 0.0],
        [0.0, 0.3],
        [0.0, 0.0],
        [0.0, 0.0]
    ])
    return generate_random(ranges)


def random_neck_pose_params():
    radian = np.pi/180.0
    ranges = np.array([
        [-25*radian, 25*radian],
        [-70*radian, 70*radian],
        [-30*radian, 30*radian]
    ])
    return generate_random(ranges)


def random_cam_params():
    ranges = np.array([
        [-0.2, 0.2],
        [-0.2, 0.2]
    ])
    return generate_random(ranges)


def random_scale_params():
    ranges = np.array([
        [0.7, 1.3]
    ])
    return generate_random(ranges).item()


def random_texture(texture_model, batch_size):
    # FLAME texture model
    texture_shape = texture_model['mean'].shape
    texture_num_pc = texture_model['tex_dir'].shape[-1]
    texture_mean = np.reshape(texture_model['mean'], (1, -1))
    texture_dir = np.reshape(texture_model['tex_dir'], (-1, texture_num_pc)).T
    texture_params = np.random.randn(texture_num_pc)[np.newaxis, :]
    texture = np.reshape(np.add(texture_mean, np.matmul(texture_params, texture_dir)), texture_shape)
    texture = texture / np.max(texture)

    faces_uvs = texture_model['ft']
    verts_uvs = texture_model['vt']

    texture = torch.cat(batch_size * [torch.tensor(texture, dtype=torch.float32).unsqueeze(0)])
    faces_uvs = torch.cat(batch_size * [torch.tensor(np.int64(faces_uvs), dtype=torch.int64).unsqueeze(0)])
    verts_uvs = torch.cat(batch_size * [torch.tensor(np.float32(verts_uvs), dtype=torch.float32).unsqueeze(0)])

    return texture, faces_uvs, verts_uvs


def render_images(vertices, faces, texture, faces_uvs, verts_uvs, crop_size, device):
    textures = Textures(maps=texture, faces_uvs=faces_uvs, verts_uvs=verts_uvs)
    meshes = Meshes(vertices, faces, textures)

    R, T = look_at_view_transform(1.0, 0.5, 0, device=device)
    camera = OpenGLPerspectiveCameras(R=R, T=T, fov=20, device=device)

    raster_settings = RasterizationSettings(
        image_size=crop_size * 2,
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=None,
        max_faces_per_bin=None
    )

    lights = PointLights(location=[[0.0, 0.0, -3.0]], device=device)

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=camera, raster_settings=raster_settings),
        shader=TexturedSoftPhongShader(cameras=camera, device=device, lights=lights)
    )

    images = renderer(meshes)
    return images


class TexturedFLAME(FLAME):

    def __init__(self, config, crop_size, device):
        super(TexturedFLAME, self).__init__(config)

        texture_model = np.load(config.texture_path)

        self.texture_shape = texture_model['mean'].shape
        self.texture_num_pc = texture_model['tex_dir'].shape[-1]
        self.register_buffer('texture_mean', torch.reshape(torch.from_numpy(texture_model['mean']).to(dtype=self.dtype),
                                                           (1, -1)))
        self.register_buffer('texture_dir', torch.reshape(torch.from_numpy(texture_model['tex_dir'])
                                                          .to(dtype=self.dtype), (-1, self.texture_num_pc)).t())

        # faces_uvs = torch.cat(flame_conf.batch_size * [
        #     torch.tensor(np.int64(faces_uvs), dtype=torch.int64, device=devices[0]).unsqueeze(0)])
        # # verts_uvs = torch.cat(flame_conf.batch_size * [torch.tensor(np.float32(verts_uvs), dtype=torch.float32, device=devices[0]).unsqueeze(0)])
        self.register_buffer('faces_uvs', torch.cat(self.batch_size * [torch.from_numpy(np.int64(texture_model['ft'])).unsqueeze(0)]))
        self.register_buffer('verts_uvs', torch.cat(self.batch_size * [torch.from_numpy(texture_model['vt']).to(dtype=self.dtype).unsqueeze(0)]))

        self.register_parameter('texture_params', nn.Parameter(torch.zeros((1, self.texture_num_pc),
                                                                          dtype=self.dtype, requires_grad=True),
                                                               requires_grad=True))

        raster_settings = RasterizationSettings(
            image_size=crop_size,
            blur_radius=0.0,
            faces_per_pixel=1,
            bin_size=None,
            max_faces_per_bin=None
        )

        R, T = look_at_view_transform(1.0, 0.5, 0)
        self.register_buffer('renderer_R', R)
        self.register_buffer('renderer_T', T)
        self.renderer_camera = OpenGLPerspectiveCameras(R=self.renderer_R, T=self.renderer_T, fov=20, device=device)
        renderer_lights = DirectionalLights(
            direction=[[0.0, 0.0, 2.0]],
            specular_color=[[0.0, 0.0, 0.0]],
            device=device
        )# PointLights(location=[[0.0, 0.0, -1.0]])
        renderer_rasterizer = MeshRasterizer(cameras=self.renderer_camera, raster_settings=raster_settings)
        renderer_shader = TexturedSoftPhongShader(cameras=self.renderer_camera, lights=renderer_lights, device=device)
        self.renderer = MeshRenderer(rasterizer=renderer_rasterizer, shader=renderer_shader)

    def forward(self, shape_params=None, expression_params=None, pose_params=None, neck_pose=None, eye_pose=None,
                transl=None, texture_params=None):
        vertices, landmarks = super(TexturedFLAME, self).forward(shape_params, expression_params, pose_params, neck_pose, eye_pose,
                                           transl)
        texture = torch.reshape(torch.add(self.texture_mean, torch.matmul(self.texture_params, self.texture_dir)), self.texture_shape)
        texture = texture.clamp(0.0, 255.0)
        texture = texture / 255.0
        texture = torch.cat(self.batch_size * [texture.unsqueeze(0)])

        textures = Textures(maps=texture, faces_uvs=self.faces_uvs, verts_uvs=self.verts_uvs)
        meshes = Meshes(vertices, torch.cat(vertices.shape[0] * [self.faces_tensor.unsqueeze(0)]), textures)

        images = self.renderer(meshes)
        return vertices, landmarks, images

    def transform_points(self, points):
        return self.renderer_camera.transform_points(points)


def get_texture_model(arg, devices):
    texture_model = np.load(arg.flame_texture_path)
    texture_shape = texture_model['mean'].shape
    texture_num_pc = texture_model['tex_dir'].shape[-1]
    texture_mean = torch.from_numpy(texture_model['mean'])
    texture_dir = torch.from_numpy(texture_model['tex_dir'])
    faces_uvs = torch.tensor(np.int64(texture_model['ft']), dtype=torch.int64).unsqueeze(0)
    verts_uvs = torch.tensor(np.float32(texture_model['vt']), dtype=torch.float32).unsqueeze(0)

    model = TexturedFLAME(texture_shape, texture_num_pc, texture_mean, texture_dir)

    if arg.cuda:
        faces_uvs = faces_uvs.cuda(device=devices[0])
        verts_uvs = verts_uvs.cuda(device=devices[0])
        model = model.cuda(device=devices[0])

    return model, faces_uvs, verts_uvs
