from utils import *
from utils.args import parse_args
from flame.FLAME import FLAME
from types import SimpleNamespace

import pyrender
import trimesh


def main(arg):
    devices = get_devices_list(arg)

    flame_conf = SimpleNamespace()
    flame_conf.flame_model_path = '/home/lichnost/programming/work/ml/head/FLAME/FaceGeometryTorch/model/generic_model.pkl'
    flame_conf.use_face_contour = True
    flame_conf.batch_size = 1
    flame_conf.shape_params = 30
    flame_conf.expression_params = 10
    flame_conf.use_3D_translation = False
    flame_conf.static_landmark_embedding_path = '/home/lichnost/programming/work/ml/head/FLAME/FaceGeometryTorch/model/flame_static_embedding.pkl'
    flame_conf.dynamic_landmark_embedding_path = '/home/lichnost/programming/work/ml/head/FLAME/FaceGeometryTorch/model/flame_dynamic_embedding.npy'

    flame = FLAME(flame_conf)
    if arg.cuda:
        flame = flame.cuda(devices[0])

    shape_params = torch.zeros(flame_conf.batch_size, flame_conf.shape_params, dtype=torch.float32)
    if arg.cuda:
        shape_params = shape_params.cuda(devices[0])

    radian = np.pi/180.0
    pose_params = torch.tensor([[0.0, 30.0 * radian, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    if arg.cuda:
        pose_params = pose_params.cuda(devices[0])

    # Cerating a batch of neutral expressions
    expression_params = torch.zeros(flame_conf.batch_size, flame_conf.expression_params, dtype=torch.float32)
    if arg.cuda:
        expression_params = expression_params.cuda(devices[0])

    vertice, landmark = flame(shape_params, expression_params, pose_params)  # For RingNet project

    i = 0
    vertices = vertice[i].detach().cpu().numpy().squeeze()
    joints = landmark[i].detach().cpu().numpy().squeeze()
    vertex_colors = np.ones([vertices.shape[0], 4]) * [0.3, 0.3, 0.3, 0.8]
    faces = flame.faces

    tri_mesh = trimesh.Trimesh(vertices, faces,
                                vertex_colors=vertex_colors)
    mesh = pyrender.Mesh.from_trimesh(tri_mesh)
    scene = pyrender.Scene()
    scene.add(mesh)
    sm = trimesh.creation.uv_sphere(radius=0.001)
    sm.visual.vertex_colors = [0.9, 0.1, 0.1, 1.0]
    tfs = np.tile(np.eye(4), (len(joints), 1, 1))
    tfs[:, :3, 3] = joints
    joints_pcl = pyrender.Mesh.from_trimesh(sm, poses=tfs)
    scene.add(joints_pcl)
    pyrender.Viewer(scene, use_raymond_lighting=True)

    scale = torch.tensor([10], dtype=landmark.dtype, device=landmark.device)

    cam_eye = torch.eye(2, m=3, dtype=landmark.dtype, device=landmark.device)
    mul_res = torch.mm(cam_eye, landmark[0].t()).t()
    landmark_2d = mul_res * scale.expand_as(mul_res)
    pass



if __name__ == '__main__':
    arg = parse_args()
    main(arg)