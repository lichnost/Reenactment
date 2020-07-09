from utils import *
from utils.args import parse_args
from flame.FLAME import FLAME
from types import SimpleNamespace

import pyrender
from pytorch3d.structures import Meshes, Textures
from pytorch3d.renderer import OpenGLPerspectiveCameras, OpenGLOrthographicCameras, look_at_view_transform,\
    RasterizationSettings, BlendParams, MeshRenderer, MeshRasterizer, HardPhongShader, HardFlatShader,\
    SoftSilhouetteShader, TexturedSoftPhongShader, PointLights
from pytorch3d.utils import ico_sphere


def main(arg):
    devices = get_devices_list(arg)

    flame_conf = SimpleNamespace()
    flame_conf.flame_model_path = '/home/lichnost/programming/work/ml/head/FLAME/model/female_model.pkl'
    flame_conf.use_face_contour = True
    flame_conf.batch_size = 2
    flame_conf.shape_params = 100
    flame_conf.expression_params = 50
    # flame_conf.pose_params = 6
    flame_conf.use_3D_translation = False
    flame_conf.static_landmark_embedding_path = '/home/lichnost/programming/work/ml/head/FLAME/model/flame_static_embedding.pkl'
    flame_conf.dynamic_landmark_embedding_path = '/home/lichnost/programming/work/ml/head/FLAME/model/flame_dynamic_embedding.npy'
    texture_path = '/home/lichnost/programming/work/ml/head/FLAME/model/FLAME_texture.npz'
    # texture_path = '/home/lichnost/programming/work/ml/head/FLAME/model/albedoModel2020_FLAME_albedoPart.npz'

    flame = FLAME(flame_conf).to(devices[0])

    shape_params = torch.zeros(flame_conf.batch_size, flame_conf.shape_params, dtype=torch.float32).to(devices[0])

    radian = np.pi/180.0
    pose_params_numpy = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                  [0.0 * radian, 0.0 * radian, 0.0, 0.0, 0.0, 0.0]
                                  # [0.0, 85.0*radian, 0.0, 0.0, 0.0, 0.0]
                                  # [0.0, -48.0*radian, 0.0, 0.0, 0.0, 0.0],
                                  # [0.0, 10.0*radian, 0.0, 0.0, 0.0, 0.0],
                                  # [0.0, -15.0*radian, 0.0, 0.0, 0.0, 0.0],
                                  # [0.0, 0.0*radian, 0.0, 0.0, 0.0, 0.0],
                                  # [0.0, -0.0*radian, 0.0, 0.0, 0.0, 0.0]
                                  ], dtype=np.float32)
    pose_params = torch.tensor(pose_params_numpy, dtype=torch.float32).to(devices[0])

    neck_pose_params_numpy = np.array([[0.0, 0.0, 0.0],
                                       [0.0, 0.0, 0.0]
                                  ], dtype=np.float32)
    neck_pose_params = torch.tensor(neck_pose_params_numpy, dtype=torch.float32).to(devices[0])

    # Cerating a batch of neutral expressions
    expression_params = torch.zeros(flame_conf.batch_size, flame_conf.expression_params, dtype=torch.float32).to(devices[0])

    vertices, landmarks = flame(shape_params, expression_params, pose_params, neck_pose_params)

    faces = torch.from_numpy(np.float32(flame.faces)).to(devices[0])
    faces = torch.cat(vertices.shape[0] * [faces.unsqueeze(0)]).to(torch.int64)

    texture_model = np.load(texture_path)

    # FLAME texture model
    texture_shape = texture_model['mean'].shape
    texture_num_pc = texture_model['tex_dir'].shape[-1]
    texture_mean = np.reshape(texture_model['mean'], (1, -1))
    texture_dir = np.reshape(texture_model['tex_dir'], (-1, texture_num_pc)).T
    texture_params = np.random.randn(texture_num_pc)[np.newaxis, :]
    texture = np.clip(np.reshape(np.add(texture_mean, np.matmul(texture_params, texture_dir)), texture_shape), 0.0, 255.0)
    texture = texture / 255.0

    # Albedo Morphable Model
    # texture_num_pc = texture_model['PC'].shape[-1]
    # texture_shape = texture_model['MU'].shape
    #
    # texture_mu = np.reshape(texture_model['MU'], (1, -1))
    # texture_params = np.random.randn(texture_num_pc)[np.newaxis, :]
    #
    # tf_MU = tf.Variable(np.reshape(texture_model['MU'], (1, -1)), name='MU', dtype=tf.float64, trainable=False)
    # tf_PC = tf.Variable(np.reshape(texture_model['PC'], (-1, num_tex_pc)).T, name='PC', dtype=tf.float64,
    #                     trainable=False)
    # tf_specMU = tf.Variable(np.reshape(texture_model['specMU'], (1, -1)), name='specMU', dtype=tf.float64,
    #                         trainable=False)
    # tf_specPC = tf.Variable(np.reshape(texture_model['specPC'], (-1, num_tex_pc)).T, name='specPC', dtype=tf.float64,
    #                         trainable=False)
    #
    # tf_diff_albedo = tf.add(tf_MU, tf.matmul(tf_tex_params, tf_PC))
    # tf_spec_albedo = tf.add(tf_specMU, tf.matmul(tf_tex_params, tf_specPC))
    # tf_tex = 255 * tf.math.pow(0.6 * tf.add(tf_diff_albedo, tf_spec_albedo), 1.0 / 2.2)

    faces_uvs = texture_model['ft']
    verts_uvs = texture_model['vt']

    # show_img(texture.astype(np.uint8))

    # colors = torch.ones_like(vertices).to(devices[0])  # (1, V, 3)
    # textures = Textures(verts_rgb=colors)
    texture = torch.cat(flame_conf.batch_size * [torch.tensor(texture, dtype=torch.float32, device=devices[0]).unsqueeze(0)])
    faces_uvs = torch.cat(flame_conf.batch_size * [torch.tensor(np.int64(faces_uvs), dtype=torch.int64, device=devices[0]).unsqueeze(0)])
    verts_uvs = torch.cat(flame_conf.batch_size * [torch.tensor(np.float32(verts_uvs), dtype=torch.float32, device=devices[0]).unsqueeze(0)])

    textures = Textures(maps=texture, faces_uvs=faces_uvs, verts_uvs=verts_uvs)
    meshes = Meshes(vertices, faces, textures)
    # meshes = meshes.scale_verts(100000)

    R, T = look_at_view_transform(0.3, 0.5, 0, device=devices[0])
    scale = ((5.0, 5.0, 5.0),)
    camera = OpenGLOrthographicCameras(R=R, T=T, scale_xyz=scale, device=devices[0])

    # blend_params = BlendParams(sigma=1e-4, gamma=1e-4)
    #
    # blur_radius = np.log(1. / 1e-4 - 1.) * blend_params.sigma
    # raster_settings = RasterizationSettings(
    #     image_size=arg.crop_size,
    #     blur_radius=blur_radius,
    #     faces_per_pixel=1
    # )
    #
    # renderer = MeshRenderer(
    #     rasterizer=MeshRasterizer(cameras=camera, raster_settings=raster_settings),
    #     shader=SoftSilhouetteShader(blend_params=blend_params)
    # )

    raster_settings = RasterizationSettings(
        image_size=arg.crop_size,
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=None,
        max_faces_per_bin=None
    )

    lights = PointLights(location=[[0.0, 0.0, -2.0]], device=devices[0])

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=camera, raster_settings=raster_settings),
        shader=TexturedSoftPhongShader(cameras=camera, device=devices[0], lights=lights)
    )

    images = renderer(meshes)

    image = images.cpu().numpy()[0, ..., :3]
    show_img(image, name='render', keep=True)

    # transform = camera.get_full_projection_transform()
    # landmark_trn = transform.transform_points(landmarks)[0].cpu().numpy()

    landmark_trn = camera.transform_points(landmarks)
    landmark_trn[:, :, 0] *= -1
    landmark_trn[:, :, 1] *= -1

    def ndc_to_pix(i, S):
        return ((i + 1) * S - 1.0) / 2.0

    for bi in range(landmark_trn.shape[0]):
        for pi in range(landmark_trn.shape[1]):
            landmark_trn[bi, pi, 0] = ndc_to_pix(landmark_trn[bi, pi, 0], arg.crop_size)
            landmark_trn[bi, pi, 1] = ndc_to_pix(landmark_trn[bi, pi, 1], arg.crop_size)

    landmark_trn = landmark_trn[0].cpu().numpy()

    img = np.zeros((arg.crop_size, arg.crop_size, 3), dtype=np.uint8)
    for i in range(0, landmark_trn.shape[0]):
        xy = (int(landmark_trn[i, 0]), int(landmark_trn[i, 1]))
        draw_circle(img, xy, color=(0, 255, 0))  # green
        draw_text(img, str(i), xy, color=(0, 0, 255), scale=0.25)
    show_img(img)
    pass



if __name__ == '__main__':
    arg = parse_args()
    main(arg)