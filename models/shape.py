import torch


class HomogeneousTransformationLayer(torch.nn.Module):
    """
    Module to perform homogeneous transformations in 2D and 3D
    (Implemented in Python)

    """

    def __init__(self, n_dims):
        """

        Parameters
        ----------
        n_dims : int
            number of dimensions

        """
        super().__init__()

        self._n_params = {}

        if n_dims == 2:
            self._n_params["scale"] = 1
            self._n_params["rotation"] = 1
            self._n_params["translation"] = 2
        elif n_dims == 3:
            self._n_params["scale"] = 3
            self._n_params["rotation"] = 3
            self._n_params["translation"] = 3

        homogen_trafo = torch.zeros(1, n_dims +1, n_dims +1)
        homogen_trafo[:, -1, :-1] = 0.
        homogen_trafo[:, -1, -1] = 1.

        self.register_buffer("_trafo_matrix", homogen_trafo)
        self._n_dims = n_dims

        total_params = 0
        for key, val in self._n_params.items():
            self.register_buffer("_indices_%s_params" % key,
                                 torch.arange(total_params, total_params + val)
                                 )
            total_params += val

    def forward(self, shapes: torch.Tensor, params: torch.Tensor):
        """
        Actual prediction

        Parameters
        ----------
        shapes : :class:`torch.Tensor`
            shapes before applied global transformation
        params : :class:`torch.Tensor`
            parameters specifying the global transformation

        Returns
        -------
        :class:`torch.Tensor`
            Transformed shapes

        """

        rotation_params = params.index_select(
            dim=1, index=getattr(self, "_indices_rotation_params")
        )
        scale_params = params.index_select(
            dim=1, index=getattr(self, "_indices_scale_params")
        )
        translation_params = params.index_select(
            dim=1, index=getattr(self, "_indices_translation_params")
        )

        """
                ensembles the homogeneous transformation matrix and applies it to the
                shape tensor

                Parameters
                ----------
                shapes : :class:`torch.Tensor`
                    shapes to transform
                rotation_params : :class:`torch.Tensor`
                    parameters specifying the rotation (one per DoF)
                translation_params : :class:`torch.Tensor`
                    parameters specifying the translation (one per dimension)
                scale_params : :class:`torch.Tensor`
                    parameter specifying the global scaling factor
                    (currently only isotropic scaling supported)

                Returns
                -------
                :class:`torch.Tensor`
                    the transformed shapes in cartesian coordinates

                """

        assert shapes.size(-1) == self._n_dims, "Layer for other " \
                                                "dimensionality specified"

        trafo_matrix = self._ensemble_trafo(rotation_params,
                                            translation_params, scale_params)

        homogen_shapes = torch.cat([shapes,
                                    shapes.new_ones(*shapes.size()[:-1], 1)],
                                   dim=-1)

        transformed_shapes = torch.bmm(homogen_shapes,
                                       trafo_matrix.permute(0, 2, 1))

        transformed_shapes = transformed_shapes[..., :-1]
        # transformed_shapes = transformed_shapes[..., :-1] / transformed_shapes[..., -1].unsqueeze(-1)

        return transformed_shapes

    def _ensemble_trafo(self, rotation_params: torch.Tensor,
                        translation_params: torch.Tensor,
                        scale_params: torch.Tensor):
        """
        ensembles the transformation matrix in 2D and 3D

        Parameters
        ----------
        rotation_params : :class:`torch.Tensor`
            parameters specifying the rotation (one per DoF)
        translation_params : :class:`torch.Tensor`
            parameters specifying the translation (one per dimension)
        scale_params : :class:`torch.Tensor`
            parameter specifying the global scaling factor
            (currently only isotropic scaling supported)

        Returns
        -------
        :class:`torch.Tensor`
            transformation matrix

        """

        rotation_params = rotation_params.view(rotation_params.size()[:2])
        translation_params = translation_params.view(
            translation_params.size()[:2])
        scale_params = scale_params.view(scale_params.size()[:2])

        if self._n_dims == 2:
            return self._ensemble_2d_matrix(rotation_params,
                                            translation_params, scale_params)
        elif self._n_dims == 3:
            return self._ensemble_3d_matrix(rotation_params,
                                            translation_params, scale_params)
        else:
            raise NotImplementedError("Implementation for n_dims = %d "
                                      "not available" % self._n_dims)

    def _ensemble_2d_matrix(self, rotation_params: torch.Tensor,
                            translation_params: torch.Tensor,
                            scale_params: torch.Tensor):
        """
        ensembles the homogeneous transformation matrix for 2D

        Parameters
        ----------
        rotation_params : :class:`torch.Tensor`
            parameters specifying the rotation (one parameter)
        translation_params : :class:`torch.Tensor`
            parameters specifying the translation (two parameters)
        scale_params : :class:`torch.Tensor`
            parameter specifying the global scaling factor (one parameter)
            (currently only isotropic scaling supported)

        Returns
        -------
        :class:`torch.Tensor`
            2D transformation matrix

        """

        homogen_trafo = getattr(self, "_trafo_matrix").repeat(
            scale_params.size(0), 1, 1).clone()

        homogen_trafo[:, 0, 0] = (scale_params *
                                  rotation_params.cos())[:, 0].clone()
        # s*sin\theta
        homogen_trafo[:, 0, 1] = (scale_params *
                                  rotation_params.sin())[:, 0].clone()
        # -s*sin\theta
        homogen_trafo[:, 1, 0] = (-scale_params *
                                  rotation_params.sin())[:, 0].clone()
        # s*cos\theta
        homogen_trafo[:, 1, 1] = (scale_params *
                                  rotation_params.cos())[:, 0].clone()

        # translation params
        homogen_trafo[:, :-1, -1] = translation_params.clone()

        return homogen_trafo

    def _ensemble_3d_matrix(self, rotation_params: torch.Tensor,
                            translation_params: torch.Tensor,
                            scale_params: torch.Tensor):
        """
        ensembles the homogeneous transformation matrix for 3D

        Parameters
        ----------
        rotation_params : :class:`torch.Tensor`
            parameters specifying the rotation (three parameters)
        translation_params : :class:`torch.Tensor`
            parameters specifying the translation (three parameters)
        scale_params : :class:`torch.Tensor`
            parameter specifying the global scaling factor (one parameter)
            (currently only isotropic scaling supported)

        Returns
        -------
        :class:`torch.Tensor`
            3D transformation matrix

        """

        homogen_trafo = getattr(self, "_trafo_matrix").repeat(
            scale_params.size(0), 1, 1).clone()

        roll = rotation_params[:, 2].unsqueeze(-1)
        pitch = rotation_params[:, 1].unsqueeze(-1)
        yaw = rotation_params[:, 0].unsqueeze(-1)

        # Note that the elements inside the transformation matrix are swapped
        # due to the zyx convention

        # s*(cos(pitch)*cos(roll))
        homogen_trafo[:, 0, 0] = (scale_params *
                                  (pitch.cos() * roll.cos()))[:, 0].clone()

        # s*(cos(pitch)*sin(roll))
        homogen_trafo[:, 0, 1] = (scale_params *
                                  (pitch.cos() * roll.sin()))[:, 0].clone()

        # s*(-sin(pitch))
        homogen_trafo[:, 0, 2] = (scale_params *
                                  (-pitch.sin()))[:, 0].clone()

        # s*(sin(yaw)*sin(pitch)*cos(roll) - cos(yaw)*sin(roll))
        homogen_trafo[:, 1, 0] = (scale_params *
                                  (yaw.sin() * pitch.sin() * roll.cos() -
                                   yaw.cos() * roll.sin()))[:, 0].clone()

        # s*(sin(yaw)*sin(pitch)*sin(roll) + cos(yaw)*cos(roll))
        homogen_trafo[:, 1, 1] = (scale_params *
                                  (yaw.sin() * pitch.sin() * roll.sin() +
                                   yaw.cos() * roll.cos()))[:, 0].clone()

        # s*(sin(yaw)*cos(pitch))
        homogen_trafo[:, 1, 2] = (scale_params *
                                  (yaw.sin() * pitch.cos()))[:, 0].clone()

        # s*(cos(yaw)*sin(pitch)*cos(roll) + sin(yaw)*sin(roll))
        homogen_trafo[:, 2, 0] = (scale_params *
                                  (yaw.cos() * pitch.sin() * roll.cos() +
                                   yaw.sin() * roll.sin()))[:, 0].clone()

        # s*(cos(yaw)*sin(pitch)*sin(roll)-sin(yaw)*cos(roll))
        homogen_trafo[:, 2, 1] = (scale_params *
                                  (yaw.cos() * pitch.sin() * roll.sin() -
                                   yaw.sin() * roll.cos()))[:, 0].clone()

        # s*(cos(yaw)*cos(pitch))
        homogen_trafo[:, 2, 2] = (scale_params *
                                  (yaw.cos() * pitch.cos()))[:, 0].clone()

        # translation params
        homogen_trafo[:, :-1, -1] = translation_params.clone()

        return homogen_trafo

    @property
    def num_params(self):
        num_params = 0
        for key, val in self._n_params.items():
            num_params += val

        return num_params


class ShapeLayer(torch.nn.Module):
    """
    Python Implementation of Shape Layer

    """
    def __init__(self, shapes):
        """

        Parameters
        ----------
        shapes : np.ndarray
            eigen shapes (obtained by PCA)

        """
        super().__init__()

        self.register_buffer("_shape_mean", torch.from_numpy(
            shapes[0]).float().unsqueeze(0))
        components = []
        for i, _shape in enumerate(shapes[1:]):
            components.append(torch.from_numpy(
                _shape).float().unsqueeze(0))

        component_tensor = torch.cat(components).unsqueeze(0)
        self.register_buffer("_shape_components", component_tensor)

    def forward(self, shape_params: torch.Tensor):
        """
        Ensemble shape from parameters

        Parameters
        ----------
        shape_params : :class:`torch.Tensor`
            shape parameters

        Returns
        -------
        :class:`torch.Tensor`
            ensembled shape

        """

        shapes = getattr(self, "_shape_mean").clone()
        shapes = shapes.expand(shape_params.size(0), *shapes.size()[1:])

        components = getattr(self, "_shape_components")
        components = components.expand(shape_params.size(0),
                                       *components.size()[1:])

        weighted_components = components.mul(
            shape_params.expand_as(components))

        shapes = shapes.add(weighted_components.sum(dim=1))

        return shapes

    @property
    def num_params(self):
        """
        Property to access these layer's parameters

        Returns
        -------
        int
            number of parameters

        """
        return getattr(self, "_shape_components").size(1)


class HomogeneousShapeLayer(torch.nn.Module):
    """
    Module to Perform a Shape Prediction
    (including a global homogeneous transformation)

    """
    def __init__(self, shapes, n_dims):
        """

        Parameters
        ----------
        shapes : np.ndarray
            shapes to construct a :class:`ShapeLayer`
        n_dims : int
            number of shape dimensions
        use_cpp : bool
            whether or not to use (experimental) C++ Implementation

        See Also
        --------
        :class:`ShapeLayer`
        :class:`HomogeneousTransformationLayer`

        """
        super().__init__()

        self._shape_layer = ShapeLayer(shapes)
        self._homogen_trafo = HomogeneousTransformationLayer(n_dims)

        self.register_buffer("_indices_shape_params",
                             torch.arange(self._shape_layer.num_params))
        self.register_buffer("_indices_homogen_params",
                             torch.arange(self._shape_layer.num_params,
                                          self.num_params))

    def forward(self, params: torch.Tensor):
        """
        Performs the actual prediction

        Parameters
        ----------
        params : :class:`torch.Tensor`
            input parameters

        Returns
        -------
        :class:`torch.Tensor`
            predicted shape

        """

        shape_params = params.index_select(
            dim=1, index=getattr(self, "_indices_shape_params")
        )

        transformation_params = params.index_select(
            dim=1, index=getattr(self, "_indices_homogen_params")
        )
        shapes = self._shape_layer(shape_params)
        transformed_shapes = self._homogen_trafo(shapes, transformation_params)

        return transformed_shapes

    @property
    def num_params(self):
        """
        Property to access these layer's number of parameters

        Returns
        -------
        int
            number of parameters

        """
        return self._shape_layer.num_params + self._homogen_trafo.num_params


class Cnv2dRelu(torch.nn.Module):
    """
    Block holding one Conv2d and one ReLU layer
    """
    def __init__(self, *args, **kwargs):
        """

        Parameters
        ----------
        *args :
            positional arguments (passed to Conv2d)
        **kwargs :
            keyword arguments (passed to Conv2d)

        """
        super().__init__()
        self._conv = torch.nn.Conv2d(*args, **kwargs)
        self._relu = torch.nn.ReLU()

    def forward(self, input_batch):
        """
        Forward batch though layers

        Parameters
        ----------
        input_batch : :class:`torch.Tensor`
            input batch

        Returns
        -------
        :class:`torch.Tensor`
            result
        """
        return self._relu(self._conv(input_batch))


class DeCnv2dRelu(torch.nn.Module):
    """
    Block holding one Conv2d and one ReLU layer
    """
    def __init__(self, *args, **kwargs):
        """

        Parameters
        ----------
        *args :
            positional arguments (passed to Conv2d)
        **kwargs :
            keyword arguments (passed to Conv2d)

        """
        super().__init__()
        self._conv = torch.nn.ConvTranspose2d(*args, **kwargs)
        self._relu = torch.nn.ReLU()

    def forward(self, input_batch):
        """
        Forward batch though layers

        Parameters
        ----------
        input_batch : :class:`torch.Tensor`
            input batch

        Returns
        -------
        :class:`torch.Tensor`
            result
        """
        return self._relu(self._conv(input_batch))


class FeatureExtractor(torch.nn.Module):

    def __init__(self, in_channels, out_params):
        super().__init__()

        """
               Build the actual model structure
    
               Parameters
               ----------
               in_channels : int
                   number of input channels
               out_params : int
                   number of outputs
               norm_class : Any
                   class implementing a normalization
               p_dropout : float
                   dropout probability
    
               Returns
               -------
               :class:`torch.nn.Module`
                   ensembled model
    
               """
        self.model = torch.nn.Sequential()

        self.model.add_module("conv_1", Cnv2dRelu(in_channels, 64, (5, 1)))
        self.model.add_module("conv_2", Cnv2dRelu(64, 64, (1, 5)))

        self.model.add_module("down_conv_1", Cnv2dRelu(64, 128, (5, 5), stride=2))
        self.model.add_module('norm_1', torch.nn.InstanceNorm2d(128))

        self.model.add_module("conv_3", Cnv2dRelu(128, 128, (5, 1)))
        self.model.add_module("conv_4", Cnv2dRelu(128, 128, (1, 5)))

        self.model.add_module("down_conv_2", Cnv2dRelu(128, 256, (5, 5), stride=2))
        self.model.add_module('norm_2', torch.nn.InstanceNorm2d(256))

        self.model.add_module("conv_5", Cnv2dRelu(256, 256, (3, 1)))
        self.model.add_module("conv_6", Cnv2dRelu(256, 256, (1, 3)))

        self.model.add_module("down_conv_3", Cnv2dRelu(256, 128, (3, 3), stride=2))
        self.model.add_module('norm_3', torch.nn.InstanceNorm2d(128))

        self.model.add_module("conv_7", Cnv2dRelu(128, 128, (2, 1)))
        self.model.add_module("conv_8", Cnv2dRelu(128, 128, (1, 2)))
        # self.model.add_module("conv_9", Cnv2dRelu(128, 128, (2, 1)))
        # self.model.add_module("conv_10", Cnv2dRelu(128, 128, (1, 2)))

        self.model.add_module("final_conv", Cnv2dRelu(128, out_params, (2, 2), stride=2))

        # self.model.add_module("conv_7", Conv2dRelu(in_channels, in_channels*out_params, (3, 1)))
        # self.model.add_module("conv_8", Conv2dRelu(in_channels*out_params, in_channels*out_params, (1, 3)))
        #
        # self.model.add_module("down_conv_4", Conv2dRelu(in_channels*out_params, out_params, (1, 1)))
        #
        # self.model.add_module("conv_9", Conv2dRelu(128, 128, (3, 1)))
        # self.model.add_module("conv_10", Conv2dRelu(128, 128, (1, 3)))
        #
        # self.model.add_module("conv_11", Conv2dRelu(128, 128, (3, 1)))
        # self.model.add_module("conv_12", Conv2dRelu(128, 128, (1, 3)))

        # self.model.add_module("final_conv", torch.nn.Conv2d(128, out_params,
        #                                                (2, 2)))

    def forward(self, input):
        return self.model(input)

class ShapeGenerator(torch.nn.Module):

    def __init__(self, num_keypoints, out_channels):
        super().__init__()
        self.model = torch.nn.Sequential()

        self.model.add_module("up_deconv_1", DeCnv2dRelu(num_keypoints, num_keypoints, (1, 2)))

        self.model.add_module("deconv_1", DeCnv2dRelu(num_keypoints, num_keypoints, (1, 2)))
        self.model.add_module("deconv_2", DeCnv2dRelu(num_keypoints, num_keypoints, (2, 1)))

        self.model.add_module("up_deconv_2", DeCnv2dRelu(num_keypoints, num_keypoints * 2, (5, 5)))

        self.model.add_module("deconv_3", DeCnv2dRelu(num_keypoints * 2, num_keypoints * 4, (1, 5)))
        self.model.add_module("deconv_4", DeCnv2dRelu(num_keypoints * 4, num_keypoints * 4, (5, 1)))

        self.model.add_module("up_deconv_3", DeCnv2dRelu(num_keypoints * 4, num_keypoints * 2, (5, 5)))

        self.model.add_module("deconv_5", DeCnv2dRelu(num_keypoints * 2, num_keypoints, (1, 9)))
        self.model.add_module("deconv_6", DeCnv2dRelu(num_keypoints, num_keypoints, (9, 1)))

        self.model.add_module("up_deconv_4", DeCnv2dRelu(num_keypoints, num_keypoints, (9, 9)))

        self.model.add_module("deconv_7", DeCnv2dRelu(num_keypoints, num_keypoints, (1, 15)))
        self.model.add_module("deconv_8", DeCnv2dRelu(num_keypoints, num_keypoints, (15, 1)))

        self.model.add_module("up_deconv_5", DeCnv2dRelu(num_keypoints, out_channels, (20, 20)))

        # self.model.add_module("deconv_7", DeConv2dRelu(in_channels, in_channels // 6, (1, 16)))
        # self.model.add_module("deconv_8", DeConv2dRelu(in_channels // 6, in_channels // 6, (16, 1)))
        #
        # self.model.add_module("up_deconv_5", DeConv2dRelu(in_channels // 6, in_channels // 6, (20, 20)))

        # self.model.add_module("conv_3", Conv2dRelu(in_channels * 4, in_channels * 8, (1, 4)))
        # self.model.add_module("conv_4", Conv2dRelu(in_channels * 8, in_channels * 8, (4, 1)))

        # self.model.add_module("conv_1", Conv2dRelu(in_channels, in_channels*4, (7, 1)))
        # self.model.add_module("conv_2", Conv2dRelu(in_channels*4, in_channels*4, (1, 7)))
        #
        # self.model.add_module("down_conv_1", Conv2dRelu(in_channels*4, in_channels*8, (7, 7), stride=2))
        #
        # self.model.add_module("conv_3", Conv2dRelu(in_channels*8, in_channels*8, (5, 1)))
        # self.model.add_module("conv_4", Conv2dRelu(in_channels*8, in_channels*8, (1, 5)))
        #
        # self.model.add_module("down_conv_2", Conv2dRelu(in_channels*8, in_channels*16, (5, 5), stride=2))
        #
        # self.model.add_module("conv_5", Conv2dRelu(in_channels*16, in_channels*16, (3, 1)))
        # self.model.add_module("conv_6", Conv2dRelu(in_channels*16, in_channels*16, (1, 3)))
        #
        #
        #


    def forward(self, input):
        return self.model(input)


class Transformer(torch.nn.Module):

    def __init__(self, encoder_layer, shape_layer, generator_layer, num_keypoints) -> None:
        super().__init__()
        self.shape_layer = shape_layer
        self.encoder_layer = encoder_layer
        self.generator_layer = generator_layer
        self.gen_in_channels = num_keypoints


    def forward(self, input):
        embeddings = self.encoder_layer(input)
        shape = self.shape_layer(embeddings)
        shape_view = shape.view(shape.size(0), self.gen_in_channels, 2, 1)
        output = self.generator_layer(shape_view)
        return shape, output

