import sys
sys.path.append(".")
import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision.models as models
import cv2
import numpy as np
import math
from functools import reduce
import matplotlib.pyplot as plt

import neural_renderer
from .model_with_hooks import NewModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.autograd.set_detect_anomaly(True)

class StyleTransferModel(nn.Module):
    def __init__(
            self,
            filename_mesh,
            filename_style,
            texture_size=4,
            camera_distance=2.732,
            camera_distance_noise=0.1,
            elevation_min=20,
            elevation_max=40,
            lr_vertices=0.01,
            lr_textures=1.0,
            lambda_style=1,
            lambda_content=2e9,
            lambda_tv=1e7,
            image_size=224,
    ):
        super(StyleTransferModel, self).__init__()
        self.image_size = image_size
        self.camera_distance = camera_distance
        self.camera_distance_noise = camera_distance_noise
        self.elevation_min = elevation_min
        self.elevation_max = elevation_max
        self.lambda_style = lambda_style
        self.lambda_content = lambda_content
        self.lambda_tv = lambda_tv
        self.xp = np

        # load feature extractor
        self.vgg16 = NewModel(output_layers=[2,7,14,21]).to(device)

        # load reference image
        reference_image = cv2.imread(filename_style)
        reference_image = cv2.resize(reference_image, (image_size, image_size))
        reference_image = reference_image.astype('float32') / 255.
        reference_image = reference_image[:, :, :3].transpose((2, 0, 1))[None, :, :, :]
        reference_image = self.xp.array(reference_image)
        with torch.autograd.no_grad():
            features_ref = [f.data for f in self.extract_style_feature(torch.from_numpy(reference_image))]
        self.features_ref = features_ref
        self.background_color = reference_image.mean((0, 2, 3))
        
        # load .obj
        
        #self.mesh = neural_renderer.Mesh.fromobj(filename_mesh, texture_size)
        vertices,faces = neural_renderer.load_obj(filename_mesh)
        shape = (faces.shape[0], texture_size, texture_size, texture_size, 3)
        textures = 0.05*torch.randn(*shape)
        
        self.vertices = nn.Parameter(vertices)
        self.register_buffer('faces', faces)
        self.register_buffer('textures', textures)

        self.vertices_original = self.vertices

        # setup renderer
        renderer = neural_renderer.Renderer()
        renderer.image_size = image_size
        renderer.background_color = self.background_color
        self.renderer = renderer


    def extract_style_feature(self, images, masks=None):
        mean = np.array([103.939, 116.779, 123.68], 'float32')  # BGR
        images = torch.from_numpy(images.cpu().detach().numpy()[:, ::-1]*255 - mean[None, :, None, None])
        images = images.to(device)
        features = self.vgg16.forward(images).values()
        if masks is None:
            masks = torch.ones((images.shape[0], images.shape[2], images.shape[3]))
        style_features = []
        for feature in features:
            scale = int(masks.shape[-1] / feature.shape[-1])
            m = F.avg_pool2d(masks[:, None, :, :], scale, scale).to(device)
            dim = feature.shape[1]

            m = m.reshape((m.shape[0], -1))
            f2 = feature.permute((0, 2, 3, 1))
            f2 = f2.reshape((f2.shape[0], -1, f2.shape[-1]))
            f2 *= torch.sqrt(m)[:, :, None]
            f2 = torch.matmul(f2.permute((0, 2, 1)), f2)
            f2 /= dim * m.sum(axis=1)[:, None, None]
            style_features.append(f2)

        return style_features

    def compute_style_loss(self, features):
        loss = [torch.mean(torch.square(f - fr.expand(f.shape))) for f, fr in zip(features, self.features_ref)]
        loss = reduce(lambda a, b: a + b, loss)
        batch_size = features[0].shape[0]
        loss /= batch_size
        return loss

    def compute_content_loss(self):
        loss = torch.mean(torch.square(self.vertices - self.vertices_original))
        return loss

    def compute_tv_loss(self, images, masks):
        s1 = torch.square(images[:, :, 1:, :-1] - images[:, :, :-1, :-1])
        s2 = torch.square(images[:, :, :-1, 1:] - images[:, :, :-1, :-1])
        masks = masks[:, None, :-1, :-1].expand(s1.shape)        
        masks = masks.data == 1
        return torch.mean(masks * (s1 + s2))

    def __call__(self, batch_size):
        xp = self.xp

        # set random viewpoints
        self.renderer.eye = neural_renderer.get_points_from_angles(
            distance=(
                    torch.from_numpy(xp.ones(batch_size, 'float32') * self.camera_distance +
                    xp.random.normal(size=batch_size).astype('float32') * self.camera_distance_noise)),
            elevation=torch.from_numpy(xp.random.uniform(self.elevation_min, self.elevation_max, batch_size).astype('float32')),
            azimuth=torch.from_numpy(xp.random.uniform(0, 360, size=batch_size).astype('float32')))

        # set random lighting direction
        angles = xp.random.uniform(0, 360, size=batch_size).astype('float32')
        y = xp.ones(batch_size, 'float32') * xp.cos(xp.radians(30).astype('float32'))
        x = xp.ones(batch_size, 'float32') * xp.sin(xp.radians(30).astype('float32')) * xp.sin(xp.radians(angles))
        z = xp.ones(batch_size, 'float32') * xp.sin(xp.radians(30).astype('float32')) * xp.cos(xp.radians(angles))
        directions = xp.concatenate((x[:, None], y[:, None], z[:, None]), axis=1)
        self.renderer.light_direction = directions

        # compute loss
        #x,y,z = self.mesh.get_batch(batch_size)
        
        x = torch.unsqueeze(self.vertices,0)
        x = x.repeat(batch_size,1,1)
        x = x.to(device)

        y = torch.unsqueeze(self.state_dict()['faces'],0)
        y = y.repeat(batch_size,1,1)
        y = y.to(device)

        z = torch.unsqueeze(self.state_dict()['textures'],0)
        z = torch.sigmoid(z.repeat(batch_size,1,1,1,1,1))
        z = z.to(device)

        images,_,_ = self.renderer.render(x,y,z)
        # for image in images:
        #     plt.imshow(image.permute(1,2,0).cpu().detach().numpy())
        #     plt.show()
        masks = self.renderer.render_silhouettes(x,y)
        # for image in masks:
        #     plt.imshow(image.cpu().detach().numpy())
        #     plt.show()
        # import IPython
        # IPython.embed()
        with torch.autograd.no_grad():
            features = self.extract_style_feature(images, masks)

        loss_style = self.compute_style_loss(features)
        loss_content = self.compute_content_loss()
        print('Content Loss=',loss_content)
        loss_tv = self.compute_tv_loss(images, masks)
        loss = self.lambda_style * loss_style + self.lambda_content * loss_content + self.lambda_tv * loss_tv

        # set default lighting direction
        self.renderer.light_direction = [0, 1, 0]

        return loss
