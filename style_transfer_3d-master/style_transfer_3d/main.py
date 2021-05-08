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
import neural_renderer
from .model_with_hooks import FeatureExtractor

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
        self.vgg16 = FeatureExtractor(layers=[0,2,5,7]).to(device)

        # load reference image
        reference_image = cv2.imread(filename_style)
        reference_image = cv2.resize(reference_image, (image_size, image_size))
        reference_image = reference_image.astype('float32') / 255.
        reference_image = reference_image[:, :, :3].transpose((2, 0, 1))[None, :, :, :]
        reference_image = self.xp.array(reference_image)
        with torch.autograd.no_grad():
            features_ref = [f.data for f in self.extract_style_feature(reference_image)]
        self.features_ref = features_ref
        self.background_color = reference_image.mean((0, 2, 3))
        # load .obj
        self.mesh = neural_renderer.Mesh.fromobj(filename_mesh, texture_size)
        self.vertices_original = self.mesh.vertices

        # setup renderer
        renderer = neural_renderer.Renderer()
        renderer.image_size = image_size
        renderer.background_color = self.background_color
        self.renderer = renderer

    def extract_style_feature(self, images, masks=None):
        xp = self.xp
        images -= images.mean(axis=(-2,-1),keepdims=1)
        images = torch.from_numpy(images)
        images = images.to(device)
        features = self.vgg16.forward(images)
        if masks is None:
            masks = torch.ones((images.shape[0], images.shape[2], images.shape[3]))
        # print('Features=',len(features))
        # for key,value in features.items():
        #     print('key ',key)
        #     print('value ',value.shape)
        style_features = []
        for key,feature in features.items():
            scale = math.ceil(masks.shape[-1] / feature.shape[-1])
            m = F.avg_pool2d(masks[:, None, :, :], scale, scale).to(device)
            dim = feature.shape[0]

            m = m.reshape((m.shape[0], -1))
            f2 = feature.permute((0, 2, 3, 1))
            f2 = f2.reshape((f2.shape[0], -1, f2.shape[-1]))
            f2 *= torch.sqrt(m)[:, :, None]
            f2 = torch.matmul(f2.permute((0, 2, 1)), f2)
            f2 /= dim * m.sum(axis=1)[:, None, None]
            style_features.append(f2)

        return style_features

    def compute_style_loss(self, features):
        loss = [torch.sum(torch.square(f - fr.expand(f.shape))) for f, fr in zip(features, self.features_ref)]
        loss = reduce(lambda a, b: a + b, loss)
        batch_size = features[0].shape[0]
        loss /= batch_size
        return loss

    def compute_content_loss(self):
        loss = torch.sum(torch.square(self.mesh.vertices - self.vertices_original))
        return loss

    def compute_tv_loss(self, images, masks):
        s1 = torch.square(images[:, :, 1:, :-1] - images[:, :, :-1, :-1])
        s2 = torch.square(images[:, :, :-1, 1:] - images[:, :, :-1, :-1])
        masks = masks[:, None, :-1, :-1].expand(s1.shape)        
        masks = masks.data == 1
        return torch.sum(masks * (s1 + s2))

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
        x,y,z = self.mesh.get_batch(batch_size)
        x = x.to(device)
        y = y.to(device)
        z = z.to(device)
        images,_,_ = self.renderer.render(x,y,z)
        masks = self.renderer.render_silhouettes(*self.mesh.get_batch(batch_size)[:2])
        # import IPython
        # IPython.embed()
        features = self.extract_style_feature(images.cpu().detach().numpy(), masks)
        loss_style = self.compute_style_loss(features)
        loss_content = self.compute_content_loss()
        loss_tv = self.compute_tv_loss(images, masks)
        loss = self.lambda_style * loss_style + self.lambda_content * loss_content + self.lambda_tv * loss_tv

        # set default lighting direction
        self.renderer.light_direction = [0, 1, 0]
        print(loss)
        return loss
