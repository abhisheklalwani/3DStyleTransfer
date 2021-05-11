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
import imageio
from skimage.transform import resize
from PIL import Image
from skimage.io import imread, imsave
import neural_renderer
import torchvision.transforms as transforms
from torchvision.utils import save_image
from .StyleLoss import get_style_model_and_losses, image_loader, run_style_transfer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
            lambda_content=1e5,
            lambda_tv=1e2,
            image_size=256,
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

        # load reference image
        self.reference_image = imread(filename_style).astype('float32') / 255.
        #self.reference_image = cv2.resize(self.reference_image,(image_size,image_size))
        self.reference_image = torch.from_numpy(self.reference_image).permute(2,0,1)[None, ::].to(device)
        self.cnn = models.vgg19(pretrained=True).features.to(device).eval()
        for param in self.cnn.parameters():
            param.requires_grad = False
        self.cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
        self.cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
        self.style_layers = ['conv_2', 'conv_4', 'conv_7', 'conv_10']

        self.background_color = self.reference_image.mean((0, 2, 3))
        
        # load .obj        
        vertices,faces = neural_renderer.load_obj(filename_mesh)
        self.vertices_original = torch.clone(vertices[None,:,:]).detach()
        self.vertices = nn.Parameter(vertices[None,:,:])
        
        self.register_buffer('faces', faces[None, :, :])

        texture_size = 4
        textures = torch.zeros(1, self.faces.shape[1], texture_size, texture_size, texture_size, 3, dtype=torch.float32)
        self.textures = nn.Parameter(textures)

        # setup renderer
        renderer = neural_renderer.Renderer(camera_mode='look_at')
        renderer.perspective = False
        renderer.light_intensity_directional = 0.0
        renderer.light_intensity_ambient = 1.0
        self.renderer = renderer

    def image_loader(image_name,imsize,device):
        loader = transforms.Compose([transforms.Resize((imsize,imsize)),transforms.ToTensor()])
        image = Image.open(image_name)
        # fake batch dimension required to fit network's input dimensions
        image = loader(image).unsqueeze(0)
        return image.to(device, torch.float)

    def compute_style_loss(self, images):
        style_loss = 0
        # print('Reference Image',self.reference_image)
        # print('Content Image',images[0])
        # plt.imshow(self.reference_image[0].permute(1,2,0).cpu().detach().numpy())
        # plt.show()
        # plt.imshow(images[0].permute(1,2,0).cpu().detach().numpy())
        # plt.show()
        for image in images:
            image = image.unsqueeze(0)
            style_loss += (run_style_transfer(self.cnn,self.cnn_normalization_mean,self.cnn_normalization_std,image,self.reference_image,self.style_layers))*100000
            style_loss += torch.sum((image - self.reference_image) ** 2)
        return style_loss

    def compute_content_loss(self):
        loss = torch.sum(torch.square(self.vertices - self.vertices_original))
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
        self.renderer.eye = neural_renderer.get_points_from_angles(2.732, 0, np.random.uniform(0, 360))

        # compute loss
        
        x = self.vertices.repeat(batch_size,1,1)
        x = x.to(device)

        y = self.faces.repeat(batch_size,1,1)
        y = y.to(device)

        z = self.textures.repeat(batch_size,1,1,1,1,1)
        z = z.to(device)

        images = self.renderer.render_rgb(x,y,torch.tanh(z))
        #images = torch.flip(images,dims=[1])
        # for image in images:
        #     plt.imshow(image.permute(1,2,0).cpu().detach().numpy())
        #     plt.show()
        masks = self.renderer.render_silhouettes(x,y)
        # for image in masks:
        #     plt.imshow(image.cpu().detach().numpy())
        #     plt.show()
        #     break
        # import IPython
        # IPython.embed()
        loss_style = self.compute_style_loss(images)
        loss_content = self.compute_content_loss()
        loss_tv = self.compute_tv_loss(images, masks)
        # print('Content weight=',self.lambda_content)
        # print('Content Loss=',loss_content)
        # print('Style weight=',self.lambda_style)
        # print('Style Loss=',loss_style)
        # print('TV Loss=',loss_tv)

        # print('Content Loss (UPGRADED)=',self.lambda_content*loss_content)
        # print('Style weight=',self.lambda_style)
        # print('Style Loss (UPGRADED)=',self.lambda_style*loss_style)
        # print('TV Loss=(UPGRADED)',self.lambda_tv*loss_tv)
        
        loss = self.lambda_style * loss_style + self.lambda_content * loss_content + self.lambda_tv * loss_tv
        # set default lighting direction

        return loss
