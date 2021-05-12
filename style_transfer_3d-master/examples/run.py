#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import glob
import os
import subprocess
import torch
import cv2

import neural_renderer
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import imageio
from skimage.io import imread, imsave
from texture_mapping import MapTexture
import style_transfer_3d
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def make_gif(filename):
    with imageio.get_writer(filename, mode='I') as writer:
        for filename in sorted(glob.glob('/tmp/_tmp_*.png')):
            writer.append_data(imageio.imread(filename))
            os.remove(filename)
    writer.close()


def run():
    # load settings
    parser = argparse.ArgumentParser()
    parser.add_argument('-im', '--filename_mesh', type=str)
    parser.add_argument('-is', '--filename_style', type=str)
    parser.add_argument('-o', '--filename_output', type=str)
    parser.add_argument('-ls', '--lambda_style', type=float, default=1)
    parser.add_argument('-lc', '--lambda_content', type=float, default=1e5)
    parser.add_argument('-ltv', '--lambda_tv', type=float, default=1e2)
    parser.add_argument('-emax', '--elevation_max', type=float, default=40.)
    parser.add_argument('-emin', '--elevation_min', type=float, default=20.)
    parser.add_argument('-lrv', '--lr_vertices', type=float, default=0.1)
    parser.add_argument('-lrt', '--lr_textures', type=float, default=0.001)
    parser.add_argument('-cd', '--camera_distance', type=float, default=2.732)
    parser.add_argument('-cdn', '--camera_distance_noise', type=float, default=0.1)
    parser.add_argument('-ts', '--texture_size', type=int, default=4)
    parser.add_argument('-lr', '--adam_lr', type=float, default=0.05)
    parser.add_argument('-ab1', '--adam_beta1', type=float, default=0.5)
    parser.add_argument('-ab2', '--adam_beta2', type=float, default=0.999)
    parser.add_argument('-bs', '--batch_size', type=int, default=4)
    parser.add_argument('-ni', '--num_iteration', type=int, default=100)
    parser.add_argument('-g', '--gpu', type=int, default=0)
    parser.add_argument('-rd', '--result_directory', type=str, default='./examples/data/results')
    args = parser.parse_args()

    # create output directory
    directory_output = os.path.dirname(args.filename_output)
    if not os.path.exists(directory_output):
        os.makedirs(directory_output)

    np.random.seed(0)

    # setup scene
    model = style_transfer_3d.StyleTransferModel(
        filename_mesh=args.filename_mesh,
        filename_style=args.filename_style,
        lambda_style=args.lambda_style,
        lambda_content=args.lambda_content,
        lambda_tv=args.lambda_tv,
        elevation_max=args.elevation_max,
        elevation_min=args.elevation_min,
        lr_vertices=args.lr_vertices,
        lr_textures=args.lr_textures,
        camera_distance=args.camera_distance,
        camera_distance_noise=args.camera_distance_noise,
        texture_size=args.texture_size,
    )
    optimizer = torch.optim.Adam([
                {'params':model.vertices,'lr':args.lr_vertices},
                {'params':model.textures,'lr':args.lr_textures}
                ], betas=(args.adam_beta1,args.adam_beta2))
    # optimization
    loop = tqdm.tqdm(range(args.num_iteration))
    for _ in loop:
        optimizer.zero_grad()
        loss = model(args.batch_size)
        loss.backward()
        optimizer.step()
        loop.set_description('Optimizing. Loss %.4f' % loss.data)

    # draw object
    #model.renderer.background_color = (1, 1, 1)
    print("Saving the object after performing style tra")
    loop = tqdm.tqdm(range(0, 360, 4))
    output_images = []
    for num, azimuth in enumerate(loop):
        loop.set_description('Drawing')
        model.renderer.eye = neural_renderer.get_points_from_angles(2.732, 30, azimuth)

        images,_,_ = model.renderer.render(model.vertices.to(device),model.faces.to(device),torch.tanh(model.textures.to(device)))
        image = images.detach().cpu().numpy()[0].transpose((1, 2, 0))
        imsave('/tmp/_tmp_%04d.png' % num, image)
    make_gif(args.filename_output)

    print("Saving the object after style transfer.")
    ##Saving the Object
    vertices = model.vertices
    faces = model.state_dict()['faces']
    textures = model.state_dict()['textures']

    filename = "stylized_" + args.filename_mesh.split("/")[-1]
    # filename = os.path.abspath(filename)
    filename_stylized_obj = args.result_directory + "/" + filename

    neural_renderer.save_obj(filename_stylized_obj, torch.squeeze(vertices), torch.squeeze(faces), torch.squeeze(textures))

    print("Generating texture mapping for the generated stylized object.")
    ##Mapping the texture
    filename = "stylized_and_textured_" + args.filename_mesh.split("/")[-1]
    filename_stylized_and_textured_obj = args.result_directory + "/" + filename
    filename = "stylized_and_textured_" + args.filename_output.split("/")[-1]
    filename_stylized_and_textured_gif = args.result_directory + "/" + filename
    map_texture_obj = MapTexture(filename_stylized_obj, args.filename_style, filename_stylized_and_textured_gif)
    map_texture_obj.train()
    map_texture_obj.save_obj(filename_stylized_and_textured_obj)

if __name__ == '__main__':
    run()
