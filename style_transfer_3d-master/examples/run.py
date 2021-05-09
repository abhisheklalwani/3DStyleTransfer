#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import glob
import os
import subprocess
import torch
import cv2

import neural_renderer
import numpy as np
import tqdm
import imageio

import style_transfer_3d
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def make_gif(working_directory, filename):
    # generate gif (need ImageMagick)
    options = '-delay 8 -loop 0 -layers optimize'
    subprocess.call('convert %s %s/_tmp_*.png %s' % (options, working_directory, filename), shell=True)
    for filename in glob.glob('%s/_tmp_*.png' % working_directory):
        os.remove(filename)


def run():
    # load settings
    parser = argparse.ArgumentParser()
    parser.add_argument('-im', '--filename_mesh', type=str)
    parser.add_argument('-is', '--filename_style', type=str)
    parser.add_argument('-o', '--filename_output', type=str)
    parser.add_argument('-ls', '--lambda_style', type=float, default=1.)
    parser.add_argument('-lc', '--lambda_content', type=float, default=2e9)
    parser.add_argument('-ltv', '--lambda_tv', type=float, default=1e7)
    parser.add_argument('-emax', '--elevation_max', type=float, default=40.)
    parser.add_argument('-emin', '--elevation_min', type=float, default=20.)
    parser.add_argument('-lrv', '--lr_vertices', type=float, default=0.01)
    parser.add_argument('-lrt', '--lr_textures', type=float, default=1.0)
    parser.add_argument('-cd', '--camera_distance', type=float, default=2.732)
    parser.add_argument('-cdn', '--camera_distance_noise', type=float, default=0.1)
    parser.add_argument('-ts', '--texture_size', type=int, default=4)
    parser.add_argument('-lr', '--adam_lr', type=float, default=0.05)
    parser.add_argument('-ab1', '--adam_beta1', type=float, default=0.9)
    parser.add_argument('-ab2', '--adam_beta2', type=float, default=0.999)
    parser.add_argument('-bs', '--batch_size', type=int, default=4)
    parser.add_argument('-ni', '--num_iteration', type=int, default=1)
    parser.add_argument('-g', '--gpu', type=int, default=0)
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
    optimizer = torch.optim.Adam(model.parameters(), lr=args.adam_lr, betas=(args.adam_beta1,args.adam_beta2))
    # optimization
    loop = tqdm.tqdm(range(args.num_iteration))
    for _ in loop:
        optimizer.zero_grad()
        loss = model(args.batch_size)
        loss.backward()
        optimizer.step()
        loop.set_description('Optimizing. Loss %.4f' % loss.data)

    # draw object
    model.renderer.background_color = (1, 1, 1)
    loop = tqdm.tqdm(range(0, 360, 4))
    output_images = []
    for num, azimuth in enumerate(loop):
        loop.set_description('Drawing')
        model.renderer.eye = neural_renderer.get_points_from_angles(2.732, 30, azimuth)
        x = torch.unsqueeze(model.vertices,0)
        x = x.repeat(1,1,1)
        x = x.to(device)

        y = torch.unsqueeze(model.state_dict()['faces'],0)
        y = y.repeat(1,1,1)
        y = y.to(device)

        z = torch.unsqueeze(model.state_dict()['textures'],0)
        z = torch.sigmoid(z.repeat(1,1,1,1,1,1))
        z = z.to(device)
        images,_,_ = model.renderer.render(x,y,z)
        image = images[0].permute((1, 2, 0)).cpu().detach().numpy()
        output_images.append(image)
    imageio.mimsave(args.filename_output,output_images)


if __name__ == '__main__':
    run()
