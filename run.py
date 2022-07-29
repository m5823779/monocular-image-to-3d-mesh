import os
import cv2
import torch
import numpy as np

import glob
import argparse

import urllib.request
import pyvista as pv

# mesh configs
image_as_texture = True  # default: True
depth_max = 0.5  # default: 0.5
depth_res_scaler = 2  # default: 2
depth_blur = True  # default: True


def write_depth(depth, bits=1, reverse=True):
    depth_min = depth.min()
    depth_max = depth.max()

    max_val = (2**(8*bits))-1

    if depth_max - depth_min > np.finfo("float").eps:
        out = max_val * (depth - depth_min) / (depth_max - depth_min)
    else:
        out = 0
    if not reverse:
        out = max_val - out

    if bits == 2:
        depth_map = out.astype("uint16")
    else:
        depth_map = out.astype("uint8")

    return depth_map

def draw_mesh(filename, img, depth):
    cy, cx = img.shape[:2]

    tex = None
    if image_as_texture == True:
        tex = pv.numpy_to_texture(img)

    # simulated screen width/height
    # scrn_cx = 1920.0
    # scrn_cy = 1080.0
    scrn_cx = cx
    scrn_cy = cy

    depth_cy, depth_cx = depth.shape[:]
    depth = cv2.resize(depth, (depth_cx * depth_res_scaler, depth_cy * depth_res_scaler), interpolation=cv2.INTER_CUBIC)

    guss_r = 20 * depth_res_scaler + 1  # needs to be an odd number
    depth = cv2.GaussianBlur(depth, (guss_r, guss_r), 0)

    mesh = depth
    mesh_cy, mesh_cx = mesh.shape
    print(f"mesh cx,cy: {mesh_cx},{mesh_cy}")

    # y-invert
    mesh = cv2.flip(mesh, 0)

    # scale the mesh to -1.0 ~ 1.0
    mesh_unit_cx = scrn_cx / 2.0
    mesh_unit_cy = scrn_cy / 2.0

    mesh_unit = mesh_unit_cx if mesh_unit_cx > mesh_unit_cy else mesh_unit_cy  # larger side as view unit

    mesh_unit_x = mesh_unit_cx / mesh_unit
    mesh_unit_y = mesh_unit_cy / mesh_unit

    print(f"mesh_unit_x,mesh_unit_y: {mesh_unit_x},{mesh_unit_y}")

    mesh_step_x = mesh_unit_x / (mesh_cx / 2.0)
    mesh_step_y = mesh_unit_y / (mesh_cy / 2.0)

    # mesh = mesh / 255.0  #normalize to 0.0 ~ 1.0

    # min/max normalization
    mesh_min = np.amin(mesh).astype(float)
    mesh_max = np.amax(mesh).astype(float)

    print(f"depth mesh min,max: {mesh_min},{mesh_max}")

    mesh = mesh - mesh_min
    mesh = mesh / (mesh_max - mesh_min)

    # apply max depth value
    mesh = mesh * depth_max

    # x = np.arange(-cx/2.0/10.0, cx/2.0/10.0, 1.0/10.0)
    # y = np.arange(-cy/2.0/10.0, cy/2.0/10.0, 1.0/10.0)
    x = np.arange(-mesh_unit_x, mesh_unit_x, mesh_step_x)
    y = np.arange(-mesh_unit_y, mesh_unit_y, mesh_step_y)
    x, y = np.meshgrid(x, y)

    curvsurf = pv.StructuredGrid(x, y, mesh)
    # curvsurf.texture_map_to_plane(inplace=True)
    curvsurf.texture_map_to_plane(use_bounds=True, point_u={1.0, 1.0, 1.0}, point_v={0.0, 0.0, 0.0}, inplace=True)
    # curvsurf.texture_map_to_plane(point_u={1.0,1.0,1.0}, point_v={0.0,0.0,0.0}, inplace=True)

    # pl = curvsurf.plot(texture=tex, cpos=[(1.1*5.0,1.5*5.0,100.0),(0.0,0.0,0.0),(0.0,1.0,0.0)])

    '''
    cam_pos = [(cx/2.0/10.0, -cy/2.0/10.0, 10.0), #(-cx/2.0, -cy/2.0, -10.0),
                (0.0, 0.0, 0.0),
                (0.0, 1.0, 0.0)
                ]
    '''

    eye_dis_x = (6.5 / 2) / (34.56 / 2.0)
    print(f"eye_dis_x: {eye_dis_x}")

    cam_pos = [(-eye_dis_x, 0.0, 3.47 * 1.5),  # (-cx/2.0, -cy/2.0, -10.0),
               (0.0, 0.0, 0.0),
               (0.0, 1.0, 0.0)
               ]
    '''            
    cam_pos = [(-2.1, 0.5, 3.47*1.5), #(-cx/2.0, -cy/2.0, -10.0),
                (0.0, 0.0, 0.0),
                (0.0, 1.0, 0.0)
                ]
    '''

    '''
    pos = curvsurf.plot(texture=tex, cpos=cam_pos, return_cpos=True, 
                #parallel_projection=True,
                show_bounds=True,
                #show_axes=False,
                text=img_file_name,
                #volume=True,
                #full_screen=True
                #off_screen=True,
                #smooth_shading=True,    # only for shading.  not for texture
                #split_sharp_edges=True, # only for shading.  not for texture
                )

    print(pos)

    '''
    plotter = pv.Plotter(notebook=False, off_screen=False)
    plotter.add_mesh(
        curvsurf,
        texture=tex,
        lighting=True,
        # smooth_shading=True
    )

    plotter.add_text(filename + " - left eye", font_size=12)
    plotter.add_axes(interactive=True)
    plotter.add_bounding_box()
    # plotter.add_scalar_bar()

    plotter.show_bounds(grid='front', location='outer')
    # plotter.parallel_projection = True

    '''
    plotter.remove_all_lights()

    light = pv.Light(position=(-eye_dis_x, 0.0, 3.47*1.5))
    light.diffuse_color = 1.0, 1.0, 1.0

    plotter.add_light(light)
    '''

    # plotter.show(interactive_update=True)

    win_cx = max((cx * 3) // 5, 960)
    win_cy = max((cy * 3) // 5, 540)
    print(f"window cx,cy: {win_cx},{win_cy}")
    pos = plotter.show(window_size=[win_cx, win_cy],
                       cpos=cam_pos,
                       return_cpos=True,
                       title=filename,
                       # show_axes=False,
                       # scalar_bar_args={"title": "Height"},
                       )
    print(pos)


def run(model_type, img_folder):
    # load a model
    midas = torch.hub.load("intel-isl/MiDaS", model_type)

    # move model to GPU if available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    midas.to(device)
    midas.eval()

    # load transforms to resize and normalize the image
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform

    # list all image in folder
    support_ext = ['png', 'jpg', 'jpeg', 'bmp']
    files = []
    [files.extend(glob.glob(img_folder + '/*.' + ext)) for ext in support_ext]

    for filename in files:
        # load image and apply transforms
        img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_batch = transform(img).to(device)

        # inference and resize
        with torch.no_grad():
            prediction = midas(input_batch)
            # prediction = prediction.unsqueeze(1).squeeze()

            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        output = prediction.cpu().numpy()
        depth = write_depth(output, 1)

        # cv2.imshow('depth', output)
        # cv2.waitKey(0)
        draw_mesh(filename, img, depth)




if __name__ == "__main__":
    SupportModel = ["DPT_Large", "DPT_Hybrid", "MiDaS_small", "MiDaS"]

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', default='./input/', type=str, help="input image folder")
    parser.add_argument('-m', '--model_type', default='DPT_Large', type=str, help="choose MiDaS model type ['DPT_Large', 'DPT_Hybrid', 'MiDaS_small', 'MiDaS']")
    args = parser.parse_args()

    assert (args.model_type in SupportModel), f"Model '{args.model_type}' is not supported.\nPlease choose 'DPT_Large', 'DPT_Hybrid', 'MiDaS_small' or 'MiDaS'"
    model_type = args.model_type

    img_folder = args.input
    run(model_type, img_folder)
