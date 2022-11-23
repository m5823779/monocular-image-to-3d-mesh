import os.path as path

import cv2
import torch
import argparse
import numpy as np
import pyvista as pv

from pyvistaqt import BackgroundPlotter
from PyQt5 import QtWidgets, QtCore


def write_depth(depth, bits=1, reverse=True):
    depth_min = depth.min()
    depth_max = depth.max()

    max_val = (2 ** (8 * bits)) - 1

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


class BaseFrameProcess:
    def __init__(self, is_bgr2rgb=False):
        self.is_bgr2rgb = is_bgr2rgb

    def forward(self, x):
        if self.is_bgr2rgb == True:
            x = x[..., ::-1].copy()

        return x


class TextureFrameProcess(BaseFrameProcess):
    def __init__(self, is_bgr2rgb=True):
        super().__init__(is_bgr2rgb=is_bgr2rgb)
        self.np_to_texture = pv.numpy_to_texture

    def forward(self, x):
        x = super().forward(x)
        x = self.np_to_texture(x)

        return x


class DepthFrameProcess(BaseFrameProcess):
    def __init__(self, res_multiplier=1, is_mmn=True, max_depth=1.0):
        super().__init__()
        self.res_mult = res_multiplier
        self.is_mmn = is_mmn
        self.max_depth = max_depth

    def minmax_norm(self, x):
        x = x / 255.0  #normalize to 0.0 ~ 1.0

        # min/max normalization
        x_min = np.amin(x).astype(float)
        x_max = np.amax(x).astype(float)

        x = x - x_min
        x = x / (x_max - x_min)

        return x

    def forward(self, x):
        assert (x.ndim == 2 or x.shape[2] == 1), "Depth image needs to be grayscale."

        cy, cx, _ = x.shape if x.ndim == 3 else (x.shape[0], x.shape[1], 0)
        x = cv2.resize(x, (depth_cx * self.res_mult, depth_cy * self.res_mult), interpolation=cv2.INTER_CUBIC)

        if self.is_mmn == True:
            x = self.minmax_norm(x)

        if self.max_depth != 0.0:
            x = x * self.max_depth

        return x

'''
camera position class
ipd     = 65    # IPD is ~65mm
scrn_sx = 345.6 # screen size width = 345.6mm (typical physical width for 15.6" screen)
cam_z   = 600   # camera z at 600mm
assumes view x ranges from -1.0 ~ 1.0 in screen's horizontal
p - denotes physical real world space
v - denotes view world space (normalized in view coordinate space)
'''


class CameraPositions:
    def __init__(self, ipd=65, screen_length=345.6, camera_position_z=600):
        self.p_ipd = ipd
        self.p_scrn_sx = screen_length
        self.p_cam_z = camera_position_z

        p_coord_unit = self.p_scrn_sx / 2.0

        self.v_ipd = self.p_ipd / p_coord_unit
        # self.v_eye_dis_x = (self.p_ipd / 2) / p_coord_unit
        self.v_cam_z = self.p_cam_z / p_coord_unit

        self.pos_list = []

    def get_ipd(self):
        return self.p_ipd

    def get_vipd(self):
        return self.v_ipd

    def get_cam_z(self):
        return self.p_cam_z

    def get_vcam_z(self):
        return self.v_cam_z

    def get_coord(self, idx):
        return self.pos_list[idx][0]

    def get_text(self, idx):
        return self.pos_list[idx][1]

    def add_pos(self, pos, text):
        self.pos_list.append((pos, text))

    def get_size(self):
        return len(self.pos_list)

    def get_list(self):
        return self.pos_list


# utility functions
def mesh_norm_units(mesh, tex_img):
    scrn_cy, scrn_cx, _ = tex_img.shape

    mesh_cy, mesh_cx = mesh.shape

    # scale the mesh to -1.0 ~ 1.0
    mesh_unit_cx = scrn_cx / 2.0
    mesh_unit_cy = scrn_cy / 2.0

    mesh_unit = mesh_unit_cx if mesh_unit_cx > mesh_unit_cy else mesh_unit_cy  # larger side as view unit

    mesh_unit_x = mesh_unit_cx / mesh_unit
    mesh_unit_y = mesh_unit_cy / mesh_unit
    # print(f"mesh_unit_x,mesh_unit_y: {mesh_unit_x},{mesh_unit_y}")

    mesh_step_x = mesh_unit_x / (mesh_cx / 2.0)
    mesh_step_y = mesh_unit_y / (mesh_cy / 2.0)

    return (mesh_unit_x, mesh_unit_y, mesh_step_x, mesh_step_y)


def event_save_screenshot():
    file_name = "screenshot.png"
    print(f"saving screenshot to {file_name}")
    plotter.screenshot(file_name)


class SurfacePicker:
    def __init__(self, plotter):
        self.plotter = None

    def __call__(self, point):
        labels = f"{point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f}"

        plotter.subplot(0, 0)
        plotter.add_point_labels(point, [labels])
        plotter.subplot(0, 1)
        plotter.add_point_labels(point, [labels])

        print(f"clicked at {labels}")

    @property
    def points(self):
        """To access all th points when done."""
        return self._points


if __name__ == "__main__":
    SupportModel = ["DPT_Large", "DPT_Hybrid", "MiDaS_small", "MiDaS"]

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-m', '--model_type', default='DPT_Large', type=str,
                        help="choose MiDaS model type ['DPT_Large', 'DPT_Hybrid', 'MiDaS_small', 'MiDaS']")
    parser.add_argument('-i', '--input_image',
                        help='texture image file')
    args = parser.parse_args()

    ''' Configuration '''
    model_type = args.model_type
    image_file_path = args.input_image
    tex_has_lighting = True  # Texture lighting
    tex_show_edges = False  # Texture rendered with edges
    depth_cmap_name = None  # Colormap names from colorcet (http://https://colorcet.holoviz.org) e.g. jet, hot, fire, bgy, bgyw, glasbey, None
    max_depth = 0.6  # Maximum depth

    depth_res_multiplier = 1  # Depth resolution multiplier
    depth_has_lighting = True  # Depth lighting
    depth_mesh_style = 'wireframe'  # depth style - 'surface', 'wireframe', 'points'
    depth_mesh_line_width = 0.1
    depth_mesh_point_size = 1
    depth_interpolate_before_mapping = False  # interpolate before mapping

    screen_length = 345.6
    cam_ipd = 65
    cam_position_z = 600
    cam_pos_def = (-2.0, 0.0, 4.0)

    assert (args.model_type in SupportModel), f"Model '{args.model_type}' is not supported.\nPlease choose 'DPT_Large', 'DPT_Hybrid', 'MiDaS_small' or 'MiDaS'"
    assert (path.exists(image_file_path)), f"Image file '{image_file_path}' does not exists."

    # setup frame process
    tex_process = TextureFrameProcess(is_bgr2rgb=True)
    depth_process = DepthFrameProcess(max_depth=max_depth, res_multiplier=depth_res_multiplier)

    ''' MiDaS '''
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

    # read texture
    tex_img = cv2.imread(image_file_path)
    tex = tex_process.forward(tex_img)

    # AI preprocessing
    img = cv2.cvtColor(tex_img, cv2.COLOR_BGR2RGB)
    input_batch = transform(img).to(device)

    # inference and resize
    with torch.no_grad():
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    output = prediction.cpu().numpy()
    depth_img = write_depth(output, 1)

    depth_save_path = path.basename(image_file_path).split(".")[0] + "_depth.png"
    cv2.imwrite(depth_save_path, depth_img)
    print(f"saving depth map to {depth_save_path}")

    # depth mesh
    depth_cy, depth_cx = depth_img.shape
    depth = depth_process.forward(depth_img)

    # mesh coordinates
    mesh_unit_x, mesh_unit_y, mesh_step_x, mesh_step_y = mesh_norm_units(depth, tex_img)

    x = np.arange(-mesh_unit_x, mesh_unit_x, mesh_step_x)  # mesh left-top x is at -1.0
    y = np.arange(mesh_unit_y, -mesh_unit_y, -mesh_step_y)  # mesh left-top y is at  1.0
    x, y = np.meshgrid(x, y)

    # create StructuredGrid
    mesh_sg = pv.StructuredGrid(x, y, depth)
    mesh_sg.texture_map_to_plane(use_bounds=True, point_u={1.0, 1.0, 1.0}, point_v={0.0, 0.0, 0.0}, inplace=True)

    # setup pyvistaqt window
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)  # DPI support
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)  # Hi-DPI icons & bitmaps

    win_cx = max((tex_img.shape[1] * 4) // 5, 960)
    win_cy = max((tex_img.shape[0] * 4) // 5, 540)

    plotter = None
    plotter = BackgroundPlotter(show=False, window_size=(win_cx, win_cy), notebook=False, off_screen=False, shape=(1, 2))
    plotter.enable_hidden_line_removal()
    renderers = plotter.renderers

    # ref for color - https://docs.pyvista.org/api/utilities/_autosummary/pyvista.Color.name.html#pyvista.Color.name
    renderers[0].set_background("dimgray")
    print(f"viewport[0]: {renderers[0].viewport}")
    print(f"viewport[1]: {renderers[1].viewport}")

    # setup mesh updates
    SMOOTH_TEXT = "smooth_text"

    depth_processor = DepthFrameProcess(max_depth=max_depth, res_multiplier=depth_res_multiplier)
    depth_img = depth_processor.forward(depth_img)

    ''' Plot '''
    plotter.subplot(0, 0)
    tmesh = plotter.add_mesh(mesh_sg, name="mesh_texture", texture=tex, lighting=tex_has_lighting, style="surface", show_edges=tex_show_edges, render=False)

    image_file_name = path.basename(image_file_path)
    plotter.add_text(image_file_name, font_size=8, render=False)

    text = f"input source resolution: {tex_img.shape[1]} x {tex_img.shape[0]}\n"
    text += f"depth map resolution: {depth.shape[1]} x {depth.shape[0]}\n"
    text += f"depth resolution multiplier: {depth_res_multiplier} \n"
    text += f"depth colormap: {depth_cmap_name}\n"
    plotter.add_text(text, name="depth_info", font_size=8, position=(0.01, 0.80), viewport=True, render=False)

    plotter.add_axes(interactive=True)

    plotter.subplot(0, 1)
    dmesh_sname = None
    if depth_cmap_name != None:
        dmesh_sname = f"depth ({depth_cmap_name})"
        mesh_sg[dmesh_sname] = depth.flatten(order='F')

    dmesh = plotter.add_mesh(
        mesh_sg,
        # texture=tex,
        lighting=depth_has_lighting,
        # smooth_shading=True,
        style=depth_mesh_style,
        line_width=depth_mesh_line_width,
        point_size=depth_mesh_point_size,
        # show_edges=True,
        render=False,
        # scalar_bar_args={'title': f"original ({depth_cmap_name})"},
        interpolate_before_map=depth_interpolate_before_mapping,
        show_scalar_bar=True if dmesh_sname != None else False,
        scalars=dmesh_sname,
        cmap=depth_cmap_name)

    plotter.add_axes(interactive=True)
    plotter.add_bounding_box()
    plotter.show_bounds(render=False)

    # setup colormap - needs to be after add_mesh()
    if depth_cmap_name != None:
        plotter.add_scalar_bar(dmesh_sname, title=f"depth colormap ({depth_cmap_name})", position_y=0.01, interactive=False, vertical=False, render=False)

    # setup surface picker
    surface_picker = SurfacePicker(plotter)
    plotter.enable_surface_picking(callback=surface_picker, show_point=True, show_message=False)

    # link both views
    plotter.link_views(1)

    ''' View port '''
    # setup camera positions
    cam_pos_list = CameraPositions(ipd=cam_ipd, screen_length=screen_length, camera_position_z=cam_position_z)

    ipd = cam_pos_list.get_vipd()
    cam_z = cam_pos_list.get_vcam_z()
    cam_pos_list.add_pos(cam_pos_def, "default")
    cam_pos_list.add_pos((0.0, 0.0, cam_z), "center")
    cam_pos_list.add_pos((-ipd / 2.0, 0.0, cam_z), "left")
    cam_pos_list.add_pos((ipd / 2.0, 0.0, cam_z), "right")

    cam_focus = (0.0, 0.0, 0.0)
    cam_up = (0.0, 1.0, 0.0)
    plotter.camera.focal_point = cam_focus
    plotter.camera.up = cam_up

    for cam_idx in range(cam_pos_list.get_size()):
        pos_vcoord = cam_pos_list.get_coord(cam_idx)
        pos_text = cam_pos_list.get_text(cam_idx)
        plotter.camera.position = pos_vcoord

        print(f"cam {cam_idx}: {pos_vcoord[0]:.3f},{pos_vcoord[1]:.3f},{pos_vcoord[2]:.3f} ({pos_text})")

        if type(plotter) == BackgroundPlotter:
            plotter.save_camera_position()

    # set camera to default
    plotter.camera.position = cam_pos_list.get_coord(0)

    # setup callbacks for key events
    plotter.add_key_event("s", event_save_screenshot)
    plotter.app_window.show()

    print("'q' key - quit")
    print("'s' key - screenshot")
    print("'p' key - click on mesh and press 'p' to add a point with coordinate")

    # input("press ENTER to exit...")
    plotter.app.exec_()


