"""
python run_latent_vis.py -d demo
demo是存放点和图的路径
"""

import os
import re
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import Divider, Size
import matplotlib.image as mpimg
import torch


def is_num(s) -> bool:
    return isinstance(s, int) or isinstance(s, float) or s.replace('.', '', 1).isdigit()


_IntRegex = re.compile(r'\d+')


def get_ints_in_str(s: str) -> list[int]:
    ss = _IntRegex.findall(s)
    result = []
    for s in ss:
        if is_num(s):
            result.append(int(s))
    return result


def zoom_factory(ax, base_scale=2.):
    """
    缩放
    """

    def zoom_fun(event):
        cur_x = ax.get_xlim()
        cur_y = ax.get_ylim()
        cur_xrange = (cur_x[1] - cur_x[0]) * .5
        cur_yrange = (cur_y[1] - cur_y[0]) * .5
        xdata = event.xdata
        ydata = event.ydata
        if event.button == 'up':
            scale_factor = 1 / base_scale
        elif event.button == 'down':
            scale_factor = base_scale
        else:
            scale_factor = 1
        ax.set_xlim([xdata - cur_xrange * scale_factor, xdata + cur_xrange * scale_factor])
        ax.set_ylim([ydata - cur_yrange * scale_factor, ydata + cur_yrange * scale_factor])
        plt.draw()

    ax.get_figure().canvas.mpl_connect('scroll_event', zoom_fun)
    return zoom_fun


def hover_factory(ax, img_paths, lines, x, y):
    """
    鼠标指向点时，展示对应图片
    """
    im = OffsetImage(np.empty((len(x), 10, 10))[0, :, :], zoom=5)
    ab = AnnotationBbox(
        im, (0.5, 0.6),
        xybox=(1.05, 0.5),
        xycoords='data',
        boxcoords='axes fraction',
        box_alignment=(0, 0),
        bboxprops=dict(facecolor='w'),
        arrowprops=dict(
            arrowstyle='->', relpos=(0, 0.5),  # 设置箭头的起始位置，左下角为(0,0),右上角为(1,1)
            connectionstyle='angle,angleA=0,angleB=90,rad=3',
        ),
        pad=0.2,
        annotation_clip=False
    )
    ab.set_visible(False)
    ax.add_artist(ab)

    def hover(event):
        if lines.contains(event)[0]:
            ind = lines.contains(event)[1]["ind"][0]

            # w, h = fig.get_size_inches() * fig.dpi
            # ws = (event.x > w / 2.) * -1 + (event.x <= w / 2.)
            # hs = (event.y > h / 2.) * -1 + (event.y <= h / 2.)
            # ab.xybox = (xy_box[0] * ws, xy_box[1] * hs)

            ab.set_visible(True)
            ab.xy = (x[ind], y[ind])
            im.set_data(mpimg.imread(img_paths[ind]))
        else:
            ab.set_visible(False)
        plt.draw()

    ax.get_figure().canvas.mpl_connect('motion_notify_event', hover)
    return hover


def set_ax_size(fig, width, height, is_3d: bool = False):
    divider = Divider(
        fig, (0, 0, 1, 1),
        [Size.Fixed(0.6), Size.Fixed(width)],  # width
        [Size.Fixed(0.6), Size.Fixed(height)],  # height
        aspect=False)
    ax = fig.add_axes(
        divider.get_position(),
        axes_locator=divider.new_locator(nx=1, ny=1),
        projection="3d" if is_3d else None)
    return ax


def lighten_color(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def init_plt(x, y, img_paths, pts_num, z=None):
    """
    初始化matplotlib展示区
    """
    width = 18
    height = 12
    fig = plt.figure(figsize=(width, height))
    fig.canvas.manager.set_window_title("Deep Scatter Visualizer")

    ax = set_ax_size(fig, width - 6, height - 1, is_3d=z is not None)

    color = [lighten_color('b', num/10) for num in pts_num]
    if z is None:
        lines = ax.scatter(x, y, c=color)
    else:
        lines = ax.scatter(x, y, z, c=color)

    f = zoom_factory(ax)
    f2 = hover_factory(ax, img_paths, lines, x, y)


def load_pt_png(base_dir: str = ""):
    """
    加载base_dir下的点数据，和一一对应的图片路径
    """

    def _index(name: str):
        nums = get_ints_in_str(name)
        return nums[0], nums[-1]

    pts = {}
    pngs = {}
    pts_num = {}
    for filename in os.listdir(base_dir):
        filepath = os.path.join(base_dir, filename)
        if not os.path.isfile(filepath):
            continue
        is_pt = filename.endswith(".pt")
        is_png = filename.endswith(".png")
        if not is_pt and not is_png:
            continue
        _name, pt_num = _index(filename)
        if is_pt:
            if _name in pts:
                raise Exception(f"exist {_name} {pts[_name]}")
            pts[_name] = filepath
        elif is_png:
            if _name in pngs:
                raise Exception(f"exist {_name} {pngs[_name]}")
            pngs[_name] = filepath
            pts_num[_name] = pt_num
    pts = sorted(pts.items(), key=lambda x: x[0])
    pngs = sorted(pngs.items(), key=lambda x: x[0])
    pts_num = sorted(pts_num.items(), key=lambda x: x[0])
    assert len(pts) == len(pngs)

    data = []
    for _, pt in pts:
        data.append(torch.load(pt).cpu().detach().numpy())
    data = np.array(data)
    return data, [item[1] for item in pngs], [item[1] for item in pts_num]


def load_x_y(data):
    """
    t-sne算法降维数据
    """
    tsne = TSNE(n_components=3, random_state=0)
    transformed = tsne.fit_transform(data)
    return transformed[:, 0], transformed[:, 1], transformed[:, 2]


def init_arg():
    """
    命令行解释器
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir", type=str, default="./", help="存放点和图片的路径", required=True)
    parser.add_argument("-n", "--component", type=int, default="2", help="嵌入空间维度")
    return parser.parse_args()


if __name__ == '__main__':
    args = init_arg()
    pts, pngs, pts_num = load_pt_png(args.dir)
    x, y, z = load_x_y(pts)
    if args.component == 2:
        z = None
    init_plt(x, y, pngs, pts_num, z)
    plt.show()
