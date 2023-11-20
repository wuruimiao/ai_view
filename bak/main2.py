from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg
import torch


def zoom_factory(ax, base_scale=2.):
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

    fig = ax.get_figure()
    fig.canvas.mpl_connect('scroll_event', zoom_fun)
    return zoom_fun


def hover_factory():
    def hover(event):
        # if the mouse is over the scatter points
        if line.contains(event)[0]:
            # find out the index within the array from the event
            ind, = line.contains(event)[1]["ind"]
            # get the figure size
            w, h = fig.get_size_inches() * fig.dpi
            ws = (event.x > w / 2.) * -1 + (event.x <= w / 2.)
            hs = (event.y > h / 2.) * -1 + (event.y <= h / 2.)
            # if event occurs in the top or right quadrant of the figure,
            # change the annotation box position relative to mouse.
            ab.xybox = (xybox[0] * ws, xybox[1] * hs)
            # make annotation box visible
            ab.set_visible(True)
            # place it at the position of the hovered scatter point
            ab.xy = (x[ind], y[ind])
            # set the image corresponding to that point
            # im.set_data(arr[ind, :, :])
            im.set_data(mpimg.imread(f"../demo/0_label(0)_ctx(23)_pts(3).png"))
        else:
            # if the mouse is not over a scatter point
            ab.set_visible(False)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('motion_notify_event', hover)
    return hover


if __name__ == '__main__':
    result = []
    for i in range(100):
        data = torch.load(f"./demo/{i}_lat.pt", map_location=torch.device("cpu"))
        data = data.cpu().detach().numpy()
        result.append(data)

    data = np.array(result)

    # tsne = TSNE(n_components=3, init='pca', random_state=0)
    tsne = TSNE(n_components=3, random_state=0)
    transformed = tsne.fit_transform(data)
    x = transformed[:, 0]
    y = transformed[:, 1]

    fig = plt.figure(figsize=(27, 18))
    plt.scatter(x, y)
    plt.legend()

    ax = fig.add_subplot(111)
    line, = ax.plot(x, y, ls="", marker="o")
    im = OffsetImage(np.empty((len(x), 10, 10))[0, :, :], zoom=5)
    xybox = (50., 50.)
    ab = AnnotationBbox(
        im, (0.15, 0.5),
        xybox=(50., 50.),
        xycoords='data',
        boxcoords="offset points",
        pad=0.5
    )
    # add it to the axes and make it invisible
    ax.add_artist(ab)
    ab.set_visible(False)
    f = zoom_factory(ax)
    f2 = hover_factory()

    # def hover(event):
    #     # if the mouse is over the scatter points
    #     if line.contains(event)[0]:
    #         # find out the index within the array from the event
    #         ind, = line.contains(event)[1]["ind"]
    #         # get the figure size
    #         w, h = fig.get_size_inches() * fig.dpi
    #         ws = (event.x > w / 2.) * -1 + (event.x <= w / 2.)
    #         hs = (event.y > h / 2.) * -1 + (event.y <= h / 2.)
    #         # if event occurs in the top or right quadrant of the figure,
    #         # change the annotation box position relative to mouse.
    #         ab.xybox = (xybox[0] * ws, xybox[1] * hs)
    #         # make annotation box visible
    #         ab.set_visible(True)
    #         # place it at the position of the hovered scatter point
    #         ab.xy = (x[ind], y[ind])
    #         # set the image corresponding to that point
    #         # im.set_data(arr[ind, :, :])
    #         im.set_data(mpimg.imread(f"./demo/0_label(0)_ctx(23)_pts(3).png"))
    #     else:
    #         # if the mouse is not over a scatter point
    #         ab.set_visible(False)
    #     fig.canvas.draw_idle()
    #
    #
    # fig.canvas.mpl_connect('motion_notify_event', hover)
    plt.show()
