from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg
import torch

if __name__ == '__main__':
    result = []
    for i in range(100):
        data = torch.load(f"./demo/{i}_lat.pt", map_location=torch.device("cpu"))
        data = data.cpu().detach().numpy()
        result.append(data)

    data = np.array(result)

    label = torch.tensor([0, 0, 1])
    label = label.numpy()
    train_index = [1]
    n_samples = data.shape[0]
    n_features = data.shape[1]

    tsne = TSNE(n_components=3, init='pca', random_state=0)
    transformed = tsne.fit_transform(data)
    x = transformed[:, 0]
    y = transformed[:, 1]

    fig = plt.figure(figsize=(9, 6))
    plt.scatter(x, y)
    plt.legend()

    ax = fig.add_subplot(111)
    line, = ax.plot(x, y, ls="", marker="o")
    arr = np.empty((len(x), 10, 10))
    im = OffsetImage(arr[0, :, :], zoom=5)
    xybox = (50., 50.)
    ab = AnnotationBbox(
        im, (0.15, 0.5),
        xybox=(50., 50.),
        xycoords='data',
        boxcoords="offset points",
        pad=0.5
    )
    ax.add_artist(ab)
    ab.set_visible(False)


    def hover(event):
        if line.contains(event)[0]:
            ind, = line.contains(event)[1]["ind"]
            w, h = fig.get_size_inches() * fig.dpi
            ws = (event.x > w / 2.) * -1 + (event.x <= w / 2.)
            hs = (event.y > h / 2.) * -1 + (event.y <= h / 2.)
            ab.xybox = (xybox[0] * ws, xybox[1] * hs)
            ab.set_visible(True)
            ab.xy = (x[ind], y[ind])
            im.set_data(mpimg.imread(f"./demo/0_label(0)_ctx(23)_pts(3).png"))
        else:
            ab.set_visible(False)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('motion_notify_event', hover)
    plt.show()
