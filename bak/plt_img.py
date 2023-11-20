import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np; np.random.seed(42)
import os

os.chdir('Path/to/your/images')

# Generate data x, y for scatter and an array of images.
x = np.arange(3)
y = np.random.rand(len(x))
jpg_name_np = np.array(['904646.jpg', '903825.jpg', '905722.jpg']).astype('<U12') # names of your images files

cmap = plt.cm.RdYlGn

# create figure and plot scatter
fig = plt.figure()
ax = fig.add_subplot(111)
#line, = ax.plot(x,y, ls="", marker="o")
line = plt.scatter(x,y,c=heat, s=10, cmap=cmap)
image_path = np.asarray(jpg_name_np)

# create the annotations box
image = plt.imread(image_path[0])
im = OffsetImage(image, zoom=0.1)
xybox=(50., 50.)
ab = AnnotationBbox(im, (0,0), xybox=xybox, xycoords='data',
        boxcoords="offset points",  pad=0.3,  arrowprops=dict(arrowstyle="->"))
# add it to the axes and make it invisible
ax.add_artist(ab)
ab.set_visible(False)

def hover(event):
    # if the mouse is over the scatter points
    if line.contains(event)[0]:
        # find out the index within the array from the event
        ind, = line.contains(event)[1]["ind"]
        # get the figure size
        w,h = fig.get_size_inches()*fig.dpi
        ws = (event.x > w/2.)*-1 + (event.x <= w/2.)
        hs = (event.y > h/2.)*-1 + (event.y <= h/2.)
        # if event occurs in the top or right quadrant of the figure,
        # change the annotation box position relative to mouse.
        ab.xybox = (xybox[0]*ws, xybox[1]*hs)
        # make annotation box visible
        ab.set_visible(True)
        # place it at the position of the hovered scatter point
        ab.xy =(x[ind], y[ind])
        # set the image corresponding to that point
        im.set_data(plt.imread(image_path[ind]))
    else:
        #if the mouse is not over a scatter point
        ab.set_visible(False)
    fig.canvas.draw_idle()

# add callback for mouse moves
fig.canvas.mpl_connect('motion_notify_event', hover)

fig = plt.gcf()
fig.set_size_inches(10.5, 9.5)

plt.show()
