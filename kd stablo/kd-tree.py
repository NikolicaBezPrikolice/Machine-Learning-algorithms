import numpy as np
import matplotlib.pyplot as plt

class Node:
    def __init__(self, point, depth, left=None, right=None):
        self.point = point
        self.depth = depth
        self.left = left
        self.right = right

class KDTree:
    def __init__(self, data):
        self.root = self.build_tree(data)

    def build_tree(self, data, depth=0):
        n = len(data)
        if n == 0:
            return None
        k = data.shape[1]
        axis = depth % k
        sorted_data = data[data[:, axis].argsort()]
        mid = n // 2
        return Node(sorted_data[mid],
                    depth,
                    self.build_tree(sorted_data[:mid], depth+1),
                    self.build_tree(sorted_data[mid+1:], depth+1))


def plot_tree(node, min_x, max_x, min_y, max_y, axis, depth):
    if node is None:
        return
    if axis == 0:
        plt.plot([node.point[0], node.point[0]], [min_y, max_y], 'k-', linewidth=0.5)
        plt.pause(1)
        plot_tree(node.left, min_x, node.point[0], min_y, max_y, 1, depth+1)
        plot_tree(node.right, node.point[0], max_x, min_y, max_y, 1, depth+1)
    else:
        plt.plot([min_x, max_x], [node.point[1], node.point[1]], 'k-', linewidth=0.5)
        plt.pause(1)
        plot_tree(node.left, min_x, max_x, min_y, node.point[1], 0, depth+1)
        plot_tree(node.right, min_x, max_x, node.point[1], max_y, 0, depth+1)

if __name__ == '__main__':
    fig, ax = plt.subplots()
    ax.autoscale(False)


    x_coords = np.empty((0,2))
    plt.suptitle('Klikom se dodaju tacke\n pritiskom na Enter se dobija rezultat\n pritiskom na r se resetuje kanvas', fontsize=11)
    kd_tree_drawn=False

    def on_click(event):
        global x_coords, kd_tree_drawn
        if event.inaxes == ax and not kd_tree_drawn:
            x = event.xdata
            y = event.ydata
            ax.scatter(x, y)
            fig.canvas.draw()     
            x_coords = np.append(x_coords, [[x,y]],axis=0)

    def on_key(event):
        global kd_tree_drawn, x_coords
        if event.key == 'enter':
            kdtree = KDTree(x_coords)
            plot_tree(kdtree.root, 0, 1, 0, 1, 0, 0)
            fig.canvas.draw()
            kd_tree_drawn=True
        elif event.key == 'r':
            ax.cla()
            ax.autoscale(False)
            x_coords = np.empty((0,2))
            fig.canvas.draw()
            kd_tree_drawn= False

    cid = fig.canvas.mpl_connect('button_press_event', on_click)
    cid2 = fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()
