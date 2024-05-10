import numpy as np
from RegressionTree import RegressionTree
import matplotlib.pyplot as plt


fig, ax = plt.subplots()
ax.autoscale(False)


x_coords = np.empty(0)
y_coords = np.empty(0)
x_mat=np.array([x_coords])
y_mat=np.array([y_coords])
plt.suptitle('Klikom se dodaju tacke\n pritiskom na Enter se dobija rezultat\n pritiskom na r se resetuje kanvas', fontsize=11)
regression_tree_drawn=False

def on_click(event):
    global x_coords,y_coords
    global x_mat, y_mat, regression_tree_drawn
    if event.inaxes == ax and not regression_tree_drawn:
        x = event.xdata
        y = event.ydata
        ax.scatter(x, y)
        fig.canvas.draw()
        
        x_coords = np.append(x_coords, [[x]])
        y_coords=np.append(y_coords, [[y]])
        x_mat = np.array([x_coords]).T
        y_mat = np.array([y_coords]).T



def on_key(event):
    global x_coords,y_coords
    global x_mat, y_mat, regression_tree_drawn
    if event.key == 'enter':
        model = RegressionTree(max_depth=10, min_samples_leaf=5)
        model.fit(x_mat, y_mat)
        predictions = model.predict(x_mat)
        ax.plot(x_mat, predictions, color='black', linewidth=2, label='Prediction')
        fig.canvas.draw()
        regression_tree_drawn=True
    elif event.key == 'r':
        ax.cla()
        ax.autoscale(False)
        fig.canvas.draw()
        x_coords = np.empty(0)
        y_coords = np.empty(0)
        x_mat=np.array([x_coords])
        y_mat=np.array([y_coords])
        regression_tree_drawn= False

cid = fig.canvas.mpl_connect('button_press_event', on_click)
cid2 = fig.canvas.mpl_connect('key_press_event', on_key)
plt.show()
