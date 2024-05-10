import numpy as np


def unit_step_func(x):
    return np.where(x > 0 , 1, 0)

class Perceptron:

    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.activation_func = unit_step_func
        self.weights = None
        self.bias = None


    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        y_ = np.where(y > 0 , 1, 0)

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_func(linear_output)

                update = self.lr * (y_[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update


    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_func(linear_output)
        return y_predicted


if __name__ == "__main__":

    import matplotlib.pyplot as plt
 
    fig, ax = plt.subplots()
    ax.autoscale(False)


    xy_coords = np.empty((0,2))
    label=np.empty(0)

    plt.suptitle('Levim klikom se dodaju plave tacke desnim crvene\n pritiskom na Enter se dobija rezultat\n pritiskom na r se resetuje kanvas', fontsize=11)
    perceptron_drawn=False
    
    def on_click(event):
        global xy_coords, label, perceptron_drawn
        if event.inaxes == ax and not perceptron_drawn:
            x = event.xdata
            y = event.ydata
            xy_coords = np.append(xy_coords, [[x,y]],axis=0)

            if event.button == 1:
                label = np.append(label, [[0]])
                ax.scatter(x, y, color="blue")
            elif event.button == 3:
                label = np.append(label, [[1]])
                ax.scatter(x, y, color="red")
            fig.canvas.draw()
            

    def on_key(event):
        global perceptron_drawn, xy_coords, label
        if event.key == 'enter':
            p = Perceptron(learning_rate=0.1, n_iters=1000)
            p.fit(xy_coords, label)
            predictions = p.predict(xy_coords)

            x0_1 = np.amin(xy_coords[:, 0])
            x0_2 = np.amax(xy_coords[:, 0])
            
            x1_1 = (-p.weights[0] * x0_1 - p.bias) / p.weights[1]
            x1_2 = (-p.weights[0] * x0_2 - p.bias) / p.weights[1]
        
            ax.plot([x0_1, x0_2], [x1_1, x1_2], "k")

            fig.canvas.draw() 
            perceptron_drawn=True
        elif event.key == 'r':
            ax.cla()
            ax.autoscale(False)
            fig.canvas.draw()
            xy_coords = np.empty((0,2))
            label=np.empty(0)
            perceptron_drawn= False

    cid = fig.canvas.mpl_connect('button_press_event', on_click)
    cid2 = fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()