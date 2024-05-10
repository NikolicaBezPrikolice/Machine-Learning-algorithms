import numpy as np

class SVM:

    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        y_ = np.where(y <= 0, -1, 1)

        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]


    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)


if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from sklearn import datasets
    import matplotlib.pyplot as plt

    
    fig, ax = plt.subplots()
    ax.autoscale(False)

    xy_coords = np.empty((0,2))
    
    label=np.empty(0)
    x_mat=np.array([[xy_coords]])
    plt.suptitle('Levim klikom se dodaju plave tacke desnim crvene\n pritiskom na r se resetuje kanvas', fontsize=11)
    svm_drawn=False

    def on_click(event):
        global xy_coords, label, svm_drawn
        global x_mat, label_mat
        if event.inaxes == ax and not svm_drawn:
            x = event.xdata
            y = event.ydata
            
            xy_coords = np.append(xy_coords, [[x,y]],axis=0)
        
            if event.button == 1:
                label = np.append(label, [[-1]])
                ax.scatter(x, y, color="blue")
            elif event.button == 3:
                label = np.append(label, [[1]])
                ax.scatter(x, y, color="red")
            fig.canvas.draw()

    def on_key(event):
        global svm_drawn, xy_coords, label
        if event.key == 'enter':
            clf = SVM(learning_rate=0.01, lambda_param=0.001, n_iters=1000)
            clf.fit(xy_coords, label)
            predictions = clf.predict(xy_coords)

            def get_hyperplane_value(x, w, b, offset):
                return (-w[0] * x + b + offset) / w[1]

            x0_1 = np.amin(xy_coords[:, 0])
            x0_2 = np.amax(xy_coords[:, 0])

            x1_1 = get_hyperplane_value(x0_1, clf.w, clf.b, 0)
            x1_2 = get_hyperplane_value(x0_2, clf.w, clf.b, 0)

            x1_1_m = get_hyperplane_value(x0_1, clf.w, clf.b, -1)
            x1_2_m = get_hyperplane_value(x0_2, clf.w, clf.b, -1)

            x1_1_p = get_hyperplane_value(x0_1, clf.w, clf.b, 1)
            x1_2_p = get_hyperplane_value(x0_2, clf.w, clf.b, 1)

            ax.plot([x0_1, x0_2], [x1_1, x1_2], "y--")
            ax.plot([x0_1, x0_2], [x1_1_m, x1_2_m], "k")
            ax.plot([x0_1, x0_2], [x1_1_p, x1_2_p], "k")
            fig.canvas.draw()
            svm_drawn=True
        elif event.key == 'r':
            ax.cla()
            ax.autoscale(False)
            fig.canvas.draw()
            xy_coords = np.empty((0,2))
            label=np.empty(0)
            svm_drawn= False 

    cid = fig.canvas.mpl_connect('button_press_event', on_click)
    cid2 = fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()