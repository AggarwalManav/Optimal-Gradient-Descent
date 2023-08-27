import numpy as np
import matplotlib.pyplot as plt

"""
Visualising the slow gradient descent process when using non-optimal learning rate
"""
def z_func(x,y):
    return x**2+5*y**2

def calc_gradient(x,y):
    return 2*x,10*y

x=np.linspace(-40,40,1000)
y=x

X,Y=np.meshgrid(x,y)

Z=z_func(X,Y)

current_pos1=(-40,-40,z_func(-40,-40))

learning_rate=0.01

ax=plt.axes(projection="3d",computed_zorder=False)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

for i in range(1000):
    print(f"iteration {i}")
    X_grad,Y_grad=calc_gradient(current_pos1[0],current_pos1[1])
    X_new,Y_new=current_pos1[0]-learning_rate*X_grad,current_pos1[1]-learning_rate*Y_grad
    current_pos1=(X_new,Y_new,z_func(X_new,Y_new))

    ax.plot_surface(X,Y,Z,cmap="viridis",zorder=0)
    ax.scatter(current_pos1[0],current_pos1[1],current_pos1[2],color="magenta")

    plt.pause(0.001)
    plt.cla()

print(f"x*:{current_pos1[0]} and y*:{current_pos1[1]}")

"""
Note that even after 1000 iterations, precision is less than 
gradient descent with optimal learning rate in just 50 iterations
"""