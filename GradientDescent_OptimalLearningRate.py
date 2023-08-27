import numpy as np
import matplotlib.pyplot as plt

"""
Visualising rapid gradient descent when using optimal learning rate in 3D
"""

def z_func(x,y):
    return x**2+5*y**2

def calc_gradient(x,y):
    return 2*x,10*y

x=np.linspace(-40,40,1000)
y=x

X,Y=np.meshgrid(x,y)

Z=z_func(X,Y)

current_pos1=(-50,-50,z_func(-50,-50))

def learning_rate(x,y):
    return (1/2)*(x**2+25*y**2)/(x**2+125*y**2)

ax=plt.axes(projection="3d",computed_zorder=False)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

for i in range(50):  #NOTE THE DECREASE IN NUMBER OF TIMES THE LOOP IS RUN
    X_grad,Y_grad=calc_gradient(current_pos1[0],current_pos1[1])
    lr=learning_rate(current_pos1[0],current_pos1[1])
    X_new,Y_new=current_pos1[0]-lr*X_grad,current_pos1[1]-lr*Y_grad
    current_pos1=(X_new,Y_new,z_func(X_new,Y_new))

    ax.plot_surface(X,Y,Z,cmap="viridis",zorder=0)
    ax.scatter(current_pos1[0],current_pos1[1],current_pos1[2],color="magenta")

    plt.pause(0.1)  #ALSO THE INCREASE IN PAUSE TIME TO SEE THE VISUALISATION
    plt.cla()

print(f"x*:{current_pos1[0]} and y*:{current_pos1[1]}")

"""
Note the high precision in just 50 iterations
"""
