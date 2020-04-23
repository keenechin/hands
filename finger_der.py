#%%
import numpy as np
from itertools import count
import random
from numba import njit
import der
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as mline
from functools import partial

#%%
def transform2d(rot_ang,trans_vec):
    o = rot_ang
    dx,dy = trans_vec
    c = np.cos(o)
    s = np.sin(o)
    frame = np.array( [[c, -s, dx],\
                       [s,  c, dy],\
                       [0,  0,  1]])
    return frame


def pt_h(point_list):
    homogenous_arr = []
    for point in point_list:
        x,y = (point[0],point[1])
        homogenous_arr.append(np.array([[x,y,1]]).T)
    return np.array(homogenous_arr)


#%%
@njit
def kinematics_builder(v_kinematic_parameters):
    n_points = len(v_kinematic_parameters)+1
    v_points0 = np.zeros((n_points,3))
    v_frames = np.zeros((n_points,3,3))
    for i in range(n_points):
        pass
    return v_frames



# %%
@njit
def nodes_from_morphology(v_morphology_parameters):
    n_nodes = len(v_morphology_parameters)
    node_params = np.zeros((n_nodes,2)) #phi/u s/u
    for i in range(n_nodes):
        node_params[i,:] = 0


# %%
def setup():
    rect = patches.Rectangle((0.5,0.5),0.1,0.1,color = 'r')
    return rect


def update_rect(frame,viewport,rect):
    x,y = rect.get_xy()
    x+= -0.01+random.random()/50
    y+= -0.01+random.random()/50
    rect.set_xy((x,y))
    viewport.add_patch(rect)
    return rect,

 #%%   
v_compliance = np.array([[0,2,0,4,0]]).T
v_max_phi = np.array([[0,90,0,90,0]]).T

v_morphology_parameters = np.hstack((v_compliance,v_max_phi))

u = 0

#%%
lengths = [30,10,31,11,32]
n_rods = len(lengths)
n_points = n_rods+1
#%%
tf_params0 = np.zeros((n_rods,2))
tf_params0[:,1] = lengths

#%%
def get_matrices(tf_params):
    mats = []
    for param in tf_params:
        phi0 = param[0]
        s0 = [param[1],0]
        mat = transform2d(phi0,s0)
        mats.append(mat)
    mats = np.array(mats)
    return mats

mats = get_matrices(tf_params0)
#get initial set of points based on initial parameters
points0 = pt_h([(0,0)])
pt = points0[0]
for mat in mats:
    pt = mat@pt
    points0 = np.append(points0,np.array([pt]),axis=0)


def draw_lines(ax,points):
    line_artists = []
    x = points[:,0]
    y = points[:,1]
    for i in range(len(points)-1):
        line = mline.Line2D(xdata=[points[i][0],points[i+1][0]],ydata=[points[i][1],points[i+1][1]])
        if i%2 == 1:
            line.set_color('r')
            line.set_linewidth('2')
        else:
            line.set_linewidth('4')
        line_artists.append(line)
    
    for line in [line_artists[i] for i in [0,2,4,1,3]]:
        ax.add_line(line)
    ax.set_xlim(min(x)-10,max(x)+10)
    ax.set_ylim(min(y)-10,max(y)+10)
    return line_artists
    



def transform_points(matrices,points):
    updated_points = np.zeros_like(points)
    while True:
        for i,mat in enumerate(matrices):
            for j,point in enumerate(points[i:,:]):
                u_pt = mat@point
                updated_points[j,:] = u_pt
        yield updated_points

def update_frame(points,ax,line_artists):
    updated_line_artists = []
    all_x = []
    all_y = []
    for i,line in enumerate(line_artists):
        xs,ys = ([points[i][j],points[i+1][j]] for j in range(2))
        all_x.append(xs)
        all_y.append(ys)
        line.set_data(xs,ys)

        updated_line_artists.append(line)
    all_x = np.array(all_x).flatten()
    all_y = np.array(all_y).flatten()
    ax.set_xlim(min(all_x)-10,max(all_x)+10)
    ax.set_ylim(min(all_y)-10,max(all_y)+10)

    return updated_line_artists



#%%
fig = plt.figure()
viewport = fig.add_axes([0,0,1,1],frameon=False,label ="viewport",facecolor = 'k')
viewport.xaxis.set_visible(False)
viewport.yaxis.set_visible(False)

end_tf = get_matrices([[0,0],[np.pi/4,0],[np.pi/4,0],[np.pi/4,0],[np.pi/4,0]]) 
updated_points = points0

for i,mat in enumerate(end_tf):
    print(mat)
    frame_origin = updated_points[i,:]
    print(frame_origin)
    for j,point in enumerate(updated_points[i:,:]):
        adjusted_point = point-frame_origin
        u_pt = mat@adjusted_point
        updated_points[i+j,:] = u_pt+frame_origin
draw_lines0 = partial(draw_lines,viewport,updated_points)
draw_lines0()

# %%
fps = 1
writer = animation.writers['ffmpeg']
writer = writer(fps = 30, metadata=dict(artist='Keene Chin'), bitrate=10000)
#ani = animation.FuncAnimation(fig,func=update_frame,interval=1000//fps,init_func=draw_lines0,frames=transform_points(mats,points0),blit=True,fargs=[viewport,draw_lines0()])
#ani.save('./test2.mp4', writer=writer)
plt.show()
# %%


