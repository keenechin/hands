#%%
import numpy as np
import random
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

def get_matrices(tf_params):
    mats = []
    for param in tf_params:
        phi0 = param[0]
        s0 = [param[1],0]
        mat = transform2d(phi0,s0)
        mats.append(mat)
    mats = np.array(mats)
    return mats

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
        

    return line_artists
    

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
    end_segment = updated_line_artists[-1].get_xydata()
    end_point = end_segment[1,:]
    plt.scatter(end_point[0],end_point[1],c='g')
    updated_line_artists[1],updated_line_artists[4] = updated_line_artists[4],updated_line_artists[1]#put red joint at end so it is drawn on top
    line_artists = updated_line_artists
    return line_artists

def transform_points(points0,theta1=0,theta2=0):
    end_tf = get_matrices([[0,0],[theta1,0],[theta1,0],[theta2,0],[theta2,0]]) 

    updated_points = points0

    for i,mat in enumerate(end_tf):
        frame_origin = updated_points[i,:]
        for j,point in enumerate(updated_points[i:,:]):
            adjusted_point = point-frame_origin
            u_pt = mat@adjusted_point
            updated_points[i+j,:] = u_pt+frame_origin
    return updated_points

def theta_frames(points0,u_start = 0, u_end = 4, compl1=2, compl2=4, n_frames = 30):
    assert(u_end>u_start)
    o1,o2 = 0,0
    for u in np.linspace(u_start,u_end,n_frames):
        kappa1 = compl1*u/500
        kappa2 = compl2*u/500
        o1 = 2*np.tan(kappa1/2)
        o2 = 2*np.tan(kappa2/2)
        points = transform_points(points0,theta1=o1,theta2=o2)
        yield points



#%%
total_length = 90
l0 = 15
l1 = (total_length - l0)/2
l2 = l1
lengths = [l0,10,l1,10,l2]
total_length = sum(lengths)
n_rods = len(lengths)
n_points = n_rods+1
tf_params0 = np.zeros((n_rods,2))
tf_params0[:,1] = lengths
mats = get_matrices(tf_params0)

#get initial config of nodes
points0 = pt_h([(0,0)])
pt = points0[0]
for mat in mats:
    pt = mat@pt
    points0 = np.append(points0,np.array([pt]),axis=0)



fig= plt.figure(frameon=False)
viewport = fig.add_axes([0,0,1,1])
viewport.xaxis.set_visible(False)
viewport.yaxis.set_visible(False)
bound = total_length+10
viewport.axis('equal')
viewport.set_xlim(-bound,bound)
viewport.set_ylim(-bound,bound)
draw_lines0 = partial(draw_lines,viewport,points0)
artists = draw_lines0()
# %%
fps = 15
writer = animation.writers['ffmpeg']
writer = writer(fps = fps, metadata=dict(artist='Keene Chin'), bitrate=1000)
compl1 = 2
compl2 = 2
ani = animation.FuncAnimation(fig,func=update_frame,interval=1000//fps,frames=theta_frames(points0,compl1=compl1,compl2=compl2),blit=True,fargs=[viewport,artists])
ani.save('./finger_l{}_c{}_c{}.mp4'.format(l0,compl1,compl2), writer=writer)
# %%


