#%%
import numpy as np
import random
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as mline
from functools import partial
import skopt
from scipy.spatial.distance import cdist
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

    updated_points = np.copy(points0)

    for i,mat in enumerate(end_tf):
        frame_origin = updated_points[i,:]
        for j,point in enumerate(updated_points[i:,:]):
            adjusted_point = point-frame_origin
            u_pt = mat@adjusted_point
            updated_points[i+j,:] = u_pt+frame_origin
    return updated_points

def generate_point_frames(points0, compl, max_angle, u_start, u_end, n_frames):
    assert(u_end>u_start)
    o1,o2 = 0,0
    max_kappa = 2*np.arctan2(max_angle[0],2), 2*np.arctan2(max_angle[1],2)
    for u in np.linspace(u_start,u_end,n_frames):
        curves1 = [compl[0]*u/50., max_kappa[0]/2.]
        curves2 = [compl[1]*u/50., max_kappa[1]/2.]
        kappa1 = min(curves1)
        kappa2 = min(curves2) 
        o1 = 2*np.tan(kappa1/2)
        o2 = 2*np.tan(kappa2/2)
        #print(np.degrees(o2))
        points = transform_points(points0,theta1=o1, theta2=o2)
        #print(points)
        yield points

def initialize_points(total_length, joint_length,l0):
    l1 = (total_length - (2*joint_length + l0))/2
    l2 = l1
    lengths = [l0,joint_length,l1,joint_length,l2]
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
    return points0

def simulate_trajectory(points0, compl1, compl2 , max_ang1, max_ang2, u_start = 0, u_end = 10, n_frames = 100):
    points_generator = generate_point_frames(points0, compl = (compl1, compl2),  max_angle = (max_ang1, max_ang2), u_start = u_start, u_end = u_end, n_frames = n_frames)
    tip_trajectory = []
    for points in points_generator:
        tip_trajectory.append([points[-1][0],points[-1][1]])
    tip_trajectory = np.array(tip_trajectory)
    return np.squeeze(tip_trajectory, axis=2)


def trajectory_cost(trajectory,waypoints):
    endpoint_cost = np.linalg.norm([trajectory[-1,0]-waypoints[-1,0], trajectory[-1,1]-waypoints[-1,1]])
    trajectory_distance_mat = cdist(trajectory,waypoints)
    waypoint_min_dists = np.min(trajectory_distance_mat,axis = 0)
    trajectory_cost = np.sum(waypoint_min_dists[:])
    cost = 0.5*endpoint_cost + 0.5*trajectory_cost 
    return cost

def get_params_cost(x,total_length,joint_length, waypoints):
    l0, compl1, compl2, max_ang1, max_ang2 = x
    points = initialize_points(total_length,joint_length,l0)
    trajectory = simulate_trajectory(points, compl1, compl2, max_ang1, max_ang2)
    cost = trajectory_cost(trajectory, waypoints)
    return cost

#%%
visualize = False

total_length, joint_length, l0 = 100, 10 , 30 
compl1 = 2
compl2 = 4
max_ang1 = np.pi
max_ang2 = np.pi


waypoints = np.array([[40,20],[0,20]])

x_initial = [l0,compl1,compl2,max_ang1,max_ang2]
#%%
points0 = initialize_points(total_length, joint_length, l0)
trajectory0 = simulate_trajectory(points0, compl1, compl2, max_ang1, max_ang2)
cost0 = trajectory_cost(trajectory0,waypoints)

#%%
objective = partial(get_params_cost,total_length = 100, joint_length = 10,waypoints=waypoints)
opt_bounds = [(2,98),(0.1,40),(0.1,40),(0,2*np.pi),(0,2*np.pi)]
res = skopt.gp_minimize(objective,opt_bounds,x0 = x_initial)
x_opt = res.x
cost_opt = res.fun
print("optimal x: {}\nfinal_cost: {}\n".format(x_opt,cost_opt))
#%%

points_opt = initialize_points(total_length, joint_length, x_opt[0])
trajectory_opt = simulate_trajectory(points_opt, x_opt[1], x_opt[2], x_opt[3], x_opt[4])
cost = trajectory_cost(trajectory_opt,waypoints)
#%%
for point in trajectory0:
    plt.scatter(point[0],point[1],c='r', s = 4)
for point in trajectory_opt:
    plt.scatter(point[0],point[1],c='b', s = 4)
plt.scatter(waypoints[:,0],waypoints[:,1],c='g',s=144, marker='*')

plt.show()

#%%
if visualize:
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
    fps = 60
    #writer = animation.writers['ffmpeg']
    writer = animation.FFMpegFileWriter(fps = fps, metadata=dict(artist='Keene Chin'), bitrate=1000)

    ani = animation.FuncAnimation(fig,func=update_frame, interval=1000//fps, frames=points_generator, blit=True, fargs=[viewport, artists])
    ani.save('./finger_l{}_c{}_c{}.mp4'.format(l0, compl1, compl2), writer=writer)
# %%


