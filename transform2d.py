import numpy as np


def transform2d(rot_ang,trans_vec):
    o = rot_ang
    dx,dy = trans_vec
    c = np.cos(o)
    s = np.sin(o)
    frame = np.array( [[c, -s, dx],\
                       [s,  c, dy],\
                       [0,  0,  1] ])
    return frame


def pt_h(x,y):
    return np.array([[x,y,1]]).T


tf = transform2d(np.pi,(1,0))
pt = pt_h(1,1)
print(tf@pt)
