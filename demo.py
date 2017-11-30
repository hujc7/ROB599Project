#! /usr/bin/python3
from glob import glob
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import sys

def rot(n, theta):
    n = n / np.linalg.norm(n, 2)
    K = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
    return np.identity(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K


def get_bbox(p0, p1):
    '''
    Input:
    *   p0, p1
        (3)
        Corners of a bounding box represented in the body frame.

    Output:
    *   v
        (3, 8)
        Vertices of the bounding box represented in the body frame.
    *   e
        (2, 14)
        Edges of the bounding box. The first 2 edges indicate the `front` side
        of the box.
    '''
    v = np.array([[p0[0], p0[0], p0[0], p0[0], p1[0], p1[0], p1[0], p1[0]],
                  [p0[1], p0[1], p1[1], p1[1], p0[1], p0[1], p1[1], p1[1]],
                  [p0[2], p1[2], p0[2], p1[2], p0[2], p1[2], p0[2], p1[2]]])
    e = np.array([[2, 3, 0, 0, 3, 3, 0, 1, 2, 3, 4, 4, 7, 7],
                  [7, 6, 1, 2, 1, 2, 4, 5, 6, 7, 5, 6, 5, 6]], dtype=np.uint8)

    return v, e


classes = ['Unknown', 'Compacts', 'Sedans', 'SUVs', 'Coupes',
           'Muscle', 'SportsClassics', 'Sports', 'Super', 'Motorcycles',
           'OffRoad', 'Industrial', 'Utility', 'Vans', 'Cycles',
           'Boats', 'Helicopters', 'Planes', 'Service', 'Emergency',
           'Military', 'Commercial', 'Trains']

# find all files in deploy
files = glob('deploy/trainval/*/*_image.jpg')
idx = np.random.randint(0, len(files))
# pick a random one
snapshot = files[idx]
print(snapshot)

img = plt.imread(snapshot)

# change the file name to get cloud.bin
# clound contain cloud points
xyz = np.memmap(snapshot.replace('_image.jpg', '_cloud.bin'), dtype=np.float32)
xyz.resize([3, xyz.size // 3])
# print(xyz)
# print(xyz.shape)
# sys.exit()


# get projection matrix
proj = np.memmap(snapshot.replace('_image.jpg', '_proj.bin'), dtype=np.float32)
proj.resize([3, proj.size // 3])
# print(proj)
# print(proj.shape)
# sys.exit()

try:
    bbox = np.memmap(snapshot.replace('_image.jpg', '_bbox.bin'), dtype=np.float32)
except:
    print('[*] bbox not found.')
    bbox = np.array([], dtype=np.float32)
bbox.resize([bbox.size // 11, 11])
print(bbox)
print(bbox.shape)
# sys.exit()

# project cloud points onto image, get 2d coordinate
uv = proj @ np.vstack([xyz, np.ones_like(xyz[0, :])])
uv = uv / uv[2, :]

clr = np.linalg.norm(xyz, axis=0)
fig1 = plt.figure(1, figsize=(16, 9))
ax1 = fig1.add_subplot(1, 1, 1)
ax1.imshow(img)
ax1.scatter(uv[0, :], uv[1, :], c=clr, marker='.', s=1)
ax1.axis('scaled')
fig1.tight_layout()

fig2 = plt.figure(2, figsize=(8, 8))
ax2 = Axes3D(fig2)
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('z')

step = 5
# draw 3d clound points, every 5 points
ax2.scatter(xyz[0, ::step], xyz[1, ::step], xyz[2, ::step], \
    c=clr[::step], marker='.', s=1)

colors = ['C{:d}'.format(i) for i in range(10)]
for k, b in enumerate(bbox):
	print('bounding box:')
	print(bbox)
	# get rotation and translation 
	# b[0:3]: theta*k
	# b[3:6]: translation
	n = b[0:3]
	theta = np.linalg.norm(n)
	n /= theta # normalized ?
	R = rot(n, theta)
	t = b[3:6]

	# size of the bbox
	sz = b[6:9]
	print('sz:')
	print(sz)
	print(-sz/2)
	print(sz/2)
	print()

	sys.exit()

	# 3d bounding box coordinates in body frame
	vert_3D, edges = get_bbox(-sz / 2, sz / 2)
	# 3d bounding box coordinates in global frame
	vert_3D = R @ vert_3D + t[:, np.newaxis]

	vert_2D = proj @ np.vstack([vert_3D, np.ones(8)])
	vert_2D = vert_2D / vert_2D[2, :]

	clr = colors[k % len(colors)]
	for e in edges.T:
	    ax1.plot(vert_2D[0, e], vert_2D[1, e], color=clr)
	    ax2.plot(vert_3D[0, e], vert_3D[1, e], vert_3D[2, e], color=clr)

	c = classes[int(b[9])]
	ignore_in_eval = bool(b[10])
	if ignore_in_eval:
	    ax2.text(t[0], t[1], t[2], c, color='r')
	else:
	    ax2.text(t[0], t[1], t[2], c)

ax2.auto_scale_xyz([-40, 40], [-40, 40], [0, 80])
ax2.view_init(elev=-30, azim=-90)

for e in np.identity(3):
    ax2.plot([0, e[0]], [0, e[1]], [0, e[2]], color=e)

plt.show()
