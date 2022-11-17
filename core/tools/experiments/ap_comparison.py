import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from scipy import interpolate
from scipy.stats import gaussian_kde

"""
这个文件用来绘制前后性能下降图
"""
name = np.array(['SECOND', 'PointPillars', 'Part-$A^2$', 'PV-RCNN', 'CenterPoint', 'Voxel R-CNN',
                 'PointRCNN', '3DSSD', '3DSSD-SASA', 'IA-SSD', 'Det6D'])


skitti_ap = np.array([37.23, 34.10, 36.92, 37.25, 36.50, 37.50,
                      39.11, 37.01, 37.28, 39.55, 73.55])
kitti_ap = np.array([76.48, 77.98, 79.47, 83.69, 79.48, 84.52,
                     78.63, 79.45, 84.80, 79.57, 84.41])
drap = kitti_ap - skitti_ap
sorted_idx = np.argsort(drap)
name = name[sorted_idx]
skitti_ap = skitti_ap[sorted_idx]
kitti_ap = kitti_ap[sorted_idx]
drap = drap[sorted_idx
]
width = 0.6

fig = plt.figure(figsize=(6.4 * 1.8, 4.8 * 0.9))
ax = plt.axes()
ax.set_facecolor('#E9E9F1')
plt.ylabel('$AP_{R40}@0.7$',fontsize=12)
plt.xlabel('the names of the detectors',fontsize=12)
plt.ylim([0, 100])
x = np.arange(len(name))
p1 = plt.bar(x, height=kitti_ap, width=width, label='KITTI', color='#C1A387')
p2 = plt.bar(x, height=skitti_ap, width=width, label='SlopedKITTI', color='#F1BE96')
plt.legend(loc='lower right')
plt.legend(loc='upper right')
ax.bar_label(p1, padding=-15, fmt='%2.2f',fontsize=11)
ax.bar_label(p2, padding=-15, fmt='%2.2f',fontsize=11)
ax.set_xticks(x, name,fontsize=11)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()

# fig = plt.figure(figsize=(6.4 * 1.8, 4.8 * 0.9))
# ax1 = plt.axes([0.05, 0.5, 0.9, 0.4])
# ax2 = plt.axes([0.05, 0.1, 0.9, 0.4])
# ax1.set_facecolor('#E9E9F1')
# ax2.set_facecolor('#E9E9F1')
# x = np.arange(len(name))
# p1 = ax1.bar(x, height=drap, width=width, label='KITTI', color='#C1A387')
# p2 = plt.bar(x, height=skitti_ap, width=width, label='SlopedKITTI', color='#F1BE96')
# plt.legend(loc='lower right')
#
# ax1.bar_label(p1, padding=3)
# # ax.bar_label(p2, padding=3)
# ax1.set_xticks(x, name)
# plt.tight_layout()
# plt.ylim([0, 100])
#
# ax2.invert_yaxis()
# plt.show()
