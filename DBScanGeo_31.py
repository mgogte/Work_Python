#!/usr/bin/env python
# -*- coding: utf-8 -*-

#####
# Code for DBScan of geolocation of users
#####

# functions from... THE FUTURE!

from __future__ import print_function
import csv
import pandas as pd
import sys
import os
import datetime
#Add matplotlib and 'Agg' to supress graphic output- Mugdha 09/14/15
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pylab as pl
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler

# options
from optparse import OptionParser

# for database operations
import sys
import MySQLdb
import pandas.io.sql as psql
import math
#save to file
now = datetime.datetime.now()
framedir = "/home/nanohub/mugdha2608/Geolocation/" + now.strftime("%Y-%m-%d-%H%M%S")
os.mkdir(framedir)

progvers = 'AUTH'
ScanName = 'DB'
fname = framedir + '/' + progvers
sname = framedir + '/' + ScanName
#DB connection--data extraction
nanohub_metrics_cn = MySQLdb.connect(user = 'hub_read', 
                                     passwd = 'N3nw3B', 
                                     db='nanohub')
cursor = nanohub_metrics_cn.cursor()

cursor.execute("select  ig.ipLATITUDE, ig.ipLONGITUDE, dayofyear(amu.datetime)                                                     \
from nanohub_metrics.andmore_usage amu,                                                                 \
nanohub_metrics.ip_geodata ig,                                                                          \
nanohub.jos_resource_assoc jra                                                                          \
where http_method = 'GET'                                                                               \
and amu.cms_action_name = jra.child_id                                                                  \
and ig.ip = amu.ip                                                                                      \
and http_return_code = 200                                                                              \
and amu.datetime>= \'2014-01-01\' and amu.datetime <= \'2014-12-31\'                                         \
and amu.cms_action_name in (select child_id from nanohub.jos_resource_assoc where parent_id = 7313)     \
group by ig.ip;")
#7708
results = cursor.fetchall()
xs=[]
ys =[]
zs = []


#3-D Scatter plot--data visualization
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
n = 100
for c, m, zl, zh in [('r', 'o', -50, -25), ('b', '^', -30, -5)]:
    xs = [x[0] for x in results]
    ys = [x[1] for x in results]
    zs = [x[2] for x in results]
    ax.scatter(xs, ys, zs, c=c, marker=m)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
pl.savefig(fname, dpi=300)
#plt.show()
X = np.asarray(results)

# Compute DBSCAN
db = DBSCAN(eps=4, min_samples=2).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print('Estimated number of clusters: %d' % n_clusters_)
#print (db.core_sample_indices_)


# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = 'k'

    class_member_mask = (labels == k)
    #print (X)
    #print (k)
    #print  (class_member_mask & core_samples_mask)
    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1],xy[:,2], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)
    print (k)
    print (xy)
    
    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], xy[:,2], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=6)
    ax.set_xlabel('Latitude')
    ax.set_ylabel('Longitude')
    ax.set_zlabel('Day of the Year')
    plt.axis([-90, 90, -180, 180])
plt.title('Estimated number of clusters for parent_id 7313: %d' % n_clusters_)
pl.savefig(sname, dpi=300)  
