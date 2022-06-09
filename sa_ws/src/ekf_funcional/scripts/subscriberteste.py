#!/usr/bin/env python
import rospy
import numpy as np
import math
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import *
from tf2_msgs import *
from tf.transformations import euler_from_quaternion
#import scipy.linalg
from cmath import pi

range_ar = np.zeros((726, 1))
pos = []
ori = []


#lidar_angles = np.linspace(-5, 5, 640)

#LINE FUNCTIONS

class Thresholds:
    def __init__(self):
        self.seg_min_length = 0.01
        self.point_dist = 0.05
        self.min_point_seg = 50

def fitline(pontos):
    # centroid de pontos considerando que o centroid de
    # um numero finito de pontos pode ser obtido como
    # a media de cada coordenada

    lixo, len = pontos.shape
    # alpha = np.zeros((1,1))

    xc, yc = pontos.sum(axis=1) / len
    dx = (pontos[0, :] - xc)
    dy = (pontos[1, :] - yc)

    num = -2 * np.matrix.sum(np.multiply(dx, dy))
    denom = np.matrix.sum(np.multiply(dy, dy) - np.multiply(dx, dx))
    alpha = math.atan2(num, denom) / 2

    r = xc * math.cos(alpha) + yc * math.sin(alpha)

    if r < 0:
        alpha = alpha + math.pi
        if alpha > pi:
            alpha = alpha - 2 * math.pi
        r = -r

    return alpha, r


def compdistpointstoline(xy, alpha, r):
    xcosa = xy[0, :] * math.cos(alpha)
    ysina = xy[1, :] * math.sin(alpha)
    d = xcosa + ysina - r
    return d


def findsplitposid(d, thresholds):
    # implementaçao simples
    # print('d = ', end = '')
    # print(d)
    N = d.shape[1]
    # print('N =', end='')
    # print(N)

    d = abs(d)
    # print(d)
    mask = d > thresholds.point_dist
    # print('mask =', end='')
    # print(mask)
    if not np.any(mask):
        splitpos = -1
        return splitpos

    splitpos = np.argmax(d)
    # print(splitpos)
    if (splitpos == 0):
        splitpos = 1
    if (splitpos == (N - 1)):
        splitpos = N - 2
    return splitpos


def findsplitpos(xy, alpha, r, thresholds):
    d = compdistpointstoline(xy, alpha, r)
    splitpos = findsplitposid(d, thresholds)
    return splitpos


def splitlines(xy, startidx, endidx, thresholds):
    N = endidx - startidx + 1

    alpha, r = fitline(xy[:, startidx:(endidx + 1)])

    if N <= 2:
        idx = [startidx, endidx]
        return alpha, r, idx

    splitpos = findsplitpos(xy[:, startidx:(endidx + 1)], alpha, r, thresholds)
    # print(splitpos)
    if (splitpos != -1):
        alpha1, r1, idx1 = splitlines(xy, startidx, splitpos + startidx, thresholds)  # se calhar start idx-1
        alpha2, r2, idx2 = splitlines(xy, splitpos + startidx, endidx, thresholds)
        alpha = np.vstack((alpha1, alpha2))
        r = np.vstack((r1, r2))
        idx = np.vstack((idx1, idx2))
    else:
        idx = np.array([startidx, endidx])

    return alpha, r, idx


def mergeColinear(xy, alpha, r, pointidx, thresholds):
    z = [alpha[0, 0], r[0, 0]]
    startidx = pointidx[0, 0]
    lastendidx = pointidx[0, 1]

    N = r.shape[0]
    zt = [0, 0]

    # rOut = np.zeros((r.shape[0],1))
    # alphaOut = np.zeros((alpha.shape[0], 1))
    # pointidxOut = np.zeros((1, 2))
    rOut = []
    alphaOut = []
    pointidxOut = []

    j = 0

    for i in range(1, N):
        endidx = pointidx[i, 1]
        # print(z)
        zt[0], zt[1] = fitline(xy[:, startidx:(endidx + 1)])

        splitpos = findsplitpos(xy[:, startidx:(endidx + 1)], zt[0], zt[1], thresholds)
        zt[1] = np.matrix.item(zt[1])
        # Se nao for necessario fazer split, fazemos merge
        # print(zt[1])
        if splitpos == -1:
            z = zt
        else:  # Sem mais merges
            # alphaOut[j, 0] = z[0]
            alphaOut.append(z[0])
            # print(z)
            # print(z[1][0, 0])
            # list = np.matrix.tolist(z[1])
            # print(list)
            rOut.append(z[1])
            # print(rOut)
            # rOut[j, 0] = z[1]
            pointidxOut.extend([startidx, lastendidx])
            # pointidxOut = np.vstack((pointidxOut,[startidx, lastendidx]))
            j = j + 1
            z = [alpha[i, 0], r[i, 0]]
            startidx = pointidx[i, 0]

        lastendidx = endidx

    # Adicionar o ultimo segmento
    alphaOut.append(z[0])
    rOut.append(z[1])
    pointidxOut.extend([startidx, lastendidx])

    pointidxOut = np.array(pointidxOut)
    pointidxOut = np.reshape(pointidxOut, (j + 1, 2))
    alphaOut = np.array(alphaOut)
    alphaOut = np.reshape(alphaOut, (j + 1, 1))
    rOut = np.array(rOut)
    rOut = np.reshape(rOut, (j + 1, 1))
    rOut = np.asmatrix(rOut)
    # print(rOut)

    return alphaOut, rOut, pointidxOut


def pol2cart(theta, rho):
    x = np.zeros((1, theta.shape[0]))
    y = np.zeros((1, theta.shape[0]))
    for i in range(0, theta.shape[0]):
        x[0, i] = rho[i, 0] * np.cos(theta[i, 0])
        y[0, i] = rho[i, 0] * np.sin(theta[i, 0])
    return x, y


def extractlines(theta, rho, thersholds):
    # passa de coordenadas polares para cartesianas

    x, y = pol2cart(theta, rho)

    xy = np.vstack((x, y))

    # xy = np.concatenate((x,y),axis=0)
    xy = np.asmatrix(xy)

    # print(xy)

    startidx = 0
    endidx = xy.shape[1] - 1  # x e y são vetores linha

    # faz a extracao das linhas
    alpha, r, pointsidx = splitlines(xy, startidx, endidx, thersholds)

    # numero de segmentos de reta, caso seja mais do que um segmento, vereifica se sao colineares
    n = r.shape[0]
    if n > 1:
        alpha, r, pointsidx = mergeColinear(xy, alpha, r, pointsidx, thersholds)
        # HA AQUI UM PROBLEMA NO R
        n = r.shape[0]
        # atualiza o numero de segmentos

    # definir coordenads dos endpoints e len dos segmentos
    segmends = np.zeros((n, 4))
    segmlen = np.zeros((n, 1))
    # for l in range(0, n):
    #    print(np.concatenate([np.transpose(xy[:, pointsidx[l, 0]]), np.transpose(xy[:, pointsidx[l, 1]])], axis = 1))
    pointsidx = np.asmatrix(pointsidx)
    for l in range(0, n):
        segmends[l, :] = np.concatenate([np.transpose(xy[:, pointsidx[l, 0]]), np.transpose(xy[:, pointsidx[l, 1]])],
                                        axis=1)
        # segmends[l, :] = [np.transpose(xy[:, pointsidx[l, 0]]), np.transpose(xy[:, pointsidx[l, 1]])]
        # for j in range(0:4):
        #    segmends[l, j] = [xy[j, pointsidx[l, 0]]]
        segmlen[l] = math.sqrt((segmends[l, 0] - segmends[l, 2]) ** 2 + (segmends[l, 1] - segmends[l, 3]) ** 2)

    segmlen = np.transpose(segmlen)
    # print(((pointsidx[:,1] - pointsidx[:,0]) >= thersholds.min_point_seg))
    # print((segmlen >= thersholds.seg_min_length))
    # print((segmlen >= thersholds.seg_min_length) & ((pointsidx[:,1] - pointsidx[:,0]) >= thersholds.min_point_seg))

    # remover segmentos demasiados pequenos
    # alterar thersholds para params.MIN_SEG_LENGTH e params.MIN_POINTS_PER_SEGMENT
    goodsegmidx = np.argwhere(
        np.transpose(segmlen >= thersholds.seg_min_length) & (
                    (pointsidx[:, 1] - pointsidx[:, 0]) >= thersholds.min_point_seg))
    # print(goodsegmidx)
    # goodsegmix2 = goodsegmidx[0, 1]:goodsegmidx[(goodsegmidx.shape[0]), 1]
    # print(goodsegmix2)

    '''
    print('1a condicao')
    print(segmlen >= thersholds.seg_min_length)
    print('2a condicao')
    print((pointsidx[:, 1] - pointsidx[:, 0]) >= thersholds.min_point_seg)
    print('and')
    print(
        np.transpose(segmlen >= thersholds.seg_min_length) & ((pointsidx[:, 1] - pointsidx[:, 0]) >= thersholds.min_point_seg))

    print('goodsegmidx')
    print(goodsegmidx)
    '''
    pointsidx = pointsidx[goodsegmidx[:, 0], :]

    # print(pointsidx)

    alpha = np.asmatrix(alpha)
    alpha = alpha[goodsegmidx[:, 0], 0]
    # r = np.asmatrix(r)
    # print(r)
    r = r[goodsegmidx[:, 0], 0]
    # print(segmends)
    segmends = segmends[goodsegmidx[:, 0], :]
    segmlen = np.transpose(segmlen)
    segmlen = segmlen[goodsegmidx[:, 0], 0]

    # print(alpha)
    # print(r)
    # z = np.zeros((alpha.shape[0] - 1, r.shape[0] - 1))
    z = np.transpose(np.hstack((alpha, r)))  # mudei para hstack
    #z = np.asarray(z)


    R_seg = np.zeros((2, 2, alpha.shape[0]))
    for coco in range(0,alpha.shape[0]):
        #R_seg[0, 0, coco] = 0.01
        #R_seg[1, 1, coco] = 0.1
        for j in range(0,2):
            R_seg[j,j,coco] = 0.5


    #R_seg = 0.1*np.identity(2)


    return z, R_seg, segmends


def normalizelineparameters(alpha, r):
    if r < 0:
        alpha = alpha + pi
        r = -r
        isRNegated = 1
    else:
        isRNegated = 0

    if alpha > math.pi:
        alpha = alpha - 2 * math.pi
    elif alpha < -math.pi:
        alpha = alpha + 2 * math.pi

    return alpha, r, isRNegated



# ROS FUNCTIONS

def callback(msg):
    #rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)
    #print(type(msg.ranges))
    #print(len(msg.ranges))
    #return len(msg.ranges)
    global range_ar
    range_ar = np.asarray(msg.ranges)

def callback2(msg):
    #print(msg.pose.pose)
    #return(msg.pose.pose)
    global pos, ori
    pos = msg.pose.pose.position
    ori = msg.pose.pose.orientation


if __name__ == '__main__':
    rospy.init_node('monkey')
    #rate = rospy.Rate(10)

    call1 = rospy.Subscriber('/scan', LaserScan, callback)
    call2 = rospy.Subscriber('/pose', Odometry, callback2)
    i = 0

    rate = rospy.Rate(10)
    while not rospy.is_shutdown():

        """
        print('Range length = ', range_ar.shape)
        print(pos)
        range_ar[2] = 9
        print('---------------------------------------------------------')
        """
        thresholds = Thresholds()
        f = 0
        dist = np.zeros((726, 1))
        thetas = np.zeros((726, 1))
        lidar = np.linspace(-2.356194496154785, 2.0923497676849365, 726)
        for k in range(0, 726):
            dist[f] = range_ar[k]
            thetas[f] = lidar[k]
            if range_ar[k] == float('inf') or math.isnan(range_ar[k]):
                dist = np.delete(dist, f)
                thetas = np.delete(thetas, f)
                f -= 1
            f += 1
        #print(dist.shape)
        #print(thetas.shape)
        dist = np.transpose(np.asmatrix(dist))
        thetas = np.transpose(np.asmatrix(thetas))
        #z, Q, segends = extractlines(thetas, dist, thresholds)
        print(i)
        i += 1


        rate.sleep()