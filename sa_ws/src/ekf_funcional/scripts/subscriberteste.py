#!/usr/bin/env python
import rospy
import numpy as np
import math
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import *
from tf2_msgs import *
import tf2_ros
import tf
from tf.transformations import euler_from_quaternion
import scipy.linalg
from cmath import pi

range_ar = np.zeros((726, 1))
#pos = []
#ori = []
odom = [0, 0, 0]
calledodom = 0

Q_est = np.diag([
    0.1,  # variance of location on x-axis
    0.1,  # variance of location on y-axis
    np.deg2rad(2.0),  # variance of theta
]) ** 2  

mapa = np.array([[1.24104594, -1.9044985,  -0.3014955,  -1.84909599, -1.85181707,  2.85779854,
   2.83997361,  2.8674252,  -1.88708245, -1.86734214,  2.82597699,  2.86858395,
   2.83188971, -1.88913393, -1.91008894, -1.87884911],
 [0.81356476,  0.96373292,  0.77672432,  7.78147595, 14.57948038,  5.644,
   1.01533104,  2.68790192,  6.99924286,  6.80869188, 14.95263583,  5.93378429,
   5.44786612,  0.93788508,   0.89029381, 14.508295]])

#lidar_angles = np.linspace(-5, 5, 640)

#LINE FUNCTIONS

class Thresholds:
    def __init__(self):
        self.seg_min_length = 0.4 #blah blah
        self.point_dist = 0.05 #line point dist threshold
        self.min_point_seg = 25

def line_regression(pontos):

    _, len = pontos.shape

    x_c, y_c = pontos.sum(axis=1) / len
    dx = (pontos[0, :] - x_c)
    dy = (pontos[1, :] - y_c)

    num = -2 * np.matrix.sum(np.multiply(dx, dy))
    den = np.matrix.sum(np.multiply(dy, dy) - np.multiply(dx, dx))
    alpha = math.atan2(num, den) / 2

    r = x_c * math.cos(alpha) + y_c * math.sin(alpha)

    if r < 0:
        alpha = alpha + math.pi
        if alpha > pi:
            alpha = alpha - 2 * math.pi
        r = -r

    return alpha, r


def dist2line(pontos, alpha, r):
    xcosa = pontos[0, :] * math.cos(alpha)
    ysina = pontos[1, :] * math.sin(alpha)
    d = xcosa + ysina - r
    return d


def split_position_id(d, thresholds):

    n_d = d.shape[1]
    d = abs(d)

    mask = d > thresholds.point_dist

    if not np.any(mask):
        split_position = -1
        return split_position

    split_position = np.argmax(d)

    if (split_position == 0):
        split_position = 1
    if (split_position == (n_d - 1)):
        split_position = n_d - 2
    return split_position


def find_split_position(pontos, alpha, r, thresholds):
    d = dist2line(pontos, alpha, r)
    split_position = split_position_id(d, thresholds)
    return split_position


def split_lines(pontos, i_id, f_id, thresholds):
    n_p = f_id - i_id + 1

    alpha, r = line_regression(pontos[:, i_id:(f_id + 1)])

    if n_p <= 2:
        ids = [i_id, f_id]
        return alpha, r, ids

    split_position = find_split_position(pontos[:, i_id:(f_id + 1)], alpha, r, thresholds)

    if (split_position != -1):
        alpha1, r1, idx1 = split_lines(pontos, i_id, split_position + i_id, thresholds)  # se calhar start ids-1
        alpha2, r2, idx2 = split_lines(pontos, split_position + i_id, f_id, thresholds)
        alpha = np.vstack((alpha1, alpha2))
        r = np.vstack((r1, r2))
        ids = np.vstack((idx1, idx2))
    else:
        ids = np.array([i_id, f_id])

    return alpha, r, ids


def merge_lines(pontos, alpha, r, p_ids, thresholds):
    z = [alpha[0, 0], r[0, 0]]
    i_id = p_ids[0, 0]
    last_id = p_ids[0, 1]

    n_lines = r.shape[0]
    z_t = [0, 0]

    r_out = []
    alpha_out = []
    p_ids_out = []

    j = 0

    for i in range(1, n_lines):
        f_id = p_ids[i, 1]

        z_t[0], z_t[1] = line_regression(pontos[:, i_id:(f_id + 1)])

        split_position = find_split_position(pontos[:, i_id:(f_id + 1)], z_t[0], z_t[1], thresholds)
        z_t[1] = np.matrix.item(z_t[1])

        if split_position == -1:
            z = z_t
        else:
            alpha_out.append(z[0])
            r_out.append(z[1])
            p_ids_out.extend([i_id, last_id])
            j = j + 1
            z = [alpha[i, 0], r[i, 0]]
            i_id = p_ids[i, 0]

        last_id = f_id

    # Adicionar o ultimo segmento
    alpha_out.append(z[0])
    r_out.append(z[1])
    p_ids_out.extend([i_id, last_id])

    p_ids_out = np.array(p_ids_out)
    p_ids_out = np.reshape(p_ids_out, (j + 1, 2))
    alpha_out = np.array(alpha_out)
    alpha_out = np.reshape(alpha_out, (j + 1, 1))
    r_out = np.array(r_out)
    r_out = np.reshape(r_out, (j + 1, 1))
    r_out = np.asmatrix(r_out)

    return alpha_out, r_out, p_ids_out


def pol2cart(theta, rho):
    x = np.zeros((1, theta.shape[0]))
    y = np.zeros((1, theta.shape[0]))
    for i in range(0, theta.shape[0]):
        x[0, i] = rho[i, 0] * np.cos(theta[i, 0])
        y[0, i] = rho[i, 0] * np.sin(theta[i, 0])
    return x, y


def split_merge(theta, rho, thersholds):

    x, y = pol2cart(theta, rho)

    pxy = np.vstack((x, y))
    pxy = np.asmatrix(pxy)

    i_id = 0
    f_id = pxy.shape[1] - 1  # x e y são vetores linha

    # faz a extracao das linhas
    alpha, r, p_ids = split_lines(pxy, i_id, f_id, thersholds)

    # numero de segmentos de reta, caso seja mais do que um segmento, verifica se sao colineares
    n = r.shape[0]
    if n > 1:
        alpha, r, p_ids = merge_lines(pxy, alpha, r, p_ids, thersholds)
        n = r.shape[0]
        # atualiza o numero de segmentos

    # definir coordenads dos endpoints e len dos segmentos
    seg_i_f = np.zeros((n, 4))
    seg_len = np.zeros((n, 1))

    p_ids = np.asmatrix(p_ids)

    if p_ids.shape[0]!=0:
        for l in range(0, n):
            seg_i_f[l, :] = np.concatenate([np.transpose(pxy[:, p_ids[l, 0]]), np.transpose(pxy[:, p_ids[l, 1]])],
                                            axis=1)
            seg_len[l] = math.sqrt((seg_i_f[l, 0] - seg_i_f[l, 2]) ** 2 + (seg_i_f[l, 1] - seg_i_f[l, 3]) ** 2)

    seg_len = np.transpose(seg_len)

    # remover segmentos demasiados pequenos
    correct_segs_ids = np.argwhere(
        np.transpose(seg_len >= thersholds.seg_min_length) & (
                    (p_ids[:, 1] - p_ids[:, 0]) >= thersholds.min_point_seg))

    '''
    print('1a condicao')
    print(seg_len >= thersholds.seg_min_length)
    print('2a condicao')
    print((p_ids[:, 1] - p_ids[:, 0]) >= thersholds.min_point_seg)
    print('and')
    print(
        np.transpose(seg_len >= thersholds.seg_min_length) & ((p_ids[:, 1] - p_ids[:, 0]) >= thersholds.min_point_seg))

    print('correct_segs_ids')
    print(correct_segs_ids)
    '''
    p_ids = p_ids[correct_segs_ids[:, 0], :]

    alpha = np.asmatrix(alpha)
    alpha = alpha[correct_segs_ids[:, 0], 0]

    r = r[correct_segs_ids[:, 0], 0]

    seg_i_f = seg_i_f[correct_segs_ids[:, 0], :]
    seg_len = np.transpose(seg_len)
    seg_len = seg_len[correct_segs_ids[:, 0], 0]

    z = np.transpose(np.hstack((alpha, r)))
    R_seg = np.zeros((2, 2, alpha.shape[0]))

    for c in range(0,alpha.shape[0]):
        for j in range(0,2):
            if j == 0:
                R_seg[j,j,c] = 0.2 ** 2
            if j == 1:
                R_seg[j,j,c] = np.deg2rad(6) ** 2

    return z, R_seg, seg_i_f


def normalize_line(alpha, r):
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


# EKF FUNCTIONS
# predict
def predict(x_est, E_est, u):
    # _predict step
    # x_est é o anterior e vai ser atualizado no final
    G_x = np.array([[1.0, 0, -u[1, 0] * math.sin(x_est[2, 0] + u[0, 0])],
                    [0, 1.0, u[1, 0] * math.cos(x_est[2, 0] + u[0, 0])],
                    [0, 0, 1.0]])

    b = b = np.array([[u[1, 0] * math.cos(x_est[2, 0] + u[0, 0])],
                  [u[1, 0] * math.sin(x_est[2, 0] + u[0, 0])],
                  [u[0, 0] + u[2, 0]]])

    E_est = G_x @ E_est @ G_x.T + Q_est
    x_est = x_est + b

    return x_est, E_est


# update
def update_mat(x, m):
    h = np.array([[m[0] - x[2,0]], [m[1] - (x[0,0] * math.cos(m[0]) + x[1,0] * math.sin(m[0]))]])
    Hxmat = np.array([[0, 0, -1], [-math.cos(m[0]), -math.sin(m[0]), 0]])

    [h[0], h[1], isdistneg] = normalize_line(h[0], h[1])

    if isdistneg:
        Hxmat[1, :] = -Hxmat[1, :]

    return h, Hxmat


def matching(x, P, z, R_seg, M, g):
    #z: Linhas observadas
    n_measurs = z.shape[1]
    n_map = M.shape[1]

    d = np.zeros((n_measurs, n_map))
    v = np.zeros((2, n_measurs * n_map))
    H = np.zeros((2, 3, n_measurs * n_map ))

    v = np.asmatrix(v)


    for aux_nme in range(0, n_measurs):
        for aux_nmap in range(0, n_map):
            z_predict, H[:, :, aux_nmap + (aux_nme) * n_map] = update_mat(x, M[:, aux_nmap])
            v[:, aux_nmap + (aux_nme) * n_map] = z[:, aux_nme] - z_predict
            W = H[:, :, aux_nmap + (aux_nme) * n_map] @ P @ np.transpose(H[:, :, aux_nmap + (aux_nme) * n_map]) + R_seg[:, :, aux_nme]
            #Distancia Mahalanahobis
            d[aux_nme, aux_nmap] = np.transpose(v[:, aux_nmap + (aux_nme) * n_map]) * np.linalg.inv(W) * v[:, aux_nmap + (aux_nme) * n_map]


    min_mahal, map_id = (np.transpose(d)).min(0), (np.transpose(d)).argmin(0)
    measure_id = np.argwhere(min_mahal < g**2)
    map_id = map_id[np.transpose(measure_id)]
    seletor = (map_id + (np.transpose(measure_id))* n_map)
    seletorl =[]

    for f in range(0,seletor.shape[1]):
        seletorl.append(seletor.item(f))

    v = v[:, seletorl]
    H = H[:, :, seletorl]

    measure_id = np.transpose(measure_id)
    measure_idl = []
    for b in range(0, measure_id.shape[1]):
        measure_idl.append(measure_id.item(b))
    if seletorl == []:
        R_seg = R_seg[:, :, seletorl]
    else:
        R_seg = R_seg[:, :, measure_idl]

    return v, H, R_seg


def update(x_est, E_est, z, R_seg, mapa, g):

    if z.shape[1]==0:
        x_up = x_est
        E_up = E_est

        return x_up, E_up

    v, H, R_seg = matching(x_est, E_est, z, R_seg, mapa, g)

    #mudar formato de v, H e R para usar nas equacoes
    y = np.reshape(v, (v.shape[0]*v.shape[1],1), 'F')

    H = np.transpose(H, [0, 2, 1])
    Hreshape = np.reshape(H, [-1, 3], 'F')

    if R_seg.shape[2] == 0:
        R_seg1 = []
    else:
        R_seg1 = R_seg[:, :, 0]
        for bruh in range(1, R_seg.shape[2]):
            R_seg1 = scipy.linalg.block_diag(R_seg1, R_seg[:, :, bruh])

    S = Hreshape @ E_est @ np.transpose(Hreshape) + R_seg1
    K = E_est @ np.transpose(Hreshape) @ (np.linalg.inv(S))

    E_up = E_est - K @ S @ np.transpose(K)
    x_up = x_est + K @ y

    return x_up, E_up


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
    global odom, calledodom #, pos, ori
    #pos = [msg.pose.pose.position.x, msg.pose.pose.position.y]
    quat = [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]
    roll, pitch, yaw = euler_from_quaternion(quat)
    odom = [msg.pose.pose.position.x, msg.pose.pose.position.y, yaw]
    calledodom = 1


def u_from_odom(pos_at, pos_prev):
    #drot = pos_at[2] - pos_prev[2]
    drot1 = math.atan2(pos_at[1] - pos_prev[1],pos_at[0] - pos_prev[0]) - pos_prev[2]
    dtrans = math.sqrt((pos_prev[0] - pos_at[0])**2 + (pos_prev[1] - pos_at[1])**2)
    drot2 = pos_at[2] - pos_prev[2] - drot1

    #u = np.array([[dtrans], [drot]])
    u = np.array([[drot1], [dtrans], [drot2]])
    return u


if __name__ == '__main__':
    rospy.init_node('ekf_location')

    # subscribe to odom and laser
    call1 = rospy.Subscriber('/scan', LaserScan, callback)
    call2 = rospy.Subscriber('/pose', Odometry, callback2)
    
    broadcaster = tf2_ros.StaticTransformBroadcaster()
    static_transformStamped = geometry_msgs.msg.TransformStamped()



    # initializations
    i = 0
    co = 0
    pos_prev = odom
    pos_at = [0, 0, 0]
    E_est = np.eye(3)
    #x_est = np.array([[0], [0.00], [0]])
    x_est = np.array([[-0.026], [-0.001], [0]])
    roll, pitch, yaw = euler_from_quaternion([-0.0, 0.0, 0.8090169943749475, -0.587785252292473])
    x_est[2] = yaw
    #x_est[2] = pi
    g = 0.8

    rate = rospy.Rate(5)
    while not rospy.is_shutdown():
        #print('-----------------------------------------------------------x-----------------------')
        #print(x_est)
        """
        print('Range length = ', range_ar.shape)
        print(pos)
        range_ar[2] = 9
        print('---------------------------------------------------------')
        """
        thresholds = Thresholds()
        #extact lines
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
        z, R, segends = split_merge(thetas, dist, thresholds)
        

        # EKF - predict
        if calledodom == 1 and co == 0:
            co = 1
            pos_prev = odom
        pos_at = odom
        u = u_from_odom(pos_at, pos_prev)
        #print('pos_prev = ', end='')
        #print(pos_prev)
        #print('pos_at = ', end='')
        #print(pos_at)
        pos_prev = pos_at
        x_est, E_est = predict(x_est, E_est, u)

        # EKF - update
        x_est, E_est = update(x_est, E_est, z, R, mapa, g)
        x_est = np.asarray(x_est)

        static_transformStamped.header.stamp = rospy.Time.now()
        static_transformStamped.header.frame_id = "map"
        static_transformStamped.child_frame_id = "laser" 
        
        static_transformStamped.transform.translation.x = x_est.item(0)
        static_transformStamped.transform.translation.y = x_est.item(1)
        static_transformStamped.transform.translation.z = 0

        quat = tf.transformations.quaternion_from_euler(0,0,x_est.item(2))
        static_transformStamped.transform.rotation.x = quat[0]
        static_transformStamped.transform.rotation.y = quat[1]
        static_transformStamped.transform.rotation.z = quat[2]
        static_transformStamped.transform.rotation.w = quat[3]
        broadcaster.sendTransform(static_transformStamped)
        
        print(x_est)

        
        i += 1


        rate.sleep()