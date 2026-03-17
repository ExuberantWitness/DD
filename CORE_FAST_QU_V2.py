import time
import numpy as np
from numba import njit


@njit
def quaternion_multiply(q1, q2):
    # 标准四元数乘法
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2,
                     w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2])


@njit
def create_rotation_quaternion(thetax, thetay, thetaz):
    # 计算基本四元数
    cx, sx = np.cos(-thetax / 2), np.sin(-thetax / 2)
    cy, sy = np.cos(-thetay / 2), np.sin(-thetay / 2)
    cz, sz = np.cos(-thetaz / 2), np.sin(-thetaz / 2)

    # 计算旋转
    q = quaternion_multiply(np.array([cz, 0, 0, sz]), np.array([cy, 0, sy, 0]))
    q = quaternion_multiply(q, np.array([cx, sx, 0, 0]))
    return q


@njit
def rotate_vector(q, vector):
    # 输入的 vector 是一个一维数组，表示单个三维向量
    v = np.zeros(4)
    v[1:] = vector  # 将三维向量拓展到四元数，首位为0
    v_rot = quaternion_multiply(quaternion_multiply(q, v), np.array([q[0], -q[1], -q[2], -q[3]]))
    rotated_vector = v_rot[1:]  # 提取旋转后的三维部分
    return rotated_vector

@njit
def quaternion_shift_numba(V_outer_I, thetax, thetay, thetaz):
    q_x = create_rotation_quaternion(thetax, 0, 0)
    q_y = create_rotation_quaternion(0, thetay, 0)
    q_z = create_rotation_quaternion(0, 0, thetaz)




    q_total = quaternion_multiply(quaternion_multiply(q_x, q_y), q_z)
    return rotate_vector(q_total, V_outer_I)  # 调用处理单个向量的函数


@njit
def FAST_TILE_4(vector,LEN):
    result = np.ones((LEN,4))
    result[:, 0] = result[:, 0] * vector[0]
    result[:, 1] = result[:, 1] * vector[1]
    result[:, 2] = result[:, 2] * vector[2]
    result[:, 3] = result[:, 3] * vector[3]
    return result

@njit
def A_quaternion_multiply(q1, q2, LEN):
    # 标准四元数乘法
    w1 = q1[:,0]
    x1 = q1[:,1]
    y1 = q1[:,2]
    z1 = q1[:,3]

    w2 = q2[:,0]
    x2 = q2[:,1]
    y2 = q2[:,2]
    z2 = q2[:,3]

    RESULT = np.zeros((LEN,4))

    RESULT[:, 0] = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    RESULT[:, 1] = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    RESULT[:, 2] = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    RESULT[:, 3] = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2

    return RESULT


@njit
def A_create_rotation_quaternion(thetax, thetay, thetaz, LEN):
    # 计算基本四元数
    cx, sx = np.cos(-thetax / 2), np.sin(-thetax / 2)
    cy, sy = np.cos(-thetay / 2), np.sin(-thetay / 2)
    cz, sz = np.cos(-thetaz / 2), np.sin(-thetaz / 2)


    Q_A_1 = np.zeros((LEN, 4))
    Q_A_1[:,0] = cz
    Q_A_1[:,3] = sz

    Q_A_2 = np.zeros((LEN, 4))
    Q_A_2[:,0] = cy
    Q_A_2[:,2] = sy


    Q_A_3 = np.zeros((LEN, 4))
    Q_A_3[:,0] = cx
    Q_A_3[:,1] = sx

    q = A_quaternion_multiply(A_quaternion_multiply(Q_A_1, Q_A_2, LEN), Q_A_3, LEN = LEN)
    return q


@njit
def A_rotate_vector(q, vector, LEN):
    # 输入的 vector 是一个一维数组，表示单个三维向量
    v = np.zeros((LEN,4))
    v[:,1] = vector[:,0]  # 将三维向量拓展到四元数，首位为0
    v[:,2] = vector[:,1]  # 将三维向量拓展到四元数，首位为0
    v[:,3] = vector[:,2]  # 将三维向量拓展到四元数，首位为0



    new_q = np.zeros((LEN,4))
    new_q[:, 0] = + q[:, 0]
    new_q[:, 1] = - q[:, 1]
    new_q[:, 2] = - q[:, 2]
    new_q[:, 3] = - q[:, 3]



    v_rot = A_quaternion_multiply(A_quaternion_multiply(q, v,LEN), new_q,LEN)

    RESULT = np.zeros((LEN, 3))
    RESULT[:, 0] = v_rot[:, 1]
    RESULT[:, 1] = v_rot[:, 2]
    RESULT[:, 2] = v_rot[:, 3]

    return RESULT

@njit
def A_quaternion_shift_numba(V_outer_I, thetax, thetay, thetaz, LEN):
    ZEROS = np.zeros(LEN)  # 其实不加也行，
    q_x = A_create_rotation_quaternion(thetax, ZEROS, ZEROS, LEN = LEN)
    q_y = A_create_rotation_quaternion(ZEROS, thetay, ZEROS, LEN = LEN)
    q_z = A_create_rotation_quaternion(ZEROS, ZEROS, thetaz, LEN = LEN)
    q_total = A_quaternion_multiply(A_quaternion_multiply(q_x, q_y,LEN = LEN), q_z,LEN = LEN)
    return A_rotate_vector(q_total, V_outer_I, LEN = LEN)  # 调用处理单个向量的函数
