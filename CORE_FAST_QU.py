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