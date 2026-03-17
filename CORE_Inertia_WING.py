import numpy as np

def calculate_centroid(p1, p2, p3, p4):
    """
    计算三维空间直角梯形的重心。
    输入参数为四个顶点的坐标：p1, p2, p3, p4
    其中 p1-p2 是直角边，p3-p4 是斜边，p1-p4 是顶边，p2-p3 是底边。
    """
    # 将输入的顶点转换为NumPy数组
    p1, p2, p3, p4 = np.array(p1), np.array(p2), np.array(p3), np.array(p4)

    # 计算底边和顶边的中点
    m = (p2 + p3) / 2  # 底边中点
    n = (p1 + p4) / 2  # 顶边中点

    # 计算底边和顶边的长度
    a = np.linalg.norm(p4 - p1)  # 顶边长度
    b = np.linalg.norm(p3 - p2)  # 底边长度

    # 计算高h
    # 使用向量投影计算高，投影点P1到底边P2P3的距离
    base_vector = p3 - p2  # 底边向量
    height_vector = p1 - p2  # p1到p2的向量
    projection_length = np.dot(height_vector, base_vector) / np.linalg.norm(base_vector)
    perpendicular_component = height_vector - (projection_length / np.linalg.norm(base_vector) ** 2) * base_vector
    h = np.linalg.norm(perpendicular_component)

    # 计算重心的相对位置
    centroid_rel = 1 / 3 * ((2 * b + a) / (a + b))

    # 计算实际的重心位置
    centroid = n + centroid_rel * (m - n)

    return centroid

def GET_Z_POS(Y, Y3, Z3, Y4, Z4):
    # 计算中间任意一点弦长
    if (Y4 - Y3)==0:
        Z = Z3
    else:
        Z = Z3 + (Z4 - Z3)/(Y4 - Y3) * (Y - Y3)
    return Z

def Inertia_WING(t = 0.05/1000,
                 S_total = 0.085,
                 D_dc = 0.005,
                 C_root = 0.0333,
                 c_tip = 0.01998,
                 dens = 1000,
                 N_S = 10,
                 N_C = 10,
                 N_T = 2
                 ):
    """
    翼的惯量计算

    :param t: 单位：m
    :param S_total: 单位：m
    :param D_dc: 单位：m
    :param C_root: 单位：m
    :param c_tip: 单位：m
    :param dens: 单位：kg/m³
    :return:
    """

    # %% 平面关系离散
    POINTS = np.zeros((N_T,N_S,N_C,3))
    D_t_seq = np.linspace(-t/2, t/2, num=N_T)
    D_s_seq = np.linspace(D_dc, S_total, num=N_S)

    # %% 循环生成
    for j in range(N_S):
        LEN_z = GET_Z_POS(D_s_seq[j], D_dc, -C_root, S_total, -c_tip)
        TEMP_SEQ_Z = np.linspace(LEN_z, 0, num=N_C)
        for k in range(N_C):
            for i in range(N_T):
                POINTS[i, j, k, 0] = D_t_seq[i]
                POINTS[i, j, k, 1] = D_s_seq[j]
                TEMP_Z = TEMP_SEQ_Z[k]
                POINTS[i,j,k,2] = TEMP_Z


    # %% 真实体积计算
    C_root = C_root
    C_tip = c_tip
    S = S_total - D_dc
    t = t

    V_r = ((C_root + C_tip)/2) * S * t

    # %% 重心位置生成
    N_pos_S = N_S - 1  # 展向离散
    N_pos_C = N_C - 1  # 弦向离散
    N_pos_T = N_T - 1  # 厚度方向离散数量

    CG_POINTS = np.zeros((N_pos_T,N_pos_S,N_pos_C,3))
    PV_POINTS = np.zeros((N_pos_T,N_pos_S,N_pos_C))

    for i in range(N_pos_T):
        for j in range(N_pos_S):
            for k in range(N_pos_C):
                P1 = POINTS[i,   j+1, k+1,:]
                P2 = POINTS[i,   j,   k+1,:] # 相对P2加在Z上
                P3 = POINTS[i,   j,   k,  :] # 基准点
                P4 = POINTS[i,   j+1, k,  :] # 相对P2加在Y上

                P5 = POINTS[i+1, j+1, k+1,:]
                P6 = POINTS[i+1, j,   k+1,:]
                P7 = POINTS[i+1, j,   k,  :]  # 相对P2加在X上
                P8 = POINTS[i+1, j+1, k,  :]


                DV_CG = calculate_centroid(P1, P2, P3, P4)


                X_POS = (P1 + P5)/2

                DV_CG[0]= X_POS[0] # 调整Z轴坐标


                CG_POINTS[i,j,k,:] = DV_CG
                mini_C_root = np.linalg.norm(P2 - P3)
                mini_C_tip = np.linalg.norm(P1 - P4)
                mini_S = np.linalg.norm(P1 - P2)
                mini_t = np.linalg.norm(P7 - P3)

                PV_POINTS[i,j,k] = (((mini_C_root + mini_C_tip)/2) * mini_S * mini_t)/V_r


    # %% 计算较准
    D_V = np.sum(PV_POINTS)
    # print("加和验证",D_V*100)

    # %% 质量分布
    MASS = V_r * dens

    # %% 惯量集成
    SUM_J_W_XX = 0.0
    SUM_J_W_YY = 0.0
    SUM_J_W_ZZ = 0.0
    SUM_J_W_XY = 0.0
    SUM_J_W_XZ = 0.0
    SUM_J_W_YZ = 0.0

    for i in range(N_pos_T):
        for j in range(N_pos_S):
            for k in range(N_pos_C):
                TEMP_X = CG_POINTS[i,j,k,0]
                TEMP_Y = CG_POINTS[i,j,k,1]
                TEMP_Z = CG_POINTS[i,j,k,2]

                d_MASS = MASS * PV_POINTS[i,j,k]

                SUM_J_W_XX = SUM_J_W_XX + d_MASS * (TEMP_Y**2 + TEMP_Z**2)
                SUM_J_W_YY = SUM_J_W_YY + d_MASS * (TEMP_X**2 + TEMP_Z**2)
                SUM_J_W_ZZ = SUM_J_W_ZZ + d_MASS * (TEMP_X**2 + TEMP_Y**2)
                SUM_J_W_YZ = SUM_J_W_YZ - d_MASS * TEMP_Y * TEMP_Z

    return SUM_J_W_XX,SUM_J_W_YY,SUM_J_W_ZZ,SUM_J_W_XY,SUM_J_W_XZ,SUM_J_W_YZ


# SUM_J_W_XX, SUM_J_W_YY, SUM_J_W_ZZ, SUM_J_W_XY, SUM_J_W_XZ, SUM_J_W_YZ = Inertia_WING()
# # %%
# print("求解完成")
# J_W_XX = 0.00000026743  # 经过校准
# J_W_YY = 0.00000002678  # 经过校准
# J_W_ZZ = 0.00000024065  # 10e-7  # 参见resonance Principle for the design of flapping wing micro mcro air vehicle
# J_W_XY = 0.0
# J_W_XZ = 0.0
# J_W_YZ = 0.00000005574  # 这个让扭转和扑动耦合
#
# print("SUM_J_W_XX",format(SUM_J_W_XX, ".15f"), (1-SUM_J_W_XX/J_W_XX)*100)
# print("SUM_J_W_YY",format(SUM_J_W_YY, ".15f"), (1-SUM_J_W_YY/J_W_YY)*100)
# print("SUM_J_W_ZZ",format(SUM_J_W_ZZ, ".15f"), (1-SUM_J_W_ZZ/J_W_ZZ)*100)
# print("SUM_J_W_YZ",format(SUM_J_W_YZ, ".15f"), (1-SUM_J_W_YZ/J_W_YZ)*100)