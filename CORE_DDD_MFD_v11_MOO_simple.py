import numpy as np
import sympy as sp
from numba import njit
from sympy.physics.mechanics import dynamicsymbols



LEN_LEG_general = 25 / 1000
SCACLE_GEM = 0.1
tf = 0.03 / 1000  # 不要改：TPU膜的厚度
E_tpu = 42 * 1000000  # 不要改：TPU的弹性模量
L_install = 20 / 1000  # 不要改：几何定义，不要改，要改就改直接整体缩放
L_inforce = 10 / 1000  # 不要改：几何定义，不要改，要改就改直接整体缩放
# 总集计算
K_stand = tf * L_inforce * E_tpu / np.radians(65) * L_install * 0.5 * SCACLE_GEM


C_ROTA = 0.0000013
# 汪亮 是是10-8  不要超过2，不要过大，不然回卡在一边
# 不要过大0.0000007  ，不然60hz直接就别卡住俄

def MOTOR_MODEL(kp,ki,kd,error,integral,derivative,SCALE):
    output = kp * error + ki * integral + kd * derivative
    return output*SCALE

def R_xyz(theta_x=0.0, theta_y=0.0, theta_z=0.0):
    """
    新的=旋转矩阵x旧的

    这是角度是正的，不然角度是负的

    :param theta_x: 当地x轴
    :param theta_y: 当地y轴
    :param theta_z: 当地z轴
    :return:
    """
    Rx = sp.Matrix([[1.0, 0.0, 0.0], [0.0, sp.cos(theta_x), -sp.sin(theta_x)], [0.0, sp.sin(theta_x), sp.cos(theta_x)]])
    Ry = sp.Matrix([[sp.cos(theta_y), 0.0, sp.sin(theta_y)], [0.0, 1.0, 0.0], [-sp.sin(theta_y), 0.0, sp.cos(theta_y)]])
    Rz = sp.Matrix([[sp.cos(theta_z), -sp.sin(theta_z), 0.0], [sp.sin(theta_z), sp.cos(theta_z), 0.0], [0.0, 0.0, 1.0]])
    return Rz.subs(theta_z, theta_z) * Ry.subs(theta_y, theta_y) * Rx.subs(theta_x, theta_x)


def BASE_Cramer_SP(eqns,variables):

    # 雅可比矩阵计算
    Jacobian = sp.Matrix([[eq.diff(var) for var in variables] for eq in eqns])

    # 输入关系调整
    FREE_ARRY = sp.Matrix([*variables])
    EQ_ARRY = sp.Matrix([*eqns])
    constants = -(EQ_ARRY - Jacobian*FREE_ARRY)

    # 克拉默法求解线性方程组
    D = sp.det(Jacobian)
    solutions = [0.0] * len(eqns)
    if D != 0:
        ID = 0
        for var in variables:
            print("参数",var,"ID",ID)
            # 用常数向量替换雅可比矩阵中相应的列，然后计算行列式
            temp_matrix = Jacobian.copy()
            temp_matrix[:, variables.index(var)] = constants
            solutions[ID] = sp.det(temp_matrix) / D
            ID = ID + 1
    else:
        print("无法使用克拉默法求解，因为系数矩阵的行列式为零。")

    return solutions



#
# def DDD_SYSTEM(CENTER_SP_1,CENTER_SP_2,CENTER_SP_3,CENTER_SP_4,J_M_ZZ = 0.000000497, J_W_XX = 0.00000027, J_W_YY = 0.00000003, J_W_ZZ = 0.00000024, J_W_XY = 0.0,J_W_XZ = 0.0, J_W_YZ = 0.00000006):
def DDD_SYSTEM(CENTER_SP_1, CENTER_SP_2, CENTER_SP_3, CENTER_SP_4):

    # %% 基本定义
    # %% 基本定义
    t = sp.symbols('t')
    eps = np.finfo(np.float64).eps

    # 这里定义的是通用的三周旋转矩阵
    # %% 系统广义自由度定义
    phi_1 = dynamicsymbols('phi_1')  # 扑动角
    phi_1_d = sp.diff(phi_1, t)
    phi_1_dd = sp.diff(phi_1_d, t)

    phi_2 = dynamicsymbols('phi_2')  # 扑动角
    phi_2_d = sp.diff(phi_2, t)
    phi_2_dd = sp.diff(phi_2_d, t)

    phi_3 = dynamicsymbols('phi_3')  # 扑动角
    phi_3_d = sp.diff(phi_3, t)
    phi_3_dd = sp.diff(phi_3_d, t)

    phi_4 = dynamicsymbols('phi_4')  # 扑动角
    phi_4_d = sp.diff(phi_4, t)
    phi_4_dd = sp.diff(phi_4_d, t)

    theta_1 = dynamicsymbols('theta_1')  # 扭转角
    theta_1_d = sp.diff(theta_1, t)
    theta_1_dd = sp.diff(theta_1_d, t)

    theta_2 = dynamicsymbols('theta_2')  # 扭转角
    theta_2_d = sp.diff(theta_2, t)
    theta_2_dd = sp.diff(theta_2_d, t)

    theta_3 = dynamicsymbols('theta_3')  # 扭转角
    theta_3_d = sp.diff(theta_3, t)
    theta_3_dd = sp.diff(theta_3_d, t)

    theta_4 = dynamicsymbols('theta_4')  # 扭转角
    theta_4_d = sp.diff(theta_4, t)
    theta_4_dd = sp.diff(theta_4_d, t)

    # %% 基本惯量参数
    J_M_ZZ = sp.symbols('J_M_ZZ')
    J_W_XX = sp.symbols('J_W_XX')
    J_W_YY = sp.symbols('J_W_YY')
    J_W_ZZ = sp.symbols('J_W_ZZ')
    J_W_XY = sp.symbols('J_W_XY')
    J_W_XZ = sp.symbols('J_W_XZ')
    J_W_YZ = sp.symbols('J_W_YZ')


    # %% 外部辅助状态量接口
    F_X_B1 = sp.symbols('F_X_B1')
    F_Y_B1 = sp.symbols('F_Y_B1')
    F_Z_B1 = sp.symbols('F_Z_B1')
    T_X_W1 = sp.symbols('T_X_W1')
    T_Y_W1 = sp.symbols('T_Y_W1')
    T_Z_W1 = sp.symbols('T_Z_W1')

    F_X_B2 = sp.symbols('F_X_B2')
    F_Y_B2 = sp.symbols('F_Y_B2')
    F_Z_B2 = sp.symbols('F_Z_B2')
    T_X_W2 = sp.symbols('T_X_W2')
    T_Y_W2 = sp.symbols('T_Y_W2')
    T_Z_W2 = sp.symbols('T_Z_W2')

    F_X_B3 = sp.symbols('F_X_B3')
    F_Y_B3 = sp.symbols('F_Y_B3')
    F_Z_B3 = sp.symbols('F_Z_B3')
    T_X_W3 = sp.symbols('T_X_W3')
    T_Y_W3 = sp.symbols('T_Y_W3')
    T_Z_W3 = sp.symbols('T_Z_W3')

    F_X_B4 = sp.symbols('F_X_B4')
    F_Y_B4 = sp.symbols('F_Y_B4')
    F_Z_B4 = sp.symbols('F_Z_B4')
    T_X_W4 = sp.symbols('T_X_W4')
    T_Y_W4 = sp.symbols('T_Y_W4')
    T_Z_W4 = sp.symbols('T_Z_W4')



    # %% 约束弹簧力
    Stiff_TE_W1 = sp.symbols('Stiff_TE_W1')
    Stiff_TE_W2 = sp.symbols('Stiff_TE_W2')
    Stiff_TE_W3 = sp.symbols('Stiff_TE_W3')
    Stiff_TE_W4 = sp.symbols('Stiff_TE_W4')

    VALUE_Te_YAW_1 = sp.symbols('VALUE_Te_YAW_1')
    VALUE_Te_YAW_2 = sp.symbols('VALUE_Te_YAW_2')
    VALUE_Te_YAW_3 = sp.symbols('VALUE_Te_YAW_3')
    VALUE_Te_YAW_4 = sp.symbols('VALUE_Te_YAW_4')

    # %% 控制输入等动态量
    K_A1 = sp.symbols('K_A1')
    K_A2 = sp.symbols('K_A2')
    K_A3 = sp.symbols('K_A3')
    K_A4 = sp.symbols('K_A4')

    CONTROL_SAFW = sp.symbols('CONTROL_SAFW')  # 半主动攻角的舵机旋转角（绕Z轴右手方向为正）
    CONTROL_SAHW = sp.symbols('CONTROL_SAHW')  # 半主动攻角的舵机旋转角（绕Z轴右手方向为正）

    TORQUE_M_1 = sp.symbols('TORQUE_M_1') # 这个是直接面对扭转角的，经过传动系统放大后的总力矩
    TORQUE_M_2 = sp.symbols('TORQUE_M_2')
    TORQUE_M_3 = sp.symbols('TORQUE_M_3')
    TORQUE_M_4 = sp.symbols('TORQUE_M_4')

    # %% 参数预先设定值

    DX_B_FW = 50 / 1000  # 驱动器安装位置Y方向(相对机体坐标系的局部位置)
    DY_B_FW = 25 / 1000  # 驱动器安装位置Y方向(相对机体坐标系的局部位置,+/-)
    DZ_B_FW = 30 / 1000  # 驱动器安装位置Y方向(相对机体坐标系的局部位置)

    DX_B_HW = -50 / 1000  # 驱动器安装位置Y方向(相对机体坐标系的局部位置)
    DY_B_HW = 25 / 1000  # 驱动器安装位置Y方向(相对机体坐标系的局部位置,+/-)
    DZ_B_HW = - 30 / 1000  # 驱动器安装位置Y方向(相对机体坐标系的局部位置)

    DZ_R_B_FWJ = 40 / 1000  # 攻角控制控制器的整体偏移
    DZ_R_B_HWJ = 40 / 1000  # 攻角控制控制器的整体偏移

    L_B_FWARM = 25 / 1000  # 攻角控制半摇臂长
    L_B_HWARM = 25 / 1000  # 攻角控制半摇臂长

    DY_W_WING_INSTALL = 5 / 1000  # 翼相对轴线的偏移位置

    VALUE_GR = 1


    # %% 惯量特性
    m_B = 1.00958400
    J_B_XX = 0.00064142
    J_B_YY = 0.00250949
    J_B_ZZ = 0.00201950
    J_B_XY = 0.0
    J_B_XZ = 0.0
    J_B_YZ = 0.0

    # 电机惯量
    # 这里引用的22XL的转矩数据+杆+弹簧
    # m_A_1234 = 0.03976000077198065 + 0.27037332/1000 + 2/1000  # 飞行器
    m_A_1234 = 0.162  # 实验台
    J_M_XX = 0.00002145383374988123
    J_M_YY = 0.00002145383374988123
    J_M_ZZ = J_M_ZZ  # 取值正确
    J_M_XY = 0.0
    J_M_XZ = 0.0
    J_M_YZ = 0.0

    # 翼的质量
    m_W_1234 = 0.00022624  # 200mg左右

    # m_W_1234 = sp.symbols('m_W_1234')
    # print("强行给了函数名称")

    # 翼中心位置
    X_R_W1234_WCOM = 0.0
    Y_R_W1234_WCOM = 0.03666667
    Z_R_W1234_WCOM = - 0.01359750

    # 翼惯性张量: ( 千克 * 平方米 )相对旋转铰点
    # J_W_XX = 0.00000027  # 经过校准
    # J_W_YY = 0.00000003  # 经过校准
    # J_W_ZZ = 0.00000024  # 10e-7  # 参见resonance Principle for the design of flapping wing micro mcro air vehicle
    # J_W_XY = 0.0
    # J_W_XZ = 0.0
    # J_W_YZ = 0.00000006  # 这个让扭转和扑动耦合
    print("反向耦合")
    g = 9.8

    # 翼的扭转刚度（主要防止计算发散）
    # N_K_W1234 = 0.001 / (np.pi / 2)

    # # %% 串列翼干扰
    # C_ROTA = 0.0000013
    # # 汪亮 是是10-8  不要超过2，不要过大，不然回卡在一边
    # # 不要过大0.0000007  ，不然60hz直接就别卡住俄


    # %% 鬼畜坐标系旋转
    phi_1_R = phi_1 + CENTER_SP_1  # phi_1：变化增量，phi_1_R：坐标系下的绝对值
    phi_2_R = phi_2 + CENTER_SP_2
    phi_3_R = phi_3 + CENTER_SP_3
    phi_4_R = phi_4 + CENTER_SP_4





    # %% 各种初始值给定

    # 驱动器初始惯量
    Z_I_BA1_COM_init = DZ_B_FW
    Z_I_BA2_COM_init = DZ_B_FW
    Z_I_BA3_COM_init = DZ_B_HW
    Z_I_BA4_COM_init = DZ_B_HW

    # 翼初始惯量
    z_I_W1_COM_init = DZ_B_FW + Z_R_W1234_WCOM
    z_I_W2_COM_init = DZ_B_FW + Z_R_W1234_WCOM
    z_I_W3_COM_init = DZ_B_HW + Z_R_W1234_WCOM
    z_I_W4_COM_init = DZ_B_HW + Z_R_W1234_WCOM


    # %% 系统关键矢径

    # P3：关键点：驱动器矢径点（其是固结在机架上的点，没有）
    R_B_BA1 = sp.Matrix([DX_B_FW, DY_B_FW, DZ_B_FW])
    POS_B_BA1 = R_B_BA1

    R_B_BA2 = sp.Matrix([DX_B_FW, -DY_B_FW, DZ_B_FW])
    POS_B_BA2 = R_B_BA2

    R_B_BA3 = sp.Matrix([DX_B_HW, -DY_B_HW, DZ_B_HW])
    POS_B_BA3 = R_B_BA3

    R_B_BA4 = sp.Matrix([DX_B_HW, DY_B_HW, DZ_B_HW])
    POS_B_BA4 = R_B_BA4

    # %% 驱动器相关参数

    # P5：总能动能
    KET_A1 = 0.5 * phi_1_d * phi_1_d * J_M_ZZ  # 这里这里直接采用局部坐标系结果
    KET_A2 = 0.5 * phi_2_d * phi_2_d * J_M_ZZ
    KET_A3 = 0.5 * phi_3_d * phi_3_d * J_M_ZZ
    KET_A4 = 0.5 * phi_4_d * phi_4_d * J_M_ZZ
    QE_K_A1234 = KET_A1 + KET_A2 + KET_A3 + KET_A4

    # 驱动重力势能
    U_GR_A1 = (POS_B_BA1[2] - Z_I_BA1_COM_init) * g * m_A_1234  # ok
    U_GR_A2 = (POS_B_BA2[2] - Z_I_BA2_COM_init) * g * m_A_1234  # ok
    U_GR_A3 = (POS_B_BA3[2] - Z_I_BA3_COM_init) * g * m_A_1234  # ok
    U_GR_A4 = (POS_B_BA4[2] - Z_I_BA4_COM_init) * g * m_A_1234  # ok

    # 驱动器弹性势能作用(这里不能直接使用初始位置，因为起始位置和弹簧中立位置不是一个)
    U_SP_A1 = 0.5 * K_A1 * (phi_1 - CENTER_SP_1) ** 2  # ok 这里必须考虑中心位置
    U_SP_A2 = 0.5 * K_A2 * (phi_2 - CENTER_SP_2) ** 2  # ok
    U_SP_A3 = 0.5 * K_A3 * (phi_3 - CENTER_SP_3) ** 2  # ok
    U_SP_A4 = 0.5 * K_A4 * (phi_4 - CENTER_SP_4) ** 2  # ok



    # QE_U_A1234 = U_GR_A1 + U_GR_A2 + U_GR_A3 + U_GR_A4 + U_SP_A1 + U_SP_A2 + U_SP_A3 + U_SP_A4
    QE_U_A1234 = U_SP_A1 + U_SP_A2 + U_SP_A3 + U_SP_A4
    # 约束：驱动器没有额外约束
    QE_C_A1234 = 0

    TW_M1 = TORQUE_M_1 * phi_1  # ok
    TW_M2 = TORQUE_M_2 * phi_2  # ok
    TW_M3 = TORQUE_M_3 * phi_3  # ok
    TW_M4 = TORQUE_M_4 * phi_4  # ok
    QE_W_A1234 = TW_M1 + TW_M2 + TW_M3 + TW_M4  # OK











    # %% 翼部分
    # A_B4W1 = R_xyz(theta_y=theta_1, theta_z=phi_1_R) # 气动力计算不需要这个，因为器利用numpy进行快速结算
    # A_B4W2 = R_xyz(theta_y=theta_2, theta_z=phi_2_R)
    # A_B4W3 = R_xyz(theta_y=theta_3, theta_z=phi_3_R)
    # A_B4W4 = R_xyz(theta_y=theta_4, theta_z=phi_4_R)
    # #
    # # # P3：关键点：翼有几何参考点(一定时,假设翼初始指向左,然后利用phi_1来调整后续参数，主要是用来计算翼的动能)
    # # R_W1234_MREF = sp.Matrix([0.0, DY_W_WING_INSTALL, 0.0])
    # # POS_B_W1_MREF = POS_B_BA1 + A_B4W1 * R_W1234_MREF  # print(sp.latex(A_B2W1 * R_W1234_MREF))
    # # POS_B_W2_MREF = POS_B_BA2 + A_B4W2 * R_W1234_MREF  # print(sp.latex(A_B2W2 * R_W1234_MREF))
    # # POS_B_W3_MREF = POS_B_BA3 + A_B4W3 * R_W1234_MREF  # print(sp.latex(A_B2W3 * R_W1234_MREF))
    # # POS_B_W4_MREF = POS_B_BA4 + A_B4W4 * R_W1234_MREF  # print(sp.latex(A_B2W4 * R_W1234_MREF))
    # #
    # # # # P3：关键点：翼的重心
    # # R_W1234_WCOM = sp.Matrix([X_R_W1234_WCOM,
    # #                           Y_R_W1234_WCOM,
    # #                           Z_R_W1234_WCOM])
    # # POS_B_W1_COM = POS_B_BA1 + A_B4W1 * R_W1234_WCOM
    # # POS_B_W2_COM = POS_B_BA2 + A_B4W2 * R_W1234_WCOM
    # # POS_B_W3_COM = POS_B_BA3 + A_B4W3 * R_W1234_WCOM
    # # POS_B_W4_COM = POS_B_BA4 + A_B4W4 * R_W1234_WCOM
    # #
    # ----------------------------------------------------------------------------------------------------------------------
    # 翼的动能计算
    # # 局部坐标系下的惯量矩阵
    JT_W_W = sp.Matrix([[J_W_XX, J_W_XY, J_W_XZ],
                        [J_W_XY, J_W_YY, J_W_YZ],
                        [J_W_XZ, J_W_YZ, J_W_ZZ]])  # print(sp.latex(JT_W_W))  # 经过经验

    # 局部坐标系下的角速度
    DT_AT_B_W1 = sp.Matrix([0.0, theta_1_d, phi_1_d])
    DT_AT_B_W2 = sp.Matrix([0.0, theta_2_d, phi_2_d])
    DT_AT_B_W3 = sp.Matrix([0.0, theta_3_d, phi_3_d])
    DT_AT_B_W4 = sp.Matrix([0.0, theta_4_d, phi_4_d])

    # 转动动能
    KET_W1 = 0.5 * DT_AT_B_W1.T * JT_W_W * DT_AT_B_W1
    KET_W2 = 0.5 * DT_AT_B_W2.T * JT_W_W * DT_AT_B_W2
    KET_W3 = 0.5 * DT_AT_B_W3.T * JT_W_W * DT_AT_B_W3
    KET_W4 = 0.5 * DT_AT_B_W4.T * JT_W_W * DT_AT_B_W4

    # # 翼的总能动能
    QE_K_W1234 = KET_W1[0] + KET_W2[0] + KET_W3[0] + KET_W4[0]  # + KET_WT1 + KET_WT2 + KET_WT3 + KET_WT4

    # # ----------------------------------------------------------------------------------------------------------------------
    # # # 翼势能
    # # P1：重力势能：-（新位置-起始位置）*重力
    #
    # # U_GR_W1 = (POS_B_W1_COM[2] - z_I_W1_COM_init) * g * m_W_1234  # 添加完正号的重力势能是对的     print(sp.latex(U_GR_W1))
    # # U_GR_W2 = (POS_B_W2_COM[2] - z_I_W2_COM_init) * g * m_W_1234
    # # U_GR_W3 = (POS_B_W3_COM[2] - z_I_W3_COM_init) * g * m_W_1234
    # # U_GR_W4 = (POS_B_W4_COM[2] - z_I_W4_COM_init) * g * m_W_1234
    #
    # # 总势能
    QE_U_W1234 = 0
    #
    # # ----------------------------------------------------------------------------------------------------------------------
    # # 翼的约束
    QE_C_W1234 = 0  # 翼本身没有约束（都计算到力中了）
    # ----------------------------------------------------------------------------------------------------------------------
    # 翼的做的功
    # 气动力做功（目前气动力作用点都平移到了绞点上，这里在机体坐标系下对器做功过程进行表述）
    # W1 -机体坐标系下-力（其坐标系转换在numpy中进行，这里输入的是已经调整好的）
    F_AERO_B1 = sp.Matrix([F_X_B1,
                           F_Y_B1,
                           F_Z_B1])
    WF_AERO_B1 = F_AERO_B1[0] * POS_B_BA1[0] + F_AERO_B1[1] * POS_B_BA1[1] + F_AERO_B1[2] * POS_B_BA1[2]

    # W2 -机体坐标系下-力（其坐标系转换在numpy中进行，这里输入的是已经调整好的）
    F_AERO_B2 = sp.Matrix([F_X_B2,
                           F_Y_B2,
                           F_Z_B2])
    WF_AERO_B2 = F_AERO_B2[0] * POS_B_BA2[0] + F_AERO_B2[1] * POS_B_BA2[1] + F_AERO_B2[2] * POS_B_BA2[2]

    # W3 -机体坐标系下-力（其坐标系转换在numpy中进行，这里输入的是已经调整好的）
    F_AERO_B3 = sp.Matrix([F_X_B3,
                           F_Y_B3,
                           F_Z_B3])
    WF_AERO_B3 = F_AERO_B3[0] * POS_B_BA3[0] + F_AERO_B3[1] * POS_B_BA3[1] + F_AERO_B3[2] * POS_B_BA3[2]

    # W4 -机体坐标系下-力（其坐标系转换在numpy中进行，这里输入的是已经调整好的）
    F_AERO_B4 = sp.Matrix([F_X_B4,
                           F_Y_B4,
                           F_Z_B4])
    WF_AERO_B4 = F_AERO_B4[0] * POS_B_BA4[0] + F_AERO_B4[1] * POS_B_BA4[1] + F_AERO_B4[2] * POS_B_BA4[2]

    # -----------------------------------------------------------------
    # W1 -翼面坐标系下-力矩
    WT_AERO_B1 = T_X_W1 * 0.0 + T_Y_W1 * theta_1 + T_Z_W1 * (phi_1)
    WT_AERO_B2 = T_X_W2 * 0.0 + T_Y_W2 * theta_2 + T_Z_W2 * (phi_2)
    WT_AERO_B3 = T_X_W3 * 0.0 + T_Y_W3 * theta_3 + T_Z_W3 * (phi_3)
    WT_AERO_B4 = T_X_W4 * 0.0 + T_Y_W4 * theta_4 + T_Z_W4 * (phi_4)

    # -----------------------------------------------------------------
    # W1 - 约束力矩作功

    SCALE_ADANCE = 40


    C_SCALE1 = (1 + SCALE_ADANCE * Stiff_TE_W1 / 0.0015)   # 参照Maxwell-粘弹性-模型
    C_SCALE2 = (1 + SCALE_ADANCE * Stiff_TE_W2 / 0.0015) # 参照Maxwell-粘弹性-模型
    C_SCALE3 = (1 + SCALE_ADANCE * Stiff_TE_W3 / 0.0015) # 参照Maxwell-粘弹性-模型
    C_SCALE4 = (1 + SCALE_ADANCE * Stiff_TE_W4 / 0.0015) # 参照Maxwell-粘弹性-模型

    W_Te_W1 = ((-1 * Stiff_TE_W1 * theta_1) * theta_1 +
               ( - C_ROTA * C_SCALE1 * theta_1_d) * theta_1 +
               VALUE_Te_YAW_1*theta_1)
    W_Te_W2 = ((-1 * Stiff_TE_W2 * theta_2) * theta_2 +
               ( - C_ROTA * C_SCALE2 * theta_2_d) * theta_2 +
               VALUE_Te_YAW_2*theta_2)
    W_Te_W3 = ((-1 * Stiff_TE_W3 * theta_3) * theta_3 +
               ( - C_ROTA * C_SCALE3 * theta_3_d) * theta_3 +
               VALUE_Te_YAW_3*theta_3)
    W_Te_W4 = ((-1 * Stiff_TE_W4 * theta_4) * theta_4 +
               ( - C_ROTA * C_SCALE4 * theta_4_d) * theta_4 +
               VALUE_Te_YAW_4*theta_4)










    # 气动力产生的总功

    QE_W_W1234 = (W_Te_W1 + W_Te_W2 + W_Te_W3 + W_Te_W4 +
                  WT_AERO_B1 + WT_AERO_B2 + WT_AERO_B3 + WT_AERO_B4 +
                  WF_AERO_B1 + WF_AERO_B2 + WF_AERO_B3 + WF_AERO_B4)

    # QE_W_W1234 = W_Te_W1 + W_Te_W2 + W_Te_W3 + W_Te_W4

    # %% 拉格朗日《函数》建立
    QE_K_TOL = QE_K_W1234 + QE_K_A1234
    QE_U_TOL = QE_U_W1234 + QE_U_A1234
    QE_C_TOL = QE_C_W1234 + QE_C_A1234
    QE_W_TOL = QE_W_W1234 + QE_W_A1234





    # %% 拉格朗日《函数》建立
    """
    任务：
        1，利用拉格朗日方法，聚合系统能量关系
    案例：
        L = T-V - Σ[λ_i * g_i(q)]
    """

    L_TOL = QE_K_TOL - QE_U_TOL - QE_C_TOL

    # %% 拉格朗日《方程》建立
    LE_phi_1 = - sp.diff(L_TOL, phi_1) + sp.diff(sp.diff(L_TOL, phi_1_d), t)
    LE_phi_2 = - sp.diff(L_TOL, phi_2) + sp.diff(sp.diff(L_TOL, phi_2_d), t)
    LE_phi_3 = - sp.diff(L_TOL, phi_3) + sp.diff(sp.diff(L_TOL, phi_3_d), t)
    LE_phi_4 = - sp.diff(L_TOL, phi_4) + sp.diff(sp.diff(L_TOL, phi_4_d), t)

    LE_theta_1 = - sp.diff(L_TOL, theta_1) + sp.diff(sp.diff(L_TOL, theta_1_d), t)
    LE_theta_2 = - sp.diff(L_TOL, theta_2) + sp.diff(sp.diff(L_TOL, theta_2_d), t)
    LE_theta_3 = - sp.diff(L_TOL, theta_3) + sp.diff(sp.diff(L_TOL, theta_3_d), t)
    LE_theta_4 = - sp.diff(L_TOL, theta_4) + sp.diff(sp.diff(L_TOL, theta_4_d), t)

    # %% 各个自由度的广义力计算
    Q_phi_1 = sp.diff(QE_W_TOL, phi_1)
    Q_phi_2 = sp.diff(QE_W_TOL, phi_2)
    Q_phi_3 = sp.diff(QE_W_TOL, phi_3)
    Q_phi_4 = sp.diff(QE_W_TOL, phi_4)

    Q_theta_1 = sp.diff(QE_W_TOL, theta_1)
    Q_theta_2 = sp.diff(QE_W_TOL, theta_2)
    Q_theta_3 = sp.diff(QE_W_TOL, theta_3)
    Q_theta_4 = sp.diff(QE_W_TOL, theta_4)

    # %% 引入广义力形成第二类拉格朗日方程
    IE_phi_1 = LE_phi_1 - Q_phi_1
    IE_phi_2 = LE_phi_2 - Q_phi_2
    IE_phi_3 = LE_phi_3 - Q_phi_3
    IE_phi_4 = LE_phi_4 - Q_phi_4

    IE_theta_1 = LE_theta_1 - Q_theta_1
    IE_theta_2 = LE_theta_2 - Q_theta_2
    IE_theta_3 = LE_theta_3 - Q_theta_3
    IE_theta_4 = LE_theta_4 - Q_theta_4

    print("系统《第二类拉格朗日方程》完成，这里将花费较多时间")  # 这里会花费较长时间
    # print(sp.latex(A_B2W1))
    # %% 方程组转化

    implicit_equations = [
        IE_phi_1, IE_phi_2, IE_phi_3, IE_phi_4,
        IE_theta_1, IE_theta_2, IE_theta_3, IE_theta_4
    ]

    if 0 in implicit_equations:
        print("警告：方程中有多余自由度，一个方程为零")
    else:
        print("祝贺：方程是完整的")


    # 创建自由度和导数符号列表
    degrees_of_freedom_dd = [phi_1_dd, phi_2_dd, phi_3_dd, phi_4_dd,
                             theta_1_dd, theta_2_dd, theta_3_dd, theta_4_dd
                             ]


    solutions = BASE_Cramer_SP(implicit_equations, degrees_of_freedom_dd)

    EQ_phi_1 = solutions[0]
    EQ_phi_2 = solutions[1]
    EQ_phi_3 = solutions[2]
    EQ_phi_4 = solutions[3]
    EQ_theta_1 = solutions[4]
    EQ_theta_2 = solutions[5]
    EQ_theta_3 = solutions[6]
    EQ_theta_4 = solutions[7]

    print("显式方程1", sp.latex(EQ_phi_1))
    print("显式方程2", sp.latex(EQ_phi_2))
    print("显式方程3", sp.latex(EQ_phi_3))
    print("显式方程4", sp.latex(EQ_phi_4))
    print("显式方程5", sp.latex(EQ_theta_1))
    print("显式方程6", sp.latex(EQ_theta_2))
    print("显式方程7", sp.latex(EQ_theta_3))
    print("显式方程8", sp.latex(EQ_theta_4))

    # %% 数值转化的状态向量拼接

    LIST_FREEDOM = [phi_1, phi_2, phi_3, phi_4,
                    theta_1, theta_2, theta_3, theta_4
                    ]
    LIST_FREEDOM_d = [phi_1_d, phi_2_d, phi_3_d, phi_4_d,
                      theta_1_d, theta_2_d, theta_3_d, theta_4_d
                      ]

    # 控制信号序列
    ARG_CONTROL = [K_A1, K_A2, K_A3, K_A4, CONTROL_SAFW, CONTROL_SAHW, TORQUE_M_1, TORQUE_M_2, TORQUE_M_3, TORQUE_M_4]

    # 气动力
    ARG_AERO = [F_X_B1, F_Y_B1, F_Z_B1, T_X_W1, T_Y_W1, T_Z_W1,
                F_X_B2, F_Y_B2, F_Z_B2, T_X_W2, T_Y_W2, T_Z_W2,
                F_X_B3, F_Y_B3, F_Z_B3, T_X_W3, T_Y_W3, T_Z_W3,
                F_X_B4, F_Y_B4, F_Z_B4, T_X_W4, T_Y_W4, T_Z_W4]

    ARG_ELASTIC = [Stiff_TE_W1, Stiff_TE_W2, Stiff_TE_W3, Stiff_TE_W4,
                   VALUE_Te_YAW_1, VALUE_Te_YAW_2, VALUE_Te_YAW_3, VALUE_Te_YAW_4]

    Inertia_characteristics = [J_M_ZZ,J_W_XX,J_W_YY,J_W_ZZ,J_W_XY,J_W_XZ,J_W_YZ]

    sympy_tuple = tuple(LIST_FREEDOM + LIST_FREEDOM_d + ARG_CONTROL + ARG_AERO + ARG_ELASTIC + Inertia_characteristics)

    # %% 数值化转化

    OR_NDE_phi_1 = sp.lambdify(sympy_tuple, EQ_phi_1)
    OR_NDE_phi_2 = sp.lambdify(sympy_tuple, EQ_phi_2)
    OR_NDE_phi_3 = sp.lambdify(sympy_tuple, EQ_phi_3)
    OR_NDE_phi_4 = sp.lambdify(sympy_tuple, EQ_phi_4)

    OR_NDE_theta_1 = sp.lambdify(sympy_tuple, EQ_theta_1)
    OR_NDE_theta_2 = sp.lambdify(sympy_tuple, EQ_theta_2)
    OR_NDE_theta_3 = sp.lambdify(sympy_tuple, EQ_theta_3)
    OR_NDE_theta_4 = sp.lambdify(sympy_tuple, EQ_theta_4)

    NDE_phi_1 = njit(OR_NDE_phi_1)
    NDE_phi_2 = njit(OR_NDE_phi_2)
    NDE_phi_3 = njit(OR_NDE_phi_3)
    NDE_phi_4 = njit(OR_NDE_phi_4)

    NDE_theta_1 = njit(OR_NDE_theta_1)
    NDE_theta_2 = njit(OR_NDE_theta_2)
    NDE_theta_3 = njit(OR_NDE_theta_3)
    NDE_theta_4 = njit(OR_NDE_theta_4)


    return NDE_phi_1,NDE_phi_2,NDE_phi_3,NDE_phi_4, NDE_theta_1,NDE_theta_2,NDE_theta_3,NDE_theta_4



@njit
def SPRING_sigmoid_NB(x, OUTER_SCALE=10000):
    """
    张紧判断

    :param x:
    :param OUTER_SCALE:
    :return:
    """
    TOTAL_SCALE = 1
    MOVE = 13  # 直接平移位置，相当于《高通滤波器》的通过频率
    x = x / OUTER_SCALE - MOVE

    x2 = 80000 / OUTER_SCALE - MOVE

    VALUE1 = TOTAL_SCALE / (1 + np.exp(-x * 2))
    VALUE2 = TOTAL_SCALE / (1 + np.exp(-x2 * 2))

    return (VALUE1/VALUE2)

@njit
def SPRING_sigmoid_SHAPE(theta, SCACLE_GEM=0.1):
    # Corrected the radians conversion for 90 degrees

    # # Calculate spring constant
    # # （决定y轴高度）刚度取值 -----------------------------------------------------------
    # tf = 0.03 / 1000      # 不要改：TPU膜的厚度
    # E_tpu = 42 * 1000000  # 不要改：TPU的弹性模量
    # L_install = 20 / 1000 # 不要改：几何定义，不要改，要改就改直接整体缩放
    # L_inforce = 10 / 1000 # 不要改：几何定义，不要改，要改就改直接整体缩放
    # # 总集计算
    # K_stand = tf * L_inforce * E_tpu / np.radians(65) * L_install * 0.5 * SCACLE_GEM

    # Sigmoid-like function calculation
    theta_n = theta * 1.35
    mu = 0
    sigma = 3
    KE_VALUE = (((1.35 - (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((theta_n - mu) / sigma) ** 8)) / 1.35) ** 2 - ((1.35 - (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((0 - mu) / sigma) ** 8)) / 1.35) ** 2)
    KE_VALUE_ref = (((1.35 - (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((np.radians(75) - mu) / sigma) ** 8)) / 1.35) ** 2 - ((1.35 - (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((0 - mu) / sigma) ** 8)) / 1.35) ** 2)

    KE_VALUE = KE_VALUE/KE_VALUE_ref

    return K_stand * KE_VALUE**2


@njit
def SIGN_square_plus_AIO(x,traget_angle,b=0.005):
    x = (x - abs(traget_angle))
    return (x + np.sqrt(x**2 + b))/2

@njit
def SPRING_sigmoid_SHAPE_AIO(theta):
    TARGET = np.radians(70)
    CONTROL_FACTOR = 10*(SIGN_square_plus_AIO(theta, TARGET) + SIGN_square_plus_AIO(-theta,-TARGET))
    return CONTROL_FACTOR**2


# %% 串列翼部分

@njit
def FAST_inter(y,CENTER_SP_1,CENTER_SP_2,CENTER_SP_3,CENTER_SP_4,config_AM,MAX_W0,MAX_W1,MAX_W2):
    # 这里建立标准角
    IA_ANGLE_1 = -(y[0] - CENTER_SP_1) / (config_AM * 2)
    IA_ANGLE_4 = -(y[3] - CENTER_SP_4) / (config_AM * 2)
    IA_ANGLE_2 = (y[1] - CENTER_SP_2) / (config_AM * 2)
    IA_ANGLE_3 = (y[2] - CENTER_SP_3) / (config_AM * 2)

    W_THETA1_A = -y[8] / (config_AM * 2 * MAX_W0 * 2)
    W_THETA2_A = -y[11] / (config_AM * 2 * MAX_W1 * 2)
    W_THETA3_A = (W_THETA1_A - W_THETA2_A) / (MAX_W2 * 2)

    W_THETA1_B = y[9] / (config_AM * 2 * MAX_W0 * 2)
    W_THETA2_B = y[10] / (config_AM * 2 * MAX_W1 * 2)
    W_THETA3_B = (W_THETA1_B - W_THETA2_B) / (MAX_W2 * 2)

    X0A = W_THETA1_A
    X1A = W_THETA2_A
    X2A = W_THETA3_A
    X3A = IA_ANGLE_1
    X4A = IA_ANGLE_4
    X5A = IA_ANGLE_1 - IA_ANGLE_4

    X6A = np.sin(2 * IA_ANGLE_1 * np.pi)
    X7A = np.sin(2 * IA_ANGLE_4 * np.pi)
    X8A = np.sin(4 * IA_ANGLE_1 * np.pi)
    X9A = np.sin(4 * IA_ANGLE_4 * np.pi)
    X10A = np.sin(8 * IA_ANGLE_1 * np.pi)
    X11A = np.sin(8 * IA_ANGLE_4 * np.pi)

    X0B = W_THETA1_B
    X1B = W_THETA2_B
    X2B = W_THETA3_B
    X3B = IA_ANGLE_2
    X4B = IA_ANGLE_3
    X5B = IA_ANGLE_2 - IA_ANGLE_3

    X6B = np.sin(2 * IA_ANGLE_2 * np.pi)
    X7B = np.sin(2 * IA_ANGLE_3 * np.pi)
    X8B = np.sin(4 * IA_ANGLE_2 * np.pi)
    X9B = np.sin(4 * IA_ANGLE_3 * np.pi)
    X10B = np.sin(8 * IA_ANGLE_2 * np.pi)
    X11B = np.sin(8 * IA_ANGLE_3 * np.pi)

    Coefficient_fore_A = INTERACTION_fore(X0A, X1A, X2A, X3A, X4A, X5A, X6A, X7A, X8A, X9A, X10A, X11A)
    Coefficient_fore_B = INTERACTION_fore(X0B, X1B, X2B, X3B, X4B, X5B, X6B, X7B, X8B, X9B, X10B, X11B)

    Coefficient_back_A = INTERACTION_back(X0A, X1A, X2A, X3A, X4A, X5A, X6A, X7A, X8A, X9A, X10A, X11A)
    Coefficient_back_B = INTERACTION_back(X0B, X1B, X2B, X3B, X4B, X5B, X6B, X7B, X8B, X9B, X10B, X11B)
    return Coefficient_fore_A, Coefficient_fore_B, Coefficient_back_A, Coefficient_back_B


@njit
def INTERACTION_fore(X0, X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11):
    part1 = X0 * X8 * (-11.453 * X0 * (
            -5 * X6 + X7) - 22.906 * X1 + 11.453 * X2 - 11.453 * X4 + 11.453 * X5 - 11.453 * X7 - 11.453 * np.sin(
        np.sin(X8)))
    part2 = X0 * X9 * (
            11.453 * X10 + 11.453 * X11 + 11.453 * X2 - 22.906 * X4 - 11.453 * X5 * (X1 - X11 - X5) - 11.453 * X7)
    part3 = 9 * X1 - 7.892 * X2 + 7.892 * X4 + X8 - 6.166

    result = part1 + part2 + part3
    return result / 100 + 1

@njit
def INTERACTION_back(X0, X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11):
    term1 = 49.314 * X1
    term2 = (X2 - X5 - np.sin(X1 - 124.935)) * (X2 + X4 - X8 - 0.993578913297163) * (X3 + X4 + X6 + X7)
    term3 = (-2 * X1 - X10 - X11 + X2 - X7 * (X1 + X7 - np.sin(X10 - X5)) * (X1 + X11 - X9 - 45.822) - X8
             - (X2 + X4 - np.sin(X1 - 124.935)) * (2 * X0 - 3 * X10 - X9 - 34.855) * (X3 + X4 + X6 + X7) - 49.314)

    result = term1 - term2 * term3
    return result / 100 + 1


# %% 偏航控制

@njit
def SIGN_square_plus(x,b=0.03,PRE = 0.002):
    """

    :param x:
    :param b:
    :param PRE:
    :return:
    """
    # 进行符号判断
    x = (x - PRE)
    return (x + np.sqrt(x**2 + b))/2




def R_xyz_NP(theta_x=0.0, theta_y=0.0, theta_z=0.0):
    """
    新的=旋转矩阵x旧的

    这是角度是正的，不然角度是负的

    :param theta_x: 当地x轴
    :param theta_y: 当地y轴
    :param theta_z: 当地z轴
    :return:
    """
    Rx = np.array([[1.0, 0.0, 0.0],
                   [0.0, np.cos(theta_x), -np.sin(theta_x)],
                   [0.0, np.sin(theta_x), np.cos(theta_x)]])
    Ry = np.array([[np.cos(theta_y), 0.0, np.sin(theta_y)],
                   [0.0, 1.0, 0.0],
                   [-np.sin(theta_y), 0.0, np.cos(theta_y)]])
    Rz = np.array([[np.cos(theta_z), -np.sin(theta_z), 0.0],
                   [np.sin(theta_z), np.cos(theta_z), 0.0],
                   [0.0, 0.0, 1.0]])

    return np.dot(np.dot(Rz,Ry),Rx)



def R_xyz_NP_z_fast(theta_z=0.0):
    """
    新的=旋转矩阵x旧的

    这是角度是正的，不然角度是负的

    :param theta_x: 当地x轴
    :param theta_y: 当地y轴
    :param theta_z: 当地z轴
    :return:
    """
    Rz = np.array([[np.cos(theta_z), -np.sin(theta_z), 0.0],
                   [np.sin(theta_z), np.cos(theta_z), 0.0],
                   [0.0, 0.0, 1.0]])
    return Rz


@njit
def GET_PS_POINT(CONTROL_SAFW,  # 前翼控制量
                 CONTROL_SAHW,  # 后翼控制量
                 DX_B_FW = 50 / 1000,
                 DY_B_FW = 25 / 1000,
                 DZ_B_FW = 30 / 1000,
                 DX_B_HW = -50 / 1000,
                 DY_B_HW = 25 / 1000,
                 DZ_B_HW = - 30 / 1000,
                 LEN_LEG = 25 / 1000,
                 LEN_DEPTH_ACT = 7 / 1000,
                 LEN_STRING = 0.009):
    """
    这里计算《偏航舵机》的尖点位置

    这个不会涉及到数值求解的迭代过程，会在每个时间步计算一次

    :param CONTROL_SAFW:
    :param CONTROL_SAHW:
    :param phi_1:
    :param phi_2:
    :param phi_3:
    :param phi_4:
    :param DX_B_FW:
    :param DY_B_FW:
    :param DZ_B_FW:
    :param DX_B_HW:
    :param DY_B_HW:
    :param DZ_B_HW:
    :param LEN_LEG:
    :param LEN_DEPTH_ACT:
    :param LEN_STRING:
    :return:
    """

    # 数据导入  -----------------------------------------------------------------------------------------------
    DZ_R_B_FWJ = LEN_DEPTH_ACT  # 攻角控制控制器的整体偏移
    DZ_R_B_HWJ = LEN_DEPTH_ACT  # 攻角控制控制器的整体偏移

    L_B_FWARM = LEN_LEG  # 攻角控制半摇臂长
    L_B_HWARM = LEN_LEG  # 攻角控制半摇臂长

    # 机体坐标系下控制点的位置 -----------------------------------------------------------------------------------------------
    X_R_B_FWJ = DX_B_FW
    X_R_B_HWJ = DX_B_HW

    Z_R_B_FWJ = DZ_B_FW + DZ_R_B_FWJ
    Z_R_B_HWJ = DZ_B_HW + DZ_R_B_HWJ

    X_R_B_D1 = - np.sin(CONTROL_SAFW) * L_B_FWARM + X_R_B_FWJ
    Y_R_B_D1 = + np.cos(CONTROL_SAFW) * L_B_FWARM
    Z_R_B_D1 = Z_R_B_FWJ
    O_POS_B_D1 = np.array([X_R_B_D1, Y_R_B_D1, Z_R_B_D1])

    X_R_B_D2 = + np.sin(CONTROL_SAFW) * L_B_FWARM + X_R_B_FWJ
    Y_R_B_D2 = - np.cos(CONTROL_SAFW) * L_B_FWARM
    Z_R_B_D2 = Z_R_B_FWJ
    O_POS_B_D2 = np.array([X_R_B_D2, Y_R_B_D2, Z_R_B_D2])

    X_R_B_D3 = + np.sin(CONTROL_SAHW) * L_B_HWARM + X_R_B_HWJ
    Y_R_B_D3 = - np.cos(CONTROL_SAHW) * L_B_HWARM
    Z_R_B_D3 = Z_R_B_HWJ
    O_POS_B_D3 = np.array([X_R_B_D3, Y_R_B_D3, Z_R_B_D3])

    X_R_B_D4 = - np.sin(CONTROL_SAHW) * L_B_HWARM + X_R_B_HWJ
    Y_R_B_D4 = + np.cos(CONTROL_SAHW) * L_B_HWARM
    Z_R_B_D4 = Z_R_B_HWJ
    O_POS_B_D4 = np.array([X_R_B_D4, Y_R_B_D4, Z_R_B_D4])


    # 翼旋转轴位置 -----------------------------------------------------------------------------------------------
    POS_B_BA1 = np.array([DX_B_FW, DY_B_FW, DZ_B_FW])
    POS_B_BA2 = np.array([DX_B_FW, -DY_B_FW, DZ_B_FW])
    POS_B_BA3 = np.array([DX_B_HW, -DY_B_HW, DZ_B_HW])
    POS_B_BA4 = np.array([DX_B_HW, DY_B_HW, DZ_B_HW])

    # 控制点在驱动系下地相对位置 -----------------------------------------------------------------------------------------------
    POS_PS_1 = O_POS_B_D1-POS_B_BA1
    POS_PS_2 = O_POS_B_D2-POS_B_BA2
    POS_PS_3 = O_POS_B_D3-POS_B_BA3
    POS_PS_4 = O_POS_B_D4-POS_B_BA4
    return POS_PS_1,POS_PS_2,POS_PS_3,POS_PS_4


GESN = 20

def ITRE_C_YAW(POS_PS_i,
               phi_i,
               theta_i,theta_id,
               L_rope = 11 / 1000,
               L_wing_off=10 / 1000,
               L_BAR=5 / 1000,
               K_YAW=K_stand*10000*GESN
               ):
    # 把 POS_PS_i 投影到驱动坐标系下（无论如何，约束的功是发生在驱动坐标系下的） --------------------------------------------------
    A_B4Wi = R_xyz_NP(theta_y=theta_i, theta_z=phi_i)

    POS_PS_W_i = np.dot(A_B4Wi.T, POS_PS_i)  # 舵机尖点在翼面坐标系下的投影，OK

    # 绳子长计算 --------------------------------------------------------------------------------------------------------
    POS_PC_W_i = np.array([0.0, L_wing_off, L_BAR])  # 摇臂顶点在翼面坐标系下的投影，OK

    VECTOR_i = POS_PC_W_i - POS_PS_W_i

    L_ps2po_i = np.linalg.norm(VECTOR_i)  # 当前绳子长

    # 刚度值计算其标准力计算 ----------------------------------------------------------------------------------------------
    D_K = softplus_K_YAW_K(L_ps2po_i - L_rope)  # 伸长量
    D_C = softplus_K_YAW_C(L_ps2po_i - L_rope)


    F_value_i = D_K * K_YAW*0.25 - D_C * 2000 * theta_id * C_ROTA*5*GESN*100

    V_F_i = F_value_i * VECTOR_i / L_ps2po_i

    # 力投影和力矩计算 ---------------------------------------------------------------------------------------------------
    V_pc_i = np.array([-np.cos(theta_i), 0.0, np.sin(theta_i)])

    PF_i = PROJECT_V(V_F_i, V_pc_i)

    Torue_i = PF_i * L_BAR # 这个方向应该是正确的，反过来后结果非常奇怪


    # %% 阻尼力
    Torue_i = Torue_i


    return Torue_i





def PROJECT_V(V_F_1,Direction_1):
    # 计算点积
    dot_product = np.dot(V_F_1, Direction_1)

    # 计算Direction_1的模长
    norm_Direction_1 = np.linalg.norm(Direction_1)

    # 计算投影的模长
    projection_length = dot_product / norm_Direction_1
    return projection_length



def softplus_K_YAW_C(x):
    """
    相对标准的YAW函数，这里将x的取值放缩在mm量级

    :param x:
    :return:
    """
    x  = x + 0.0008  # 这个不要调整，不然就会让扭转过程不能成仙上部平坦的过程
    x = (x)*10000
    return ((np.log(0.0001+np.exp(x))+9.210340371976182)/3000)**2


def softplus_K_YAW_K(x):
    """
    相对标准的YAW函数，这里将x的取值放缩在mm量级

    :param x:
    :return:
    """
    x = x + 0.005 # 这个不要调整，不然就会让扭转过程不能成仙上部平坦的过程
    x = x *10000
    return ((np.log(0.0001+np.exp(x))+9.210340371976182)/3000)**2


def SHIFT_J(J_W_XX, J_W_YY, J_W_ZZ, J_W_XY, J_W_XZ, J_W_YZ):
    # 构造原始惯量矩阵
    J_W = np.array([[J_W_XX, J_W_XY, J_W_XZ],
                    [J_W_XY, J_W_YY, J_W_YZ],
                    [J_W_XZ, J_W_YZ, J_W_ZZ]])

    # 绕y轴旋转45度的变换矩阵
    theta = np.pi / 4  # 角度转换为弧度
    R = np.array([[np.cos(theta), 0, np.sin(theta)],
                  [0, 1, 0],
                  [-np.sin(theta), 0, np.cos(theta)]])

    # 计算新的惯量矩阵
    J_new = R @ J_W @ R.T

    # 提取新的J'_ZZ值
    J_W_ZZ_new = J_new[2, 2]

    return J_W_ZZ_new