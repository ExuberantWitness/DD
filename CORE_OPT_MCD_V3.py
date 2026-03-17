# import time
import numpy as np
from scipy.integrate import solve_ivp
from CONTROL_PIDcontrol_V6 import PID_interface
from CONTROL_SIGNAL_V5 import GPG_signal
from CORE_DDD_MFD_v10_MOO import SPRING_sigmoid_SHAPE, SPRING_sigmoid_NB,SHIFT_J
import pandas as pd
from CMFD_QS_MODEL_FAST_206_FLY import AERO_QS_FLY,\
    GENE_SHAPE_DATE_trapezoid,FROM_DATE_TO_FUN,compute_F_rot2,BQS_compute_r2,BQS_trapz
from CORE_Inertia_WING import Inertia_WING
from CORE_MOTOR_MODEL_V2 import GET_MOTOR_DATA_V2,ELE_MOTOR,NEST_INT
import matplotlib.pyplot as plt
from CORE_MCD import trajectory_generation_V5,frechet3D_V4,MCD_GENERAL_SHIFT
from datetime import datetime



# 固定参数
D_dc = 0.005
N_blade = 51
D_bat = 1400.6  # J/g 电池能量密度
SHIFT_N2g = 1/9.8*1000 # 用来将升力转化成g力
dt = 0.0005
CONST_c = 0.1  # 变大会让不良状态进入到张紧判断中，进而出现非常奇怪的现象，比如高速振荡，但是0.01比较合理
VALUE_Te_YAW_1 = 0
VALUE_Te_YAW_2 = 0
VALUE_Te_YAW_3 = 0
VALUE_Te_YAW_4 = 0
Gear_Motor = 12       # 电机齿轮上的齿轮数量
M_gear = 0.13 / 1000 # 传动传动系统模数
COUNT_P_etc = 1  # 电子系统功耗，单位W
const_SWB = 4        # 中枢模式振荡器的预先数量
const_DT = 0.0005     # 时间步长
const_NUMB = 400      # 评估步数量
GENE_SCALE = 7
NUMB = 2000
MCD_AM_MAX_LIFT = 90
CONST_INIT_P = 24.0e-03 * GENE_SCALE
CONST_INIT_I = 0.1e-05 * GENE_SCALE
CONST_INIT_D = 35e-06 * GENE_SCALE
CONST_E_trans = 0.8 # 目前假设机械传动效率为90%，参见《Untethered Flight of an At-Scale Dual-Motor Hummingbird Robot With Bio-Inspired Decoupled Wings》
# %% 核心计算
def MCD_OPT_CORE(MCD_AM,MCD_FRE_A,MCD_SPAN,ID_MOTOR,MCD_GEAR_RATIO,
                 MCD_S_AR,MCD_TR,
                 NDE_phi_1, NDE_phi_2, NDE_phi_3, NDE_phi_4,NDE_theta_1, NDE_theta_2, NDE_theta_3, NDE_theta_4,
                 P_core_HOVER,R_ref,Individual_Differences_HOVER,FAST_inter,
                 CENTER_SP_1, CENTER_SP_2, CENTER_SP_3, CENTER_SP_4,IS_CHECK=False,STAND_NEAB = 250):
    """

    :param MCD_AM:
    :param MCD_FRE_A:
    :param MCD_SPAN:
    :param ID_MOTOR:
    :param MCD_GEAR_RATIO:
    :param MCD_S_AR:
    :param MCD_TR:

    :param NDE_phi_1:
    :param NDE_phi_2:
    :param NDE_phi_3:
    :param NDE_phi_4:
    :param NDE_theta_1:
    :param NDE_theta_2:
    :param NDE_theta_3:
    :param NDE_theta_4:

    :param P_core_HOVER:

    :param R_ref:
    :param NUMB:
    :param Individual_Differences_HOVER:
    :param FAST_inter:

    :param const_SWB:
    :param const_DT:
    :param const_NUMB:
    :param CONST_INIT_P:
    :param CONST_INIT_I:
    :param CONST_INIT_D:
    :param CONST_E_trans:
    :param CENTER_SP_1:
    :param CENTER_SP_2:
    :param CENTER_SP_3:
    :param CENTER_SP_4:
    :param IS_SAVE:
    :return:
    """
    FUXX_FLAP = 0
    P_aircraft,_,_ = trajectory_generation_V5(MCD_AM*2,
                                              0,
                                              MCD_FRE_A,
                                              0,
                                              180, R_ref, NUMB,dt)
    C_dym = frechet3D_V4(P_core_HOVER, P_aircraft) / R_ref
    if C_dym < Individual_Differences_HOVER:
        C_dym = 0
    else:
        C_dym = C_dym - Individual_Differences_HOVER

    D_STATIC = MCD_GENERAL_SHIFT(MCD_SPAN * 1000)
    D_DYM = MCD_GENERAL_SHIFT(C_dym*MCD_SPAN * 1000,S_aim = 0)
    D_total = D_STATIC + D_DYM
    # 结果输出
    # print("MCD_静态",D_STATIC)
    # print("MCD_动态",D_DYM)
    # print("MCD_总",D_total)

    # %% 电机参数读取和基本惯量参数
    U_motor, I_zero, I_max, R_motor, KV_motor, W_max, MASS_MOTOR, INT_MOTOR = GET_MOTOR_DATA_V2(NEST_INT(ID_MOTOR))
    MCD_J_M_ZZ = INT_MOTOR/10000000 *100
    J_GEAR = 4.7 / 1000000000 * (MCD_GEAR_RATIO / 10) ** 2  # 参见《Untethered Flight of an At-Scale Dual-Motor Hummingbird Robot With Bio-Inspired Decoupled Wings》

    # %% 辅助部件生成
    S_total = MCD_SPAN

    C_mean = (S_total - D_dc) / MCD_S_AR
    c_tip = 2 * C_mean / (1 + 1 / MCD_TR)
    C_root = c_tip / MCD_TR

    Data_R,Data_LE,Data_TE = GENE_SHAPE_DATE_trapezoid(S_AR = MCD_S_AR,
                                                       TR = MCD_TR,
                                                       SPAN =S_total - D_dc,
                                                       num_blade = N_blade,
                                                       IS_PLOT = False) # 普度大学数据

    J_W_XX,J_W_YY,J_W_ZZ,J_W_XY,J_W_XZ,J_W_YZ = Inertia_WING(S_total = S_total,
                                                             D_dc = D_dc,
                                                             C_root = C_root,
                                                             c_tip = c_tip)
    J_W_ZZ_NEW = SHIFT_J(J_W_XX, J_W_YY, J_W_ZZ, J_W_XY, J_W_XZ, J_W_YZ)

    CONST_J = MCD_J_M_ZZ + J_GEAR + J_W_ZZ_NEW*1.3

    # %% 翼框架计算

    FUN_LE, R_new_LE, Z_new_LE = FROM_DATE_TO_FUN(Data_R, Data_LE, N_blade)
    FUN_TE, R_new_TE, Z_new_TE = FROM_DATE_TO_FUN(Data_R, Data_TE, N_blade)
    IS_PLOT = False
    if IS_PLOT == True:
        # 绘制结果
        plt.figure(figsize=(10, 6))
        plt.plot(R_new_LE, Z_new_LE, 'o', label='interp1d')  # 已知数据点
        plt.plot(Data_R, Data_LE, '-', label='real')  # 插值曲线
        plt.legend()
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.plot(R_new_TE, Z_new_TE, 'o', label='interp1d')  # 已知数据点
        plt.plot(Data_R, Data_TE, '-', label='real')  # 插值曲线
        plt.legend()
        plt.show()

    # 基本几何参数计算
    rho = 1.225
    PHIm = 60 * np.pi / 180
    f = 35
    v = 1.506e-5


    r = R_new_LE
    del_R = min(Data_R)
    R = max(Data_R)
    # print("------最大翼长",self.R)
    B_base = max(Data_R) - min(Data_R)
    # print("------基准翼长", self.B_base)
    dr = abs(R_new_LE[0] - R_new_LE[1])  # 获得离散步长


    C = np.abs(Z_new_LE - Z_new_TE)

    CORE_F_rot2 = compute_F_rot2(Z_new_LE, Z_new_TE, r, rho, 1, dr)

    X_ROT = np.zeros(Z_new_LE.shape[0])


    # 累积变量
    S = np.sum(dr * C)
    # print("------面积：", self.S) # OK
    AR = B_base ** 2 / S
    # print("------展弦比：", self.AR)
    r2 = BQS_compute_r2(R_new_LE, C, S, R)

    # print("------r2值：", self.r2)
    R2 = R * r2  # 计算有量纲面积二阶矩 # OK
    # # print("------R2值：", self.R2)
    # C2_R = BQS_trapz(R_new_LE, C ** 2 * R_new_LE)
    # # print("------C2_R值：", self.C2_R)
    # F_rot2 = compute_F_rot2(Z_new_LE, Z_new_TE, R_new_LE, rho, 1, dr)

    # 等效翼尖根弦长
    coefficients_LE = np.polyfit(Data_R, Data_LE, 1)
    slope, intercept = coefficients_LE  # polyfit返回的系数，对于线性拟合，coefficients[0]是斜率，coefficients[1]是截距
    # 创建一个拟合线，用于绘图或计算
    C_ROOT_LE = slope * min(Data_R) + intercept
    C_TIP_LE = slope * max(Data_R) + intercept

    coefficients_TE = np.polyfit(Data_R, Data_TE, 1)
    slope, intercept = coefficients_TE  # polyfit返回的系数，对于线性拟合，coefficients[0]是斜率，coefficients[1]是截距
    # 创建一个拟合线，用于绘图或计算
    C_ROOT_TE = slope * min(Data_R) + intercept
    C_TIP_TE = slope * max(Data_R) + intercept

    C_TIP = np.abs(C_TIP_LE - C_TIP_TE)
    C_ROOT = np.abs(C_ROOT_LE - C_ROOT_TE)
    C_mean = (C_TIP + C_ROOT) / 2
    # # print("------C_mean值：", self.C_mean)
    # lamb = C_TIP / C_ROOT
    # # print("------lamb值：", self.lamb)

    # 常量气动力参数
    I = PHIm * 2 * 2  # OK
    Uref = R2 * I * f  # OK
    Re = C_mean * Uref / v  # 雷诺数 # OK
    Ro = R2 / C_mean  # OK

    # 虚拟压心(用来计算参考值)
    DUM_RC_R2_tr = 1.0768000587784088 * R2  # 主要用于计算旋转力和附加质量力
    DUM_XC_c_mean_tr = 0.38841422581129026 * C_mean  # 主要用于计算旋转力和附加质量力
    DUM_XC_c_mean_tr_ratio = 0.38841422581129026  # 主要用于计算每个片条上的平动力

    # f_lma = 47.7 * lamb ** (-0.0019) - 46.7
    # f_ARa = 1.294 - 0.590 * AR ** (-0.662)
    # f_a = 0.776 + 1.911 * Re ** (-0.687)
    # add_c = np.pi * rho * 0.25
    # ADD1_trapz = BQS_trapz(R_new_LE, C ** 2 * R_new_LE)
    # ADD2_trapz = BQS_trapz(R_new_LE, C ** 2 * (np.abs(Z_new_LE + Z_new_TE) / 2 - X_ROT))

    # f_r = 1.570 - -1.239 * (1 / R * BQS_trapz(R_new_LE, X_ROT) / C_mean);
    # C_rot1 = 0.842 - 0.507 * Re ** (-0.1577)

    # 转动力和附加质量力的恒定压心
    P_S_cop_rot = 0.993 * R2
    P_C_cop_rot = 0.398 * C_mean
    # P_S_cop_add = 1.078 * R2
    # P_C_cop_add = 0.5 * C_mean
    #
    # 翼参数初始化
    QSM_W1 = AERO_QS_FLY(C,
                         r,
                         Re,
                         AR,
                         Ro,
                         N_blade,
                         dr,
                         DUM_XC_c_mean_tr_ratio,
                         CORE_F_rot2,
                         P_S_cop_rot,
                         P_C_cop_rot,
                         DUM_XC_c_mean_tr,
                         DUM_RC_R2_tr,
                         rho,
                         CENTER = CENTER_SP_1)
    QSM_W2 = AERO_QS_FLY(C,
                         r,
                         Re,
                         AR,
                         Ro,
                         N_blade,
                         dr,
                         DUM_XC_c_mean_tr_ratio,
                         CORE_F_rot2,
                         P_S_cop_rot,
                         P_C_cop_rot,
                         DUM_XC_c_mean_tr,
                         DUM_RC_R2_tr,
                         rho,
                         CENTER = CENTER_SP_2)
    QSM_W3 = AERO_QS_FLY(C,
                         r,
                         Re,
                         AR,
                         Ro,
                         N_blade,
                         dr,
                         DUM_XC_c_mean_tr_ratio,
                         CORE_F_rot2,
                         P_S_cop_rot,
                         P_C_cop_rot,
                         DUM_XC_c_mean_tr,
                         DUM_RC_R2_tr,
                         rho,
                         CENTER = CENTER_SP_3)
    QSM_W4 = AERO_QS_FLY(C,
                         r,
                         Re,
                         AR,
                         Ro,
                         N_blade,
                         dr,
                         DUM_XC_c_mean_tr_ratio,
                         CORE_F_rot2,
                         P_S_cop_rot,
                         P_C_cop_rot,
                         DUM_XC_c_mean_tr,
                         DUM_RC_R2_tr,
                         rho,
                         CENTER = CENTER_SP_4)

    # %% 基本建模参数（# 初始扑动角，y轴正方向为0，正方向为z轴正方向）
    INIT_PHI_1, INIT_PHI_2, INIT_PHI_3, INIT_PHI_4 = CENTER_SP_1, CENTER_SP_2, CENTER_SP_3, CENTER_SP_4
    INIT_THETA_1, INIT_THETA_2, INIT_THETA_3, INIT_THETA_4 = 0.0, 0.0, 0.0, 0.0


    # %% 中枢模式振荡器初始化
    EXP_s_SYS_r = GPG_signal(Horizon = const_SWB + 2,
                             CONST_Phase_different=np.pi,
                             INIT_M1_PP=0,
                             INIT_M2_PP=0,
                             INIT_M3_PP=0,
                             INIT_M4_PP=0,
                             INIT_M1_P=0,
                             INIT_M2_P=0,
                             INIT_M3_P=0,
                             INIT_M4_P=0,
                             dt=const_DT,
                             CPG_SCALE_RATIO = 10)

    EXP_s_SYS_r_MAX_LIFT = GPG_signal(Horizon = const_SWB + 2,
                                      CONST_Phase_different=np.pi,
                                      INIT_M1_PP=0,
                                      INIT_M2_PP=0,
                                      INIT_M3_PP=0,
                                      INIT_M4_PP=0,
                                      INIT_M1_P=0,
                                      INIT_M2_P=0,
                                      INIT_M3_P=0,
                                      INIT_M4_P=0,
                                      dt=const_DT,
                                      CPG_SCALE_RATIO = 10)

    # %% 自适应PID控制器初始化
    pid_controller_5 = PID_interface(CONST_INIT_P, CONST_INIT_I, CONST_INIT_D, const_DT)
    pid_controller_6 = PID_interface(CONST_INIT_P, CONST_INIT_I, CONST_INIT_D, const_DT)
    pid_controller_7 = PID_interface(CONST_INIT_P, CONST_INIT_I, CONST_INIT_D, const_DT)
    pid_controller_8 = PID_interface(CONST_INIT_P, CONST_INIT_I, CONST_INIT_D, const_DT)

    # %% 循环迭代准备
    LAST_positions = [0.0] * 8
    LAST_velocities = [0.0] * 8
    LAST_positions[0], LAST_positions[1], LAST_positions[2], LAST_positions[3] = INIT_PHI_1, INIT_PHI_2, INIT_PHI_3, INIT_PHI_4
    LAST_positions[4], LAST_positions[5], LAST_positions[6], LAST_positions[7] = INIT_THETA_1, INIT_THETA_2, INIT_THETA_3, INIT_THETA_4
    NOW_conditions = np.concatenate((LAST_positions, LAST_velocities))   # 必须有，DAE求解基本条件
    NEW_conditions = np.concatenate((LAST_positions, LAST_velocities))   # 必须有，DAE求解基本条件，但是第一次要使用
    LAST_O_STATE = np.zeros(8) # 必须有，PID基本条件

    # %% 主体组件
    Inertia_characteristics = [MCD_J_M_ZZ, J_W_XX, J_W_YY, J_W_ZZ, J_W_XY, J_W_XZ, J_W_YZ]
    def first_order_equations(t, y, *args):
        # 将y分解为位置和速度
        # %%状态转包
        positions, velocities = y[:8], y[8:]

        # %% 状态参数生成
        N_Stiff_TE_W1 = (
                SPRING_sigmoid_SHAPE(positions[4]) * (SPRING_sigmoid_NB(-velocities[0] * velocities[4]) + CONST_c))
        N_Stiff_TE_W2 = (
                SPRING_sigmoid_SHAPE(positions[5]) * (SPRING_sigmoid_NB(-velocities[1] * velocities[5]) + CONST_c))
        N_Stiff_TE_W3 = (
                SPRING_sigmoid_SHAPE(positions[6]) * (SPRING_sigmoid_NB(-velocities[2] * velocities[6]) + CONST_c))
        N_Stiff_TE_W4 = (
                SPRING_sigmoid_SHAPE(positions[7]) * (SPRING_sigmoid_NB(-velocities[3] * velocities[7]) + CONST_c))



        INST_ARG_ELASTIC = [N_Stiff_TE_W1, N_Stiff_TE_W2, N_Stiff_TE_W3, N_Stiff_TE_W4,
                            VALUE_Te_YAW_1, VALUE_Te_YAW_2, VALUE_Te_YAW_3, VALUE_Te_YAW_4]

        # %% 串列翼影响系数
        # Coefficient_fore_A, Coefficient_fore_B, Coefficient_back_A, Coefficient_back_B = FAST_inter(y)
        Coefficient_fore_A = 1
        Coefficient_fore_B = 1
        Coefficient_back_A = 1
        Coefficient_back_B = 1

        # # 气动载荷计算
        PHI_inst_0, PHI_b_dot_inst_0, theta_inst_0, theta_b_dot_inst_0 = positions[0], velocities[0], positions[4], \
            velocities[4]
        FX_TOL_1, FY_TOL_1, FZ_TOL_1, MX_TOL_1, MY_TOL_1, MZ_TOL_1, _, _, _ = QSM_W1.AERO_SOLVER_FLY(PHI_inst_0,
                                                                                                     PHI_b_dot_inst_0,
                                                                                                     theta_inst_0,
                                                                                                     theta_b_dot_inst_0,
                                                                                                     Coefficient_fore_A)

        PHI_inst_1, PHI_b_dot_inst_1, theta_inst_1, theta_b_dot_inst_1 = positions[1], velocities[1], positions[5], \
            velocities[5]
        FX_TOL_2, FY_TOL_2, FZ_TOL_2, MX_TOL_2, MY_TOL_2, MZ_TOL_2, _, _, _ = QSM_W2.AERO_SOLVER_FLY(PHI_inst_1,
                                                                                                     PHI_b_dot_inst_1,
                                                                                                     theta_inst_1,
                                                                                                     theta_b_dot_inst_1,
                                                                                                     Coefficient_fore_B)

        PHI_inst_2, PHI_b_dot_inst_2, theta_inst_2, theta_b_dot_inst_2 = positions[2], velocities[2], positions[6], \
            velocities[6]
        FX_TOL_3, FY_TOL_3, FZ_TOL_3, MX_TOL_3, MY_TOL_3, MZ_TOL_3, _, _, _ = QSM_W3.AERO_SOLVER_FLY(PHI_inst_2,
                                                                                                     PHI_b_dot_inst_2,
                                                                                                     theta_inst_2,
                                                                                                     theta_b_dot_inst_2,
                                                                                                     Coefficient_back_B)

        PHI_inst_3, PHI_b_dot_inst_3, theta_inst_3, theta_b_dot_inst_3 = positions[3], velocities[3], positions[7], \
            velocities[7]
        FX_TOL_4, FY_TOL_4, FZ_TOL_4, MX_TOL_4, MY_TOL_4, MZ_TOL_4, _, _, _ = QSM_W4.AERO_SOLVER_FLY(PHI_inst_3,
                                                                                                     PHI_b_dot_inst_3,
                                                                                                     theta_inst_3,
                                                                                                     theta_b_dot_inst_3,
                                                                                                     Coefficient_back_A)

        INST_ARG_AERO = [FX_TOL_1, FY_TOL_1, FZ_TOL_1, MX_TOL_1, MY_TOL_1, MZ_TOL_1,
                         FX_TOL_2, FY_TOL_2, FZ_TOL_2, MX_TOL_2, MY_TOL_2, MZ_TOL_2,
                         FX_TOL_3, FY_TOL_3, FZ_TOL_3, MX_TOL_3, MY_TOL_3, MZ_TOL_3,
                         FX_TOL_4, FY_TOL_4, FZ_TOL_4, MX_TOL_4, MY_TOL_4, MZ_TOL_4]

        # %% 输入量组合
        SOLVER_ARG = positions.tolist() + velocities.tolist() + INST_ARG_CONTROL + INST_ARG_AERO + INST_ARG_ELASTIC + Inertia_characteristics
        # 计算加速度（即二阶导数）
        accelerations = [
            NDE_phi_1(*SOLVER_ARG), NDE_phi_2(*SOLVER_ARG), NDE_phi_3(*SOLVER_ARG), NDE_phi_4(*SOLVER_ARG),NDE_theta_1(*SOLVER_ARG), NDE_theta_2(*SOLVER_ARG), NDE_theta_3(*SOLVER_ARG), NDE_theta_4(*SOLVER_ARG)]
        # 返回一阶导数（即速度和加速度）
        return np.concatenate((velocities, accelerations))

    # %% 多次迭代恒定值计算
    N_K_A1 = MCD_FRE_A ** 2 * 4 * np.pi ** 2 * CONST_J
    N_K_A2 = MCD_FRE_A ** 2 * 4 * np.pi ** 2 * CONST_J
    N_K_A3 = MCD_FRE_A ** 2 * 4 * np.pi ** 2 * CONST_J
    N_K_A4 = MCD_FRE_A ** 2 * 4 * np.pi ** 2 * CONST_J
    N_CONTROL_SAFW = 0.0
    N_CONTROL_SAHW = 0.0

    # %% 正式循环执行
    DATA = np.zeros((const_NUMB, 16))
    for i_step in range(const_NUMB):
    # for i_step in tqdm(range(const_NUMB)):
        HORIZON_LIST = EXP_s_SYS_r.HORIZON_CONTROL_ALLOCATION(MCD_FRE_A,
                                                              T_AM1 = np.radians(MCD_AM),
                                                              T_AM2 = np.radians(MCD_AM),
                                                              T_AM3 = np.radians(MCD_AM),
                                                              T_AM4 = np.radians(MCD_AM))

        # %% 期望指令，以及相应的前馈环节
        # 基于中枢模式振荡器轨迹生成（外部状态无关，主要给PID进行跟踪，MPC会在内部直接调用这个控制序列）
        Fur_ANGLE_1 = HORIZON_LIST[:, 0]
        Fur_ANGLE_2 = HORIZON_LIST[:, 1]
        Fur_ANGLE_3 = HORIZON_LIST[:, 2]
        Fur_ANGLE_4 = HORIZON_LIST[:, 3]

        # %% 控制指令生成
        """
        目前分成3个模式：
        1,1000HZ pid
        2,2000HZ pid
        3,1000HZ 混合RL
        """
        N_TORQUE_M_1, _, _, _, _ = pid_controller_5.update_ACT((LAST_O_STATE[4] - LAST_O_STATE[0]))  # 用两个状态
        N_TORQUE_M_2, _, _, _, _ = pid_controller_6.update_ACT((LAST_O_STATE[5] - LAST_O_STATE[1]))
        N_TORQUE_M_3, _, _, _, _ = pid_controller_7.update_ACT((LAST_O_STATE[6] - LAST_O_STATE[2]))
        N_TORQUE_M_4, _, _, _, _ = pid_controller_8.update_ACT((LAST_O_STATE[7] - LAST_O_STATE[3]))

        # %% DAE核心求解
        INST_ARG_CONTROL = [N_K_A1, N_K_A2, N_K_A3, N_K_A4, N_CONTROL_SAFW, N_CONTROL_SAHW, N_TORQUE_M_1, N_TORQUE_M_2, N_TORQUE_M_3, N_TORQUE_M_4]

        # %% 偏航控制过程
        t_start = i_step * const_DT
        t_end = (i_step + 1) * const_DT

        solution = solve_ivp(
            first_order_equations,
            t_span=(t_start, t_end),
            y0=NOW_conditions,
            method='Radau',
        )
        RESULT = solution.y
        NEW_conditions = RESULT[:, -1]

        NNN_positions, NNN_velocities = NEW_conditions[:8], NEW_conditions[8:]


        # %% 升力计算
        y = NEW_conditions
        positions, velocities = y[:8], y[8:]


        Coefficient_fore_A, Coefficient_fore_B, Coefficient_back_A, Coefficient_back_B = FAST_inter(y)

        # # 气动载荷计算
        PHI_inst_0, PHI_b_dot_inst_0, theta_inst_0, theta_b_dot_inst_0 = positions[0], velocities[0], positions[4], velocities[4]
        PHI_inst_1, PHI_b_dot_inst_1, theta_inst_1, theta_b_dot_inst_1 = positions[1], velocities[1], positions[5], velocities[5]
        PHI_inst_2, PHI_b_dot_inst_2, theta_inst_2, theta_b_dot_inst_2 = positions[2], velocities[2], positions[6], velocities[6]
        PHI_inst_3, PHI_b_dot_inst_3, theta_inst_3, theta_b_dot_inst_3 = positions[3], velocities[3], positions[7], velocities[7]



        _, _, _, _, _, _, LIFT1, _, _ = QSM_W1.AERO_SOLVER_FLY(PHI_inst_0,
                                                               PHI_b_dot_inst_0,
                                                               theta_inst_0,
                                                               theta_b_dot_inst_0,
                                                               Coefficient_fore_A)

        _, _, _, _, _, _, LIFT2, _, _ = QSM_W2.AERO_SOLVER_FLY(PHI_inst_1,
                                                               PHI_b_dot_inst_1,
                                                               theta_inst_1,
                                                               theta_b_dot_inst_1,
                                                               Coefficient_fore_B)

        _, _, _, _, _, _, LIFT3, _, _ = QSM_W3.AERO_SOLVER_FLY(PHI_inst_2,
                                                               PHI_b_dot_inst_2,
                                                               theta_inst_2,
                                                               theta_b_dot_inst_2,
                                                               Coefficient_back_B)

        _, _, _, _, _, _, LIFT4, _, _ = QSM_W4.AERO_SOLVER_FLY(PHI_inst_3,
                                                               PHI_b_dot_inst_3,
                                                               theta_inst_3,
                                                               theta_b_dot_inst_3,
                                                               Coefficient_back_A)

        # %% 结果统计[这里类似采用了滑动窗口法（手工记录多个时间步）+差分化（手工创造速度）]
        # 这个PID计算需要
        NEXT_state_ORIGIN = np.array([(NEW_conditions[0] - CENTER_SP_1),
                                      (NEW_conditions[1] - CENTER_SP_2),
                                      (NEW_conditions[2] - CENTER_SP_3),
                                      (NEW_conditions[3] - CENTER_SP_4),
                                      (Fur_ANGLE_1[1]),
                                      (Fur_ANGLE_2[1]),
                                      (Fur_ANGLE_3[1]),
                                      (Fur_ANGLE_4[1])])

        # %% 电机系统模型（电气属性计算）
        T_MOTOR_1 = N_TORQUE_M_1 / MCD_GEAR_RATIO / CONST_E_trans
        T_MOTOR_2 = N_TORQUE_M_2 / MCD_GEAR_RATIO / CONST_E_trans
        T_MOTOR_3 = N_TORQUE_M_3 / MCD_GEAR_RATIO / CONST_E_trans
        T_MOTOR_4 = N_TORQUE_M_4 / MCD_GEAR_RATIO / CONST_E_trans

        W_MOTOR_1 = NNN_velocities[0] * MCD_GEAR_RATIO
        W_MOTOR_2 = NNN_velocities[1] * MCD_GEAR_RATIO
        W_MOTOR_3 = NNN_velocities[2] * MCD_GEAR_RATIO
        W_MOTOR_4 = NNN_velocities[3] * MCD_GEAR_RATIO

        P_MOTOR_1, U_motor_1, I_motor_1 = ELE_MOTOR(W_MOTOR_1, T_MOTOR_1, R_motor, I_zero, KV_motor)
        P_MOTOR_2, U_motor_2, I_motor_2 = ELE_MOTOR(W_MOTOR_2, T_MOTOR_2, R_motor, I_zero, KV_motor)
        P_MOTOR_3, U_motor_3, I_motor_3 = ELE_MOTOR(W_MOTOR_3, T_MOTOR_3, R_motor, I_zero, KV_motor)
        P_MOTOR_4, U_motor_4, I_motor_4 = ELE_MOTOR(W_MOTOR_4, T_MOTOR_4, R_motor, I_zero, KV_motor)

        # %% 数据记录
        # 名称：当前时间步 ---功能：时间标签 ---备注：

        DATA[i_step, 0] = P_MOTOR_1
        DATA[i_step, 1] = P_MOTOR_2
        DATA[i_step, 2] = P_MOTOR_3
        DATA[i_step, 3] = P_MOTOR_4

        DATA[i_step, 4] = LIFT1 * SHIFT_N2g
        DATA[i_step, 5] = LIFT2 * SHIFT_N2g
        DATA[i_step, 6] = LIFT3 * SHIFT_N2g
        DATA[i_step, 7] = LIFT4 * SHIFT_N2g

        DATA[i_step, 8] = W_MOTOR_1
        DATA[i_step, 9] = W_MOTOR_2
        DATA[i_step, 10] = W_MOTOR_3
        DATA[i_step, 11] = W_MOTOR_4

        DATA[i_step, 12] = I_motor_1
        DATA[i_step, 13] = I_motor_2
        DATA[i_step, 14] = I_motor_3
        DATA[i_step, 15] = I_motor_4



        # %% DAE和指令系统值执行必要条件
        NOW_conditions = NEW_conditions  # 用来计算DAE，不能改变，上下次迭代
        LAST_O_STATE = NEXT_state_ORIGIN # 这个PID计算需要


    NEW_DATA = DATA[STAND_NEAB:,:] # 切分一下，


    # %% 功耗校验 I_max, W_max
    MAX_W_MOTOR_1 = np.max(NEW_DATA[:, 8])
    MAX_W_MOTOR_2 = np.max(NEW_DATA[:, 9])
    MAX_W_MOTOR_3 = np.max(NEW_DATA[:, 10])
    MAX_W_MOTOR_4 = np.max(NEW_DATA[:, 11])

    if MAX_W_MOTOR_1<W_max and MAX_W_MOTOR_2<W_max and MAX_W_MOTOR_3<W_max and MAX_W_MOTOR_4<W_max:
        MAX_I_MOTOR_1 = np.max(NEW_DATA[:, 12])
        MAX_I_MOTOR_2 = np.max(NEW_DATA[:, 13])
        MAX_I_MOTOR_3 = np.max(NEW_DATA[:, 14])
        MAX_I_MOTOR_4 = np.max(NEW_DATA[:, 15])
        if MAX_I_MOTOR_1 < I_max and MAX_I_MOTOR_2 < I_max and MAX_I_MOTOR_3 < I_max and MAX_I_MOTOR_4 < I_max:

            # %% 结果后处理
            MEAN_P1 = np.mean(NEW_DATA[:,0])
            MEAN_P2 = np.mean(NEW_DATA[:,1])
            MEAN_P3 = np.mean(NEW_DATA[:,2])
            MEAN_P4 = np.mean(NEW_DATA[:,3])
            # print("平均功率",(MEAN_P1+MEAN_P2+MEAN_P3+MEAN_P4)/4,"---",MEAN_P1,MEAN_P2,MEAN_P3,MEAN_P4)
            P_total = MEAN_P1 + MEAN_P2 + MEAN_P3 + MEAN_P4 + COUNT_P_etc
            # print("飞行器总功率",P_total)

            # %% 升力特性分析
            MEAN_L1 = np.mean(NEW_DATA[:,4])
            MEAN_L2 = np.mean(NEW_DATA[:,5])
            MEAN_L3 = np.mean(NEW_DATA[:,6])
            MEAN_L4 = np.mean(NEW_DATA[:,7])
            # print("平均升力",(MEAN_L1+MEAN_L2+MEAN_L3+MEAN_L4)/4,"---",MEAN_L1, MEAN_L2, MEAN_L3, MEAN_L4)

            TAKE_OFF = MEAN_L1+MEAN_L2+MEAN_L3+MEAN_L4
            PERCENT_MOTOR = (MASS_MOTOR*4)/TAKE_OFF
            # print("起飞重量",TAKE_OFF,"电机比例",PERCENT_MOTOR) # 0.5左右比较健康

            # %% 电池模型
            M_bat = TAKE_OFF * 0.2018
            E_bat = M_bat * D_bat
            E_bat = E_bat * 0.9 * 0.95  #
            T_endurance = E_bat / P_total
            if PERCENT_MOTOR<0.5:
                # print("悬停-飞行时间",T_endurance,"s",T_endurance/60,"min")

                # %% 循环迭代准备
                LAST_positions = [0.0] * 8
                LAST_velocities = [0.0] * 8
                LAST_positions[0], LAST_positions[1], LAST_positions[2], LAST_positions[3] = INIT_PHI_1, INIT_PHI_2, INIT_PHI_3, INIT_PHI_4
                LAST_positions[4], LAST_positions[5], LAST_positions[6], LAST_positions[7] = INIT_THETA_1, INIT_THETA_2, INIT_THETA_3, INIT_THETA_4
                NOW_conditions = np.concatenate((LAST_positions, LAST_velocities))   # 必须有，DAE求解基本条件
                NEW_conditions = np.concatenate((LAST_positions, LAST_velocities))   # 必须有，DAE求解基本条件，但是第一次要使用
                LAST_O_STATE = np.zeros(8) # 必须有，PID基本条件




                DATA = np.zeros((const_NUMB, 16))
                # start_time = time.time()
                for i_step in range(const_NUMB):
                # for i_step in tqdm(range(const_NUMB)):
                    # print("=" * 150)
                    # print("当前时间步：", i_step)
                    HORIZON_LIST = EXP_s_SYS_r_MAX_LIFT.HORIZON_CONTROL_ALLOCATION(MCD_FRE_A,
                                                                          T_AM1 = np.radians(MCD_AM_MAX_LIFT),
                                                                          T_AM2 = np.radians(MCD_AM_MAX_LIFT),
                                                                          T_AM3 = np.radians(MCD_AM_MAX_LIFT),
                                                                          T_AM4 = np.radians(MCD_AM_MAX_LIFT))

                    # 期望指令，以及相应的前馈环节
                    # 基于中枢模式振荡器轨迹生成（外部状态无关，主要给PID进行跟踪，MPC会在内部直接调用这个控制序列）
                    Fur_ANGLE_1 = HORIZON_LIST[:, 0]
                    Fur_ANGLE_2 = HORIZON_LIST[:, 1]
                    Fur_ANGLE_3 = HORIZON_LIST[:, 2]
                    Fur_ANGLE_4 = HORIZON_LIST[:, 3]

                    # 控制指令生成
                    """
                    目前分成3个模式：
                    1,1000HZ pid
                    2,2000HZ pid
                    3,1000HZ 混合RL
                    """
                    N_TORQUE_M_1, _, _, _, _ = pid_controller_5.update_ACT((LAST_O_STATE[4] - LAST_O_STATE[0]))  # 用两个状态
                    N_TORQUE_M_2, _, _, _, _ = pid_controller_6.update_ACT((LAST_O_STATE[5] - LAST_O_STATE[1]))
                    N_TORQUE_M_3, _, _, _, _ = pid_controller_7.update_ACT((LAST_O_STATE[6] - LAST_O_STATE[2]))
                    N_TORQUE_M_4, _, _, _, _ = pid_controller_8.update_ACT((LAST_O_STATE[7] - LAST_O_STATE[3]))



                    INST_ARG_CONTROL = [N_K_A1, N_K_A2, N_K_A3, N_K_A4, N_CONTROL_SAFW, N_CONTROL_SAHW, N_TORQUE_M_1, N_TORQUE_M_2, N_TORQUE_M_3, N_TORQUE_M_4]

                    # 偏航控制过程
                    t_start = i_step * const_DT
                    t_end = (i_step + 1) * const_DT

                    solution = solve_ivp(
                        first_order_equations,
                        t_span=(t_start, t_end),
                        y0=NOW_conditions,
                        method='Radau',
                    )
                    RESULT = solution.y
                    NEW_conditions = RESULT[:, -1]

                    NNN_positions, NNN_velocities = NEW_conditions[:8], NEW_conditions[8:]


                    # 升力计算
                    y = NEW_conditions
                    positions, velocities = y[:8], y[8:]


                    Coefficient_fore_A, Coefficient_fore_B, Coefficient_back_A, Coefficient_back_B = FAST_inter(y)

                    # # 气动载荷计算
                    PHI_inst_0, PHI_b_dot_inst_0, theta_inst_0, theta_b_dot_inst_0 = positions[0], velocities[0], positions[4], velocities[4]
                    PHI_inst_1, PHI_b_dot_inst_1, theta_inst_1, theta_b_dot_inst_1 = positions[1], velocities[1], positions[5], velocities[5]
                    PHI_inst_2, PHI_b_dot_inst_2, theta_inst_2, theta_b_dot_inst_2 = positions[2], velocities[2], positions[6], velocities[6]
                    PHI_inst_3, PHI_b_dot_inst_3, theta_inst_3, theta_b_dot_inst_3 = positions[3], velocities[3], positions[7], velocities[7]



                    _, _, _, _, _, _, LIFT1, _, _ = QSM_W1.AERO_SOLVER_FLY(PHI_inst_0,
                                                                           PHI_b_dot_inst_0,
                                                                           theta_inst_0,
                                                                           theta_b_dot_inst_0,
                                                                           Coefficient_fore_A)

                    _, _, _, _, _, _, LIFT2, _, _ = QSM_W2.AERO_SOLVER_FLY(PHI_inst_1,
                                                                           PHI_b_dot_inst_1,
                                                                           theta_inst_1,
                                                                           theta_b_dot_inst_1,
                                                                           Coefficient_fore_B)

                    _, _, _, _, _, _, LIFT3, _, _ = QSM_W3.AERO_SOLVER_FLY(PHI_inst_2,
                                                                           PHI_b_dot_inst_2,
                                                                           theta_inst_2,
                                                                           theta_b_dot_inst_2,
                                                                           Coefficient_back_B)

                    _, _, _, _, _, _, LIFT4, _, _ = QSM_W4.AERO_SOLVER_FLY(PHI_inst_3,
                                                                           PHI_b_dot_inst_3,
                                                                           theta_inst_3,
                                                                           theta_b_dot_inst_3,
                                                                           Coefficient_back_A)

                    # 结果统计[这里类似采用了滑动窗口法（手工记录多个时间步）+差分化（手工创造速度）]
                    # 这个PID计算需要
                    NEXT_state_ORIGIN = np.array([(NEW_conditions[0] - CENTER_SP_1),
                                                  (NEW_conditions[1] - CENTER_SP_2),
                                                  (NEW_conditions[2] - CENTER_SP_3),
                                                  (NEW_conditions[3] - CENTER_SP_4),
                                                  (Fur_ANGLE_1[1]),
                                                  (Fur_ANGLE_2[1]),
                                                  (Fur_ANGLE_3[1]),
                                                  (Fur_ANGLE_4[1])])

                    # 电机系统模型（电气属性计算）
                    T_MOTOR_1 = N_TORQUE_M_1 / MCD_GEAR_RATIO / CONST_E_trans
                    T_MOTOR_2 = N_TORQUE_M_2 / MCD_GEAR_RATIO / CONST_E_trans
                    T_MOTOR_3 = N_TORQUE_M_3 / MCD_GEAR_RATIO / CONST_E_trans
                    T_MOTOR_4 = N_TORQUE_M_4 / MCD_GEAR_RATIO / CONST_E_trans

                    W_MOTOR_1 = NNN_velocities[0] * MCD_GEAR_RATIO
                    W_MOTOR_2 = NNN_velocities[1] * MCD_GEAR_RATIO
                    W_MOTOR_3 = NNN_velocities[2] * MCD_GEAR_RATIO
                    W_MOTOR_4 = NNN_velocities[3] * MCD_GEAR_RATIO

                    P_MOTOR_1, U_motor_1, I_motor_1 = ELE_MOTOR(W_MOTOR_1, T_MOTOR_1, R_motor, I_zero, KV_motor)
                    P_MOTOR_2, U_motor_2, I_motor_2 = ELE_MOTOR(W_MOTOR_2, T_MOTOR_2, R_motor, I_zero, KV_motor)
                    P_MOTOR_3, U_motor_3, I_motor_3 = ELE_MOTOR(W_MOTOR_3, T_MOTOR_3, R_motor, I_zero, KV_motor)
                    P_MOTOR_4, U_motor_4, I_motor_4 = ELE_MOTOR(W_MOTOR_4, T_MOTOR_4, R_motor, I_zero, KV_motor)

                    DATA[i_step, 0] = P_MOTOR_1
                    DATA[i_step, 1] = P_MOTOR_2
                    DATA[i_step, 2] = P_MOTOR_3
                    DATA[i_step, 3] = P_MOTOR_4

                    DATA[i_step, 4] = LIFT1 * SHIFT_N2g
                    DATA[i_step, 5] = LIFT2 * SHIFT_N2g
                    DATA[i_step, 6] = LIFT3 * SHIFT_N2g
                    DATA[i_step, 7] = LIFT4 * SHIFT_N2g


                    DATA[i_step, 8] = W_MOTOR_1
                    DATA[i_step, 9] = W_MOTOR_2
                    DATA[i_step, 10] = W_MOTOR_3
                    DATA[i_step, 11] = W_MOTOR_4

                    DATA[i_step, 12] = I_motor_1
                    DATA[i_step, 13] = I_motor_2
                    DATA[i_step, 14] = I_motor_3
                    DATA[i_step, 15] = I_motor_4



                    # DAE和指令系统值执行必要条件
                    NOW_conditions = NEW_conditions  # 用来计算DAE，不能改变，上下次迭代
                    LAST_O_STATE = NEXT_state_ORIGIN # 这个PID计算需要

                NEW_DATA = DATA[STAND_NEAB:,:] # 切分一下，提升一下控制精度

                # %% 功耗校验 I_max, W_max
                MAX_W_MOTOR_1 = np.max(NEW_DATA[:, 8])
                MAX_W_MOTOR_2 = np.max(NEW_DATA[:, 9])
                MAX_W_MOTOR_3 = np.max(NEW_DATA[:, 10])
                MAX_W_MOTOR_4 = np.max(NEW_DATA[:, 11])

                if MAX_W_MOTOR_1 < W_max and MAX_W_MOTOR_2 < W_max and MAX_W_MOTOR_3 < W_max and MAX_W_MOTOR_4 < W_max:
                    MAX_I_MOTOR_1 = np.max(NEW_DATA[:, 12])
                    MAX_I_MOTOR_2 = np.max(NEW_DATA[:, 13])
                    MAX_I_MOTOR_3 = np.max(NEW_DATA[:, 14])
                    MAX_I_MOTOR_4 = np.max(NEW_DATA[:, 15])
                    if MAX_I_MOTOR_1 < I_max and MAX_I_MOTOR_2 < I_max and MAX_I_MOTOR_3 < I_max and MAX_I_MOTOR_4 < I_max:
                        # %% 最大升力计算
                        MEAN_L1_MAXLIIFT = np.mean(NEW_DATA[:,4])
                        MEAN_L2_MAXLIIFT = np.mean(NEW_DATA[:,5])
                        MEAN_L3_MAXLIIFT = np.mean(NEW_DATA[:,6])
                        MEAN_L4_MAXLIIFT = np.mean(NEW_DATA[:,7])

                        MAX_LIFT = MEAN_L1_MAXLIIFT + MEAN_L2_MAXLIIFT + MEAN_L3_MAXLIIFT + MEAN_L4_MAXLIIFT

                        # print("最大升力",MAX_LIFT)
                        BETA = np.arccos(TAKE_OFF/MAX_LIFT)
                        # print("飞行器倾斜角",BETA*57.1)
                        TAKE_OFF_res = TAKE_OFF * 1.2

                        if MAX_LIFT>TAKE_OFF_res:

                            THURST = (np.sqrt(MAX_LIFT**2 - TAKE_OFF_res**2) /1000)*9.8

                            # %% 功率特性
                            MEAN_P1_ML = np.mean(NEW_DATA[:,0])
                            MEAN_P2_ML = np.mean(NEW_DATA[:,1])
                            MEAN_P3_ML = np.mean(NEW_DATA[:,2])
                            MEAN_P4_ML = np.mean(NEW_DATA[:,3])

                            P_total_ML = MEAN_P1_ML + MEAN_P2_ML + MEAN_P3_ML + MEAN_P4_ML + COUNT_P_etc
                            # print("最大升力-飞行器总功率",P_total)
                            T_endurance_ML = E_bat/P_total_ML
                            # print("前飞-飞行时间",T_endurance_ML,"s",T_endurance_ML/60,"min")

                            # %% 迎风面积计算
                            S_wing = (c_tip + C_root) * (S_total - D_dc)
                            S = (MCD_GEAR_RATIO + 1) * Gear_Motor * M_gear * C_root * 2 + S_wing * 2# 本来应该除2的

                            # %% 前飞速度
                            S_real = S * np.cos(BETA)
                            CD = 1.3 # 参见《》
                            V = np.sqrt(2*THURST/(CD*1.225*S_real))
                            # print("前飞速度",V)

                            # %% 综合任务
                            Flight_time = D_total/V
                            # print("RUSH时间",Flight_time)
                            Engry_TWIN_rush = (Flight_time * 2)*P_total_ML
                            # print("RUSH能量",Engry_TWIN_rush,"能量百分比",Engry_TWIN_rush/E_bat*100,"%")

                            Engry_rest = E_bat - Engry_TWIN_rush
                            if Engry_rest > 0:
                                print("----------------------------------------LV0: 恭喜,正常")
                                REST_HOVER_s = Engry_rest/P_total
                                # print("RUSH后悬停时间",REST_HOVER_s,"s",REST_HOVER_s/60,"min")
                            else:
                                # print("警告:用于悬停的能量过少,应该大于0",Engry_rest)
                                print("----------------------------------------LV7: 用于悬停的能量过少")
                                REST_HOVER_s = 0
                                T_endurance = 0
                                V = -100
                                MAX_LIFT = -100
                                TAKE_OFF_res = -100
                                FUXX_FLAP = FUXX_FLAP + 1
                        else:
                            # print("警告:前飞可能性较低,应该大于1",MAX_LIFT/TAKE_OFF_res,MAX_LIFT,TAKE_OFF_res)
                            print("----------------------------------------LV6: 前飞升力余量过小")
                            REST_HOVER_s = -100
                            T_endurance = 0
                            V = -100
                            MAX_LIFT = -100
                            TAKE_OFF_res = -100
                            FUXX_FLAP = FUXX_FLAP + 1

                    else:
                        # 前飞电流检测不过
                        print("----------------------------------------LV5: 前飞电流检测不过")
                        REST_HOVER_s = -100
                        T_endurance = 0
                        V = -100
                        MAX_LIFT = -100
                        TAKE_OFF_res = -100
                        FUXX_FLAP = FUXX_FLAP + 1

                else:
                    # 前飞转速检测过不过
                    print("----------------------------------------LV4: 前飞转速检测过不过")
                    REST_HOVER_s = -100
                    T_endurance = 0
                    V = -100
                    MAX_LIFT = -100
                    TAKE_OFF_res = -100
                    FUXX_FLAP = FUXX_FLAP + 1

            else:
                print("----------------------------------------LV3: 电机比例过大")
                # print("警告:电机比例异常,应该小于0.5",PERCENT_MOTOR,MASS_MOTOR,TAKE_OFF)
                REST_HOVER_s = -100
                T_endurance = 0
                V = -100
                MAX_LIFT = -100
                TAKE_OFF_res = -100
                FUXX_FLAP = FUXX_FLAP + 1
        else:
            # 悬停电流检测不过
            print("----------------------------------------LV2: 悬停电流检测不过")
            REST_HOVER_s = -100
            T_endurance = 0
            V = -100
            MAX_LIFT = -100
            TAKE_OFF_res = -100
            FUXX_FLAP = FUXX_FLAP + 1
    else:
        # 悬停转速检测不过
        print("----------------------------------------LV1: 悬停转速检测不过")
        REST_HOVER_s = -100
        T_endurance = 0
        V = -100
        MAX_LIFT = -100
        TAKE_OFF_res = -100
        FUXX_FLAP = FUXX_FLAP + 1


    if IS_CHECK == True:
        # 创建pandas DataFrame
        df = pd.DataFrame(NEW_DATA)

        # 获取当前时间，并格式化为字符串作为文件名
        current_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        filename = f'CHECK_{current_time}.csv'

        # 将DataFrame保存到CSV文件
        df.to_csv(filename, index=False)

    EVAL_MCD = D_total
    EVAL_T_HOVER = T_endurance
    EVAL_T_MISSION = REST_HOVER_s
    EVAL_V_MAX = V

    return EVAL_MCD,EVAL_T_HOVER,EVAL_T_MISSION,EVAL_V_MAX,FUXX_FLAP