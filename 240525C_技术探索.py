#!/usr/bin/env Python  # 必须加上，不然不能运行
# coding=utf-8 # 必须加上，不然不能运行
# %% 核心组件
import time
from builtins import print
import numpy as np
from CORE_DDD_MFD_v11_MOO_simple import DDD_SYSTEM
import random
from numba import njit
from CORE_MCD import trajectory_generation_V5
from CORE_OPT_MCD_V2 import MCD_OPT_CORE
import time
from builtins import print
import numpy as np
from CORE_DDD_MFD_v11_MOO_simple import DDD_SYSTEM
import random
from numba import njit
from CORE_MCD import trajectory_generation_V5
from CORE_OPT_MCD_V6 import MCD_OPT_CORE
import pandas as pd
from datetime import datetime
import pandas as pd
from datetime import datetime
# %% 主要设计变量




MCD_AM = 60  # 随便给个误差


# %% 提前计算及恒定参数
dt = 0.0005
CENTER_SP_1, CENTER_SP_2, CENTER_SP_3, CENTER_SP_4 = 0.0, -np.pi, -np.pi, 0  # 坐标系鬼畜旋转

# %% 真实生物蜻蜓的-真实数据
# phi_am_1 = 60        # 扑动幅度-下限
# phi_am_2 = 90       # 扑动幅度-上限
# phi_md_1 = 0      # 中位-下限
# phi_md_2 = 0         # 中位-上限
# frequency_1 = 38.8     # 频率-下限
# frequency_2 = 41     # 频率-下限
# beta_FW_1 = 0       # 扑动平面角-下限
# beta_FW_2 = 0    # 扑动平面角-上限
# phase_D_FW_1 = 180  # 相位-下限
# phase_D_FW_2 = 180   # 相位-上限

R_ref = 1           # Unit: mm 翼展
NUMB = 2000
# 悬停阶段的仿生隐蔽性背景噪声计算
# Individual_Differences_HOVER,_ = GENE_Individual_Differences(R_ref, NUMB, dt,
#                                                                         phi_am_1, phi_am_2,
#                                                                         phi_md_1, phi_md_2,
#                                                                         frequency_1, frequency_2,
#                                                                         beta_FW_1, beta_FW_2,
#                                                                         phase_D_FW_1, phase_D_FW_2,
#                                                                         NUM_SPLIT = 10)
Individual_Differences_HOVER = 0.6268480922223615

# 中间悬停轨迹实现
phi_am = (60+90)/2
phi_md = 0
frequency = (38.8+41.2)/2
beta = 0
phase = 180
P_core_HOVER,_,_ = trajectory_generation_V5(phi_am, phi_md, frequency, beta, phase, R_ref, NUMB,dt)


# %% 串列翼干扰模型
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



MAX_W0 = 150  # 这个需要提前看
MAX_W1 = 150  # 这个需要提前看
MAX_W2 = 300  # 这个需要提前看
config_AM1 = MCD_AM
config_AM2 = MCD_AM
config_AM3 = MCD_AM
config_AM4 = MCD_AM
config_AM = (config_AM1 + config_AM2 + config_AM3 + config_AM4) / 4

@njit
def FAST_inter(y):
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

(NDE_phi_1,
 NDE_phi_2,
 NDE_phi_3,
 NDE_phi_4,
 NDE_theta_1,
 NDE_theta_2,
 NDE_theta_3,
 NDE_theta_4) = DDD_SYSTEM(CENTER_SP_1,
                           CENTER_SP_2,
                           CENTER_SP_3,
                           CENTER_SP_4)


# %% DENG
print("="*150)
print("DDD")

MCD_S_AR = 3.302     # 为[1,5]
MCD_TR = 0.40        # 为[0.25~1]
MCD_AM = 80
MCD_FRE_A = 34
MCD_SPAN = 0.075
ID_MOTOR = 4
MCD_GEAR_RATIO = 10

(EVAL_MCD, \
 EVAL_T_HOVER, \
 EVAL_T_MISSION, \
 EVAL_V_MAX, \
 FUXX_FLAP,P_mean) = MCD_OPT_CORE(MCD_AM, MCD_FRE_A, MCD_SPAN, ID_MOTOR, MCD_GEAR_RATIO, MCD_S_AR,
                                                  MCD_TR,
                                                  NDE_phi_1, NDE_phi_2, NDE_phi_3, NDE_phi_4, NDE_theta_1,
                                                  NDE_theta_2,
                                                  NDE_theta_3, NDE_theta_4,
                                                  P_core_HOVER, R_ref, Individual_Differences_HOVER, FAST_inter,
                                                  CENTER_SP_1, CENTER_SP_2, CENTER_SP_3, CENTER_SP_4,IS_CHECK = True)
print("期望升力",20.4/2)
print("我们计算的功率",P_mean)
P_stand = 13*0.47 -0.5
print("综合建模精度",(P_mean/P_stand-1)*100,"标准结果",P_stand)
