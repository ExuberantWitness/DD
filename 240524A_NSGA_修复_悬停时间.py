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
import time
from builtins import print
import numpy as np
from CORE_DDD_MFD_v11_MOO_simple import DDD_SYSTEM
import random
from numba import njit
from CORE_MCD import trajectory_generation_V5
from CORE_OPT_MCD_V3 import MCD_OPT_CORE
import pandas as pd
from datetime import datetime
# %% 主要设计变量
MCD_AM = 80          # 为[10~85]
MCD_FRE_A = 34       # 为[5,65]
MCD_SPAN = 0.075     # 为[0.005,0.015]
MCD_S_AR = 3.302     # 为[1,5]
MCD_TR = 0.40        # 为[0.25~1]
ID_MOTOR = 3         # 为 Choice，其取值为0，~20的整数
MCD_GEAR_RATIO = 25  # 取值范围为[5,35]

# %% 提前计算及恒定参数
dt = 0.0005
CENTER_SP_1, CENTER_SP_2, CENTER_SP_3, CENTER_SP_4 = 0.0, -np.pi, -np.pi, 0  # 坐标系鬼畜旋转

# %% 真实生物蜻蜓的-真实数据
print("MCD前置计算")
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


# %% pymoo
import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.algorithms.moo.sms import SMSEMOA
# from pymoo.algorithms.moo.unsga3 import UNSGA3
# from pymoo.termination import get_termination
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.algorithms.moo.rvea import RVEA
from pymoo.algorithms.moo.ctaea import CTAEA
from pymoo.core.population import Population, Individual


NUM_TARGET = 2

# 定义优化问题
class MyProblem(ElementwiseProblem):
    def __init__(self):
        super().__init__(n_var=5,
                         n_obj=2,
                         n_constr=0,
                         xl=np.array([10, 15, 0.05, -0.5, 5]),
                         xu=np.array([85, 50, 0.12, 20.49, 35]))
    def _evaluate(self, x, out, *args, **kwargs):
        MCD_AM, MCD_FRE_A, MCD_SPAN, ID_MOTOR, MCD_GEAR_RATIO = x



        # 示例函数，您需要定义具体的 MCD_OPT_CORE 函数
        EVAL_MCD, EVAL_T_HOVER, EVAL_T_MISSION, EVAL_V_MAX, FUXX_FLAP = MCD_OPT_CORE(
            MCD_AM, MCD_FRE_A, MCD_SPAN, ID_MOTOR, MCD_GEAR_RATIO,
            MCD_S_AR, MCD_TR, NDE_phi_1, NDE_phi_2, NDE_phi_3, NDE_phi_4,
            NDE_theta_1, NDE_theta_2, NDE_theta_3, NDE_theta_4, P_core_HOVER,
            R_ref, Individual_Differences_HOVER, FAST_inter, CENTER_SP_1,
            CENTER_SP_2, CENTER_SP_3, CENTER_SP_4
        )

        if FUXX_FLAP > 0:
            SCALE_MOTOR = 0.00000000001
        else:
            SCALE_MOTOR = 1.0

        EVAL_MCD = EVAL_MCD  # 这个不进行缩放,没有必要
        EVAL_T_HOVER = EVAL_T_HOVER * SCALE_MOTOR

        print("--------违约", FUXX_FLAP, "EVAL_MCD: {:.3f}:".format(EVAL_MCD),
              "EVAL_T_HOVER: {:.3f}".format(EVAL_T_HOVER), "MCD_AM: {:.3f}".format(MCD_AM),
              "MCD_FRE_A: {:.3f}".format(MCD_FRE_A), "MCD_SPAN: {:.3f}".format(MCD_SPAN),
              "ID_MOTOR: {:.3f}".format(ID_MOTOR), "MCD_GEAR_RATIO: {:.3f}".format(MCD_GEAR_RATIO))
        # 目标固件（最小化）
        out["F"] = [EVAL_MCD, -EVAL_T_HOVER]

# 定义问题实例
problem = MyProblem()


# # 定义参考方向
ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=100) # 这个大好,可以让前沿面更加致密
# 定义算法



algorithm = SMSEMOA(pop_size=200)
# algorithm.setup(problem, pop=initial_population)  不可用



# 捕获回调
class CaptureCallback:
    def __init__(self):
        self.data = []

    def __call__(self, algorithm):
        for ind in algorithm.pop:
            self.data.append((ind.get("X"), ind.get("F")))

capture_callback = CaptureCallback()

# 执行优化
res = minimize(problem,
               algorithm,
               termination=('n_gen', 60), # 这个大好,可以让前沿面更加致密
               seed=1,
               verbose=True,
               callback=capture_callback,
               save_history=True,
               copy_algorithm=False,
               # initial_population=initial_population 不可work
               )



# %% 所有数据记录，及保存
evaluated_X = [x for x, _ in capture_callback.data]
evaluated_F = [f for _, f in capture_callback.data]
import matplotlib.pylab as plt
evaluated_F = np.array(evaluated_F)
plt.scatter(evaluated_F[:,0],evaluated_F[:,1])
plt.xlabel("MCD")
plt.ylabel("hover_time")
plt.show()
df = pd.DataFrame(evaluated_F)
# 获取当前时间，并格式化为字符串作为文件名
current_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
filename = f'TOTAL_evaluated_X_{current_time}.csv'
# 将DataFrame保存到CSV文件
df.to_csv(filename, index=False)



DESIGN_VALUE = np.zeros((len(evaluated_X),5))
for i in range(len(evaluated_X)):
    DESIGN_VALUE[i, 0] = evaluated_X[i][0]
    DESIGN_VALUE[i, 1] = evaluated_X[i][1]
    DESIGN_VALUE[i, 2] = evaluated_X[i][2]
    DESIGN_VALUE[i, 3] = evaluated_X[i][3]
    DESIGN_VALUE[i, 4] = evaluated_X[i][4]
df = pd.DataFrame(DESIGN_VALUE)

# 获取当前时间，并格式化为字符串作为文件名
current_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
filename = f'DESIGN_VALUE_{current_time}.csv'

# 将DataFrame保存到CSV文件
df.to_csv(filename, index=False)




# %% 后处理:标准计算结果
LEN_result = res.X.shape[0]


ARRAR_OPT_TARGET = np.zeros((LEN_result,NUM_TARGET))
ARRAR_OPT_PARAM = np.zeros((LEN_result,5))


for i in range(LEN_result):
    TEMP_TARGET = res.F[i]
    ARRAR_OPT_TARGET[i, 0] = TEMP_TARGET[0]
    ARRAR_OPT_TARGET[i, 1] = TEMP_TARGET[1]

    ARRAR_OPT_PARAM[i, 0] = res.X[i][0]
    ARRAR_OPT_PARAM[i, 1] = res.X[i][1]
    ARRAR_OPT_PARAM[i, 2] = res.X[i][2]
    ARRAR_OPT_PARAM[i, 3] = res.X[i][3]
    ARRAR_OPT_PARAM[i, 4] = res.X[i][4]

# 结果输出
df = pd.DataFrame(ARRAR_OPT_TARGET)

# 获取当前时间，并格式化为字符串作为文件名
current_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
filename = f'ARRAR_OPT_TARGET_{current_time}.csv'

# 将DataFrame保存到CSV文件
df.to_csv(filename, index=False)

df = pd.DataFrame(ARRAR_OPT_PARAM)

# 获取当前时间，并格式化为字符串作为文件名
current_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
filename = f'ARRAR_OPT_PARAM_{current_time}.csv'

# 将DataFrame保存到CSV文件
df.to_csv(filename, index=False)


import matplotlib.pylab as plt


plt.scatter(ARRAR_OPT_TARGET[:,0],ARRAR_OPT_TARGET[:,1])
plt.xlabel("MCD")
plt.ylabel("hover_time")
plt.show()
