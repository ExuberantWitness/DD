import numpy as np
from numba import njit
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from datetime import datetime






# %% 个体差异生成
def GENE_Individual_Differences(R_ref, NUMB, dt,
                                phi_am_1, phi_am_2,
                                phi_md_1, phi_md_2,
                                frequency_1, frequency_2,
                                beta_FW_1, beta_FW_2,
                                phase_D_FW_1, phase_D_FW_2,
                                NUM_SPLIT = 10):




    # 计算得到参考轨迹
    phi_am_refe_FW = (phi_am_2 + phi_am_1) / 2
    phi_md_refe_FW = (phi_md_2 + phi_md_1) / 2
    frequency_refe_FW = (frequency_2 + frequency_1) / 2
    beta_refe_FW = (beta_FW_2 + beta_FW_1) / 2
    phase_D_FW_refe = (phase_D_FW_2 + phase_D_FW_1) / 2

    P_core, PHI, TIME = trajectory_generation_V5(phi_am_refe_FW, phi_md_refe_FW, frequency_refe_FW, beta_refe_FW,
                                                 phase_D_FW_refe, R_ref, NUMB, dt)


    # 批量生成（这里注重发现最大值值，因此采用线性关系）
    LIST_phi_am = np.linspace(phi_am_1, phi_am_2, NUM_SPLIT)
    LIST_phi_md = np.linspace(phi_md_1, phi_md_2, NUM_SPLIT)
    LIST_frequency = np.linspace(frequency_1, frequency_2, NUM_SPLIT)
    LIST_beta = np.linspace(beta_FW_1, beta_FW_2, NUM_SPLIT)
    LIST_phase = np.linspace(phase_D_FW_1, phase_D_FW_2, NUM_SPLIT)


    print("正在解算个体仿生隐蔽性，耗时大致2min")
    DATA = np.zeros((NUM_SPLIT, NUM_SPLIT, NUM_SPLIT, NUM_SPLIT, NUM_SPLIT))
    for i in range(NUM_SPLIT):
        for j in range(NUM_SPLIT):
            for k in range(NUM_SPLIT):
                for l in range(NUM_SPLIT):
                    for m in range(NUM_SPLIT):
                        # print("位置", i, j, k, l, m)
                        TEMP_phi_am = LIST_phi_am[i]
                        TEMP_phi_md = LIST_phi_md[j]
                        TEMP_frequency = LIST_frequency[k]
                        TEMP_beta = LIST_beta[l]
                        TEMP_phase = LIST_phase[m]

                        POS, PHI, TIME = trajectory_generation_V5(TEMP_phi_am, TEMP_phi_md, TEMP_frequency, TEMP_beta,
                                                               TEMP_phase, R_ref, NUMB, dt)
                        DATA[i, j, k, l, m] = frechet3D_V4(POS, P_core) / R_ref

    Individual_Differences = np.max(DATA)
    print("个体差异", Individual_Differences)
    return Individual_Differences,P_core







def GENE_ARRAY_am_fre_beta_C_i_DYN(P_core,Individual_Differences,R_ref, NUMB, dt,
                                   NUM_SPLIT_am,phi_am_1A,phi_am_2A,
                                   NUM_SPLIT_FRE,FRE_1A,FRE_2A):

    LIST_phi_aircraft_am = np.linspace(phi_am_1A, phi_am_2A, NUM_SPLIT_am)
    LIST_aircraft_fre = np.linspace(FRE_1A, FRE_2A, NUM_SPLIT_FRE)
    DATA = np.zeros((NUM_SPLIT_am, NUM_SPLIT_FRE))

    TEMP_phase = 180
    TEMP_phi_md = 0
    TEMP_beta = 0

    for i in range(NUM_SPLIT_am):
        for j in range(NUM_SPLIT_FRE):


            TEMP_phi_am = LIST_phi_aircraft_am[i]
            TEMP_frequency = LIST_aircraft_fre[j]


            POS, PHI, TIME = trajectory_generation_V5(TEMP_phi_am, TEMP_phi_md, TEMP_frequency, TEMP_beta, TEMP_phase, R_ref, NUMB, dt)
            TEMP_MCD = frechet3D_V4(POS, P_core) / R_ref


            if TEMP_MCD < Individual_Differences:
                TEMP_MCD = 0
            else:
                TEMP_MCD = TEMP_MCD - Individual_Differences

            DATA[i, j] = TEMP_MCD

    return DATA




# %% 核心计算程序
@njit
def trajectory_generation_V5(phi_am, phi_md, frequency, beta, phase, R, num_points,dt):
    """

    :param phi_am: 扑动幅度，度
    :param phi_md: 扑动中位，度
    :param frequency: 频率，Hz
    :param beta: 扑动平面角，度，0为水平面
    :param phase: 相差差，度
    :param R: 单翼长，m（虽然可以无量纲化）
    :param num_points: 评估的总点的数量
    :param dt: 时间步长
    :return:
    """
    TIME = np.arange(0, num_points*dt, dt)
    PHI = np.radians(phi_md + phi_am / 2 * np.sin(2 * np.pi * frequency * TIME + np.deg2rad(phase)))
    POS = np.zeros((num_points,3))
    POS_out = np.zeros((num_points, 3))
    POS[:, 0] = -R * np.sin(PHI)
    POS[:, 1] = R * np.cos(PHI)

    POS_out[:, 0] = POS[:, 0] * np.cos(beta)
    POS_out[:, 2] = -POS[:, 0] * np.sin(beta)
    POS_out[:, 1] = POS[:, 1]

    return POS_out, PHI, TIME

@njit
def frechet3D_V4(P1, P2):
    DIFF = P1 - P2
    NORMAL = np.sqrt(np.sum(DIFF**2,axis=1))
    return np.mean(NORMAL)




# %% 可视化处理
def Trajectory_3D_PLOT(POS):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制P_core轨迹
    ax.plot(POS[:, 0], POS[:, 1], POS[:, 2], label='P_core', color='b')

    # 设置图例
    ax.legend()

    # 设置坐标轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])

    # 设置标题
    ax.set_title('3D Trajectories of P_core and P_temp')

    # 显示图形
    plt.show()

def Trajectory_2D_PLOT(TIME, PHI):
    plt.plot(TIME, PHI)
    plt.title('Angle')
    plt.legend()

    # 显示图表
    plt.show()

def Trajectory_save_excel(POS):
    df = pd.DataFrame(POS)

    # 获取当前时间，并格式化为字符串作为文件名
    current_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    filename = f'MCD-ARRAY_{current_time}.csv'

    # 将DataFrame保存到CSV文件
    df.to_csv(filename, index=False)



def MCD_STATIC_SHIFT(S_air):
    S_aim = 30
    C_eys = 6.82/10000


    if S_air<S_aim:
        DIST = 0
    else:
        DIST = (S_air - S_aim) / C_eys


    return DIST




def MCD_GENERAL_SHIFT(S_air,S_aim = 30):
    """

    :param S_air: 飞行器翼展，单位mm
    :param S_aim:
    :return: 输出的为m
    """
    C_eys = 6.82/10000

    if S_air<S_aim:
        DIST = 0.0
        # print("MCD GENERAL", 0,"m，飞行器翼展小于现有生物体积")
    else:
        DIST = (S_air - S_aim) / C_eys /1000
        # print("MCD GENERAL", DIST, "m")
    return DIST



def COMPARE_AIRCRAFT(SPAN_aircraft,phi_am_aircraft,phi_md_aircraft,fre_aircraft,P_core,NUMB,dt,Individual_Differences = 0):
    R_ref = 1
    P_aircraft, _, _ = trajectory_generation_V5(phi_am_aircraft,
                                                phi_md_aircraft,
                                                fre_aircraft,
                                                0,
                                                180, R_ref, NUMB, dt)

    C_dym = frechet3D_V4(P_core, P_aircraft) / R_ref

    if C_dym < Individual_Differences:
        C_dym = 0
    else:
        C_dym = C_dym - Individual_Differences

    D_STATIC = MCD_GENERAL_SHIFT(SPAN_aircraft)
    D_DYM = MCD_GENERAL_SHIFT(C_dym * SPAN_aircraft, S_aim=0)
    D_total = D_STATIC + D_DYM
    return D_total
