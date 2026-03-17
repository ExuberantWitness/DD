import pandas as pd
import numpy as np
from numba import njit

ID_MOTOR = 0     # 目前初始导入电机0的数据

def GET_MOTOR_DATA_V2(ID_MOTOR):
    # 读取Excel文件
    df = pd.read_excel('MOTOR_LIST_240521.xlsx')

    # 转换成numpy array
    MOTOR_array = df.to_numpy()
    Numb_motor = MOTOR_array.shape[0]
    # print("导入电机数量", Numb_motor)

    # 关键数据索引，及单位转换
    # 电机电压
    U_motor = MOTOR_array[ID_MOTOR, 2]  # 单位：V

    # 空载电流
    I_zero = MOTOR_array[ID_MOTOR, 4] / 1000  # 单位：A

    # 堵转电流
    I_max = MOTOR_array[ID_MOTOR, 9]  # 单位：A

    # 电机电阻
    R_motor = MOTOR_array[ID_MOTOR, 11]  # 单位：O

    # 电机KV值
    KV_motor = MOTOR_array[ID_MOTOR, 14]  # 单位：O

    # 电机最大转速(基于轴承)
    W_max = MOTOR_array[ID_MOTOR, 18] * 2 * np.pi / 60  # 单位：rad/s


    # 电机重量
    MASS_MOTOR = MOTOR_array[ID_MOTOR, 19]  # 单位：G

    # 电机惯量
    INT_MOTOR = MOTOR_array[ID_MOTOR, 17]






    return U_motor, I_zero, I_max, R_motor, KV_motor, W_max,MASS_MOTOR,INT_MOTOR

@njit
def ELE_MOTOR(W_MOTOR, T_MOTOR,R_motor,I_zero,KV_motor):
    W_MOTOR_RPM = W_MOTOR * 60/(2*np.pi)
    R_motor = R_motor/np.sqrt(3)



    T_MOTOR = np.abs(T_MOTOR)
    W_MOTOR_RPM = np.abs(W_MOTOR_RPM)
    Kt = 30/(np.pi * KV_motor)
    I_motor = I_zero + T_MOTOR/Kt
    U_motor = R_motor * I_motor + W_MOTOR_RPM / KV_motor
    P_MOTOR = U_motor * I_motor
    return P_MOTOR, U_motor, I_motor

def NEST_INT(num):
    return round(num)


U_motor, I_zero, I_max, R_motor, KV_motor, W_max,MASS_MOTOR,INT_MOTOR = GET_MOTOR_DATA_V2(0)


