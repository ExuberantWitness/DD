#!/usr/bin/env Python  # 必须加上，不然不能运行
# coding=utf-8 # 必须加上，不然不能运行



import numpy as np
import torch
from numba import njit
from torch import nn
torch.set_default_dtype(torch.float64)
torch.set_printoptions(precision=10)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


CRIT_VALUE = 1000
CRIT_VALUE1 = 0/57.1


INIT_SHIFT = 2

INIT_P_SCALE_1 = 0.00001 + 1
INIT_P_SCALE_2 = 0.00001 + 1

class PID_interface:
    def __init__(self, kp, ki, kd, dt):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.previous_error = 0
        self.integral = 0
        self.derivative = 0

        self.ORIGN_P = kp
        self.ORIGN_I = ki
        self.ORIGN_F = kd


    def GET_ERORR(self,error):

        # if abs(error)>CRIT_VALUE1/ 57.1:
        min_val = -CRIT_VALUE / 57.1
        max_val = CRIT_VALUE / 57.1
        error = max(min(error, max_val), min_val)


        derivative = (error - self.previous_error) / self.dt




        integral = self.integral + error * self.dt
        return error,integral,derivative


    def update_ACT(self, error):
        min_val = -CRIT_VALUE / 57.1
        max_val = CRIT_VALUE / 57.1
        error = max(min(error, max_val), min_val)


        derivative = (error - self.previous_error) / self.dt
        self.integral += error * self.dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.previous_error = error

        P_error = error
        I_error = self.integral
        D_error = derivative



        self.derivative = derivative

        if abs(error)<CRIT_VALUE1:
            output = 0
            FLAG = False
        else:
            FLAG = True
        return output, P_error, I_error, D_error,FLAG


    def update_PID_outer(self,NEW_P,NEW_I,NEW_D):
        #这里就是直接拷贝现有的PID参数


        self.kp = self.ORIGN_P*(elu(NEW_P) + INIT_P_SCALE_1 + 0.2)# 不要一下子拉成0》有PI，PD，没有ID控制
        self.ki = self.ORIGN_I*(elu(NEW_I) + INIT_P_SCALE_2)
        self.kd = self.ORIGN_F*(elu(NEW_D) + INIT_P_SCALE_2)




class PID_interface_P_PINN_torch(nn.Module):
    def __init__(self,kp, ki, kd, dt):
        super(PID_interface_P_PINN_torch, self).__init__()
        Numb_parallel = 4
        self.kp = (torch.ones(Numb_parallel) * kp).to(device)
        self.ki = (torch.ones(Numb_parallel) * ki).to(device)
        self.kd = (torch.ones(Numb_parallel) * kd).to(device)
        self.dt = dt
        self.previous_error = torch.zeros(Numb_parallel).to(device)
        self.integral = torch.zeros(Numb_parallel).to(device)

        self.ORIGN_P = kp
        self.ORIGN_I = ki
        self.ORIGN_F = kd
        self.Numb_parallel = Numb_parallel
        self.ONE_LIST = torch.ones(self.Numb_parallel).to(device)


    def Synchronize_error(self,TENSOR_previous_error,TENSOR_integral):
        self.previous_error = TENSOR_previous_error
        self.integral = TENSOR_integral

    def GET_ERORR(self,error):
        min_val = -CRIT_VALUE / 57.1
        max_val = CRIT_VALUE / 57.1
        error = error.clamp(min_val, max_val)

        derivative = (error - self.previous_error) / self.dt
        integral = self.integral + error * self.dt
        return error,integral,derivative

    def update_ACT(self, error):
        min_val = -CRIT_VALUE / 57.1
        max_val = CRIT_VALUE / 57.1
        error = error.clamp(min_val, max_val)

        derivative = (error - self.previous_error) / self.dt
        self.integral += error * self.dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.previous_error = error

        return output

    def update_PID_outer(self,NEW_P,NEW_I,NEW_D):
        #这里就是直接拷贝现有的PID参数

        self.kp = self.ORIGN_P * (torch.relu(NEW_P) + self.ONE_LIST * INIT_P_SCALE_1 + self.ONE_LIST*0.2)   #不要一下子拉成0》有PI，PD，没有ID控制
        self.ki = self.ORIGN_I * (torch.relu(NEW_I) + self.ONE_LIST * INIT_P_SCALE_2)
        self.kd = self.ORIGN_F * (torch.relu(NEW_D) + self.ONE_LIST * INIT_P_SCALE_2)
    def PACK_ACTION_POS(self,NEW_P,NEW_I,NEW_D,error):
        self.update_PID_outer(NEW_P,NEW_I,NEW_D)
        FORCE = self.update_ACT(error)
        return FORCE

@njit
def elu(x, alpha=1.0):
    return np.where(x >= 0, x, alpha * (np.exp(x) - 1))





