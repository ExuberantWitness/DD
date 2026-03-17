#!/usr/bin/env Python  # 必须加上，不然不能运行
# coding=utf-8 # 必须加上，不然不能运行

import numpy as np
import torch
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PID_interface_P_PINN_numpy:
    def __init__(self, Numb_parallel,kp, ki, kd, dt, SCALE):
        self.kp = np.ones(Numb_parallel) * kp
        self.ki = np.ones(Numb_parallel) * ki
        self.kd = np.ones(Numb_parallel) * kd
        self.dt = dt
        self.previous_error = np.zeros(Numb_parallel)
        self.integral = np.zeros(Numb_parallel)

        self.ORIGN_P = kp
        self.ORIGN_I = ki
        self.ORIGN_F = kd
        self.SCALE = SCALE
        self.Numb_parallel = Numb_parallel


        self.previous_error_LOCK = self.previous_error
        self.integral_LOCK = self.integral

    def SAVE_sim_STATE(self):
        self.previous_error_LOCK = self.previous_error
        self.integral_LOCK = self.integral


    def LOAD_sim_STATE(self):
        self.previous_error = self.previous_error_LOCK
        self.integral = self.integral_LOCK

    def update_ACT(self, error):
        derivative = (error - self.previous_error) / self.dt
        self.integral += error * self.dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.previous_error = error

        return output*self.SCALE


    def update_PID_outer(self,NEW_P,NEW_I,NEW_D):
        #这里就是直接拷贝现有的PID参数
        self.kp = self.ORIGN_P * ( np.ones(self.Numb_parallel) + NEW_P )
        self.ki = self.ORIGN_I * ( np.ones(self.Numb_parallel) + NEW_I )
        self.kd = self.ORIGN_F * ( np.ones(self.Numb_parallel) + NEW_D )



class PID_interface_P_PINN_torch:
    def __init__(self, Numb_parallel,kp, ki, kd, dt, SCALE):
        self.kp = (torch.ones(Numb_parallel) * kp)
        self.ki = (torch.ones(Numb_parallel) * ki)
        self.kd = (torch.ones(Numb_parallel) * kd)
        self.dt = dt
        self.previous_error = torch.zeros(Numb_parallel)
        self.integral = torch.zeros(Numb_parallel)

        self.ORIGN_P = kp
        self.ORIGN_I = ki
        self.ORIGN_F = kd
        self.SCALE = SCALE
        self.Numb_parallel = Numb_parallel

        self.previous_error_LOCK = self.previous_error
        self.integral_LOCK = self.integral

    def SAVE_sim_STATE(self):
        self.previous_error_LOCK = self.previous_error.clone()
        self.integral_LOCK = self.integral.clone()

    def LOAD_sim_STATE(self):
        self.previous_error = self.previous_error_LOCK.clone()
        self.integral = self.integral_LOCK.clone()

    def update_ACT(self, error):
        derivative = (error - self.previous_error) / self.dt
        self.integral += error * self.dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.previous_error = error

        return output*self.SCALE

    def update_PID_outer(self,NEW_P,NEW_I,NEW_D):
        #这里就是直接拷贝现有的PID参数
        self.kp = self.ORIGN_P * ( torch.ones(self.Numb_parallel) + NEW_P )
        self.ki = self.ORIGN_I * ( torch.ones(self.Numb_parallel) + NEW_I )
        self.kd = self.ORIGN_F * ( torch.ones(self.Numb_parallel) + NEW_D )


    def PACK_ACTION_POS(self,NEW_P,NEW_I,NEW_D,error):
        self.update_PID_outer(NEW_P,NEW_I,NEW_D)
        FORCE = self.update_ACT(error)
        return FORCE



