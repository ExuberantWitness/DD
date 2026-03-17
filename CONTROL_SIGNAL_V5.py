
from builtins import print
from datetime import datetime

import numpy as np
import pandas as pd
from datetime import datetime
import pandas as pd


class CPG_UNIT:
    def __init__(self, dt, CPG_am=50,
                 V_PP=0, V_P=0):
        self.CPG_m = 1
        self.CPG_c = CPG_am
        self.CPG_k = CPG_am ** 2 / 4
        self.CPG_F = CPG_am ** 2 * 1 / 4
        self.dt = dt

        self.reset(V_PP, V_P)
    def act(self, TARGAT, V_P, V_PP):
        OUT_PUT = (self.dt * self.dt * self.CPG_F * TARGAT - (self.dt * self.dt * self.CPG_k - 2 * self.CPG_m) * V_P - (self.CPG_m - 0.5 * self.dt * self.CPG_c) * V_PP) / (self.CPG_m + 0.5 * self.CPG_c * self.dt)

        return OUT_PUT

    def reset(self, V_PP=0, V_P=0):
        self.V_PP = V_PP
        self.V_P = V_P



class GPG_signal:
    def __init__(self,
                 Horizon,
                 CONST_Phase_different,
                 INIT_M1_PP,
                 INIT_M2_PP,
                 INIT_M3_PP,
                 INIT_M4_PP,
                 INIT_M1_P,
                 INIT_M2_P,
                 INIT_M3_P,
                 INIT_M4_P,
                 dt,
                 CPG_SCALE_RATIO=5,
                 ):

        # %% 基本参数设定

        self.CONST_Phase_different = CONST_Phase_different
        self.Horizon = Horizon + 2
        self.CPG_SCALE_RATIO = CPG_SCALE_RATIO
        CPG_am = 10 * self.CPG_SCALE_RATIO

        self.dt = dt

        # %% 中枢模式振荡器的相位轴
        self.Phase1 = 0
        self.Phase2 = 0
        self.Phase3 = self.CONST_Phase_different # 必须在这里进行提前加载
        self.Phase4 = self.CONST_Phase_different # 必须在这里进行提前加载


        # %% 振荡器核心初始化
        self.CPG_UNIT_M1 = CPG_UNIT(dt, CPG_am = CPG_am)
        self.CPG_UNIT_M2 = CPG_UNIT(dt, CPG_am = CPG_am)
        self.CPG_UNIT_M3 = CPG_UNIT(dt, CPG_am = CPG_am)
        self.CPG_UNIT_M4 = CPG_UNIT(dt, CPG_am = CPG_am)




        # %% 迭代量占位
        self.M1_PP = INIT_M1_PP
        self.M2_PP = INIT_M2_PP
        self.M3_PP = INIT_M3_PP
        self.M4_PP = INIT_M4_PP
        self.M1_P = INIT_M1_P
        self.M2_P = INIT_M2_P
        self.M3_P = INIT_M3_P
        self.M4_P = INIT_M4_P



    def CONTROL_ALLOCATION(self,
                           T_FRE, T_AM1,T_AM2,T_AM3,T_AM4,
                           Phase1,Phase2,Phase3,Phase4,M1_P,M2_P,M3_P,M4_P, M1_PP, M2_PP, M3_PP, M4_PP):

        # 1,相位轴推进
        D_phase1 = D_phase2 = D_phase3 = D_phase4 = self.dt * 2 * np.pi * T_FRE
        Phase1 = Phase1 + D_phase1
        Phase2 = Phase2 + D_phase2
        Phase3 = Phase3 + D_phase3
        Phase4 = Phase4 + D_phase4


        # 2,CPG执行,得到振荡幅度
        M1_CPG= self.CPG_UNIT_M1.act(T_AM1, M1_P,M1_PP)
        M2_CPG= self.CPG_UNIT_M1.act(T_AM2, M2_P,M2_PP)
        M3_CPG= self.CPG_UNIT_M1.act(T_AM3, M3_P,M3_PP)
        M4_CPG= self.CPG_UNIT_M1.act(T_AM4, M4_P,M4_PP)


        # 3,CPG结果组装
        M1 = M1_CPG * np.cos(Phase1)  # 这里是纯角度，单位：弧度
        M2 = M2_CPG * np.cos(Phase2)  # 这里是纯角度，单位：弧度
        M3 = M3_CPG * np.cos(Phase3)  # 这里是纯角度，单位：弧度
        M4 = M4_CPG * np.cos(Phase4)  # 这里是纯角度，单位：弧度


        return (M1,-M2,-M3,M4,
                Phase1,Phase2,Phase3,Phase4,
                M1_CPG,M2_CPG,M3_CPG,M4_CPG)

    def HORIZON_CONTROL_ALLOCATION(self,T_FRE, T_AM1,T_AM2,T_AM3,T_AM4):
        """
        这里是和外部交互的API

        :param T_FRE:
        :param T_AM1:
        :param T_AM2:
        :param T_AM3:
        :param T_AM4:
        :return:
        """
        # 1,规划前准备
        ARRAY_SIGNAL = np.zeros((self.Horizon,4))



        Phase1 = self.Phase1
        Phase2 = self.Phase2
        Phase3 = self.Phase3
        Phase4 = self.Phase4


        M1_P = self.M1_P
        M2_P = self.M2_P
        M3_P = self.M3_P
        M4_P = self.M4_P
        M1_PP = self.M1_PP
        M2_PP = self.M2_PP
        M3_PP = self.M3_PP
        M4_PP = self.M4_PP




        # 2,真实步进
        (ARRAY_SIGNAL[0, 0],ARRAY_SIGNAL[0, 1],ARRAY_SIGNAL[0, 2],ARRAY_SIGNAL[0, 3],Phase1,Phase2,Phase3,Phase4,
         M1_CPG,M2_CPG,M3_CPG,M4_CPG) = self.CONTROL_ALLOCATION(T_FRE, T_AM1, T_AM2, T_AM3, T_AM4,
                                                                Phase1, Phase2, Phase3, Phase4,
                                                                M1_P, M2_P, M3_P, M4_P,
                                                                M1_PP, M2_PP, M3_PP, M4_PP)

        M1_PP = M1_P
        M2_PP = M2_P
        M3_PP = M3_P
        M4_PP = M4_P

        M1_P = M1_CPG
        M2_P = M2_CPG
        M3_P = M3_CPG
        M4_P = M4_CPG




        self.Phase1 = Phase1 # 仅仅记录状态
        self.Phase2 = Phase2 # 仅仅记录状态
        self.Phase3 = Phase3 # 仅仅记录状态
        self.Phase4 = Phase4 # 仅仅记录状态
        self.M1_PP = self.M1_P # 仅仅记录状态
        self.M2_PP = self.M2_P # 仅仅记录状态
        self.M3_PP = self.M3_P # 仅仅记录状态
        self.M4_PP = self.M4_P # 仅仅记录状态
        self.M1_P = M1_CPG # 仅仅记录状态
        self.M2_P = M2_CPG # 仅仅记录状态
        self.M3_P = M3_CPG # 仅仅记录状态
        self.M4_P = M4_CPG # 仅仅记录状态



        # 3,虚拟递进模式（这里假设后续的期望轨迹形式不会发生变化，但是实际上动作可能会相差较大）
        for i in range(self.Horizon - 1):
            (ARRAY_SIGNAL[i+1, 0], ARRAY_SIGNAL[i+1, 1], ARRAY_SIGNAL[i+1, 2], ARRAY_SIGNAL[i+1, 3],
             Phase1, Phase2, Phase3, Phase4,
             M1_CPG, M2_CPG, M3_CPG, M4_CPG) = self.CONTROL_ALLOCATION(T_FRE, T_AM1,T_AM2,T_AM3,T_AM4,
                                                                       Phase1, Phase2, Phase3, Phase4,
                                                                       M1_P, M2_P, M3_P, M4_P,
                                                                       M1_PP, M2_PP, M3_PP, M4_PP)

            M1_PP = M1_P
            M2_PP = M2_P
            M3_PP = M3_P
            M4_PP = M4_P

            M1_P = M1_CPG
            M2_P = M2_CPG
            M3_P = M3_CPG
            M4_P = M4_CPG

        return ARRAY_SIGNAL
