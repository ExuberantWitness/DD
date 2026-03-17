import numpy as np
from numba import njit
from scipy.interpolate import interp1d
eps = np.finfo(np.float64).eps
import matplotlib.pyplot as plt
import pandas as pd
# import quaternionic
from CORE_FAST_QU_V2 import A_quaternion_shift_numba
from CORE_FAST_QU import quaternion_shift_numba

def FROM_DATE_TO_FUN(data_x,data_y,numb_blade=1000):
    """
    从翼根到翼尖

    :param data_x: 展向-位置-数据（起点和终点必须正确，尽量单调变化）
    :param data_y: 弦向-位置-数据
    :param numb_blade:片条数量
    :return:
    """
    # 创建一个插值函数，使用立方插值

    # print("y轴-数据格式",data_x.shape,"z轴-数据格式",data_y.shape)


    FUN = interp1d(data_x, data_y, kind='cubic')
    x_new = np.linspace(min(data_x), max(data_x), numb_blade)
    y_new = FUN(x_new)
    return FUN,x_new,y_new



def GENE_SHAPE_DATE_trapezoid(S_AR,TR,SPAN,num_blade,INSTALL_D = 0.005, IS_PLOT =True):
    """
    根据建立几何关系确定多关系

    :param S_AR: 单翼的展弦比,大于1，但是仅仅计算基本翼
    :param TR: 根梢比，小于1，仅仅计算基本翼
    :param SPAN: 基本翼翼展（不含INSTALL_D）
    :param num_blade:  当前离散量，可以和后买你不同
    :param INSTALL_D: 翼膜和翼根轴的偏差程度
    :return:
    """
    # 基本几何参数计算
    C_mean = (SPAN)/S_AR
    POS_Z_tip = - 2 * C_mean/(1 + 1/TR)
    POS_Z_root = POS_Z_tip/ TR

    # 几何计算关键坐标计算
    Y1 = INSTALL_D
    Z1 = POS_Z_root

    Y2 = SPAN + INSTALL_D
    Z2 = POS_Z_tip

    # 关键参数生成
    Data_R = np.linspace(Y1, Y2, num_blade)
    Data_LE = np.zeros(num_blade)
    Data_TE = np.zeros(num_blade)

    for i in range(num_blade):
        yi = Data_R[i]
        SLOPE = (Z2 - Z1)/(Y2 - Y1)
        Data_TE[i] = SLOPE*(yi - Y1) + Z1

    if IS_PLOT== True:
        plt.plot(Data_R, Data_LE, 'r')
        plt.plot(Data_R, Data_TE, 'r')
        plt.show()

    return Data_R,Data_LE,Data_TE



class AERO_QS_FLY():
    # 这个版本直接引入了对攻角本身的预先修正项
    def __init__(self,
                 num_blade,
                 Data_R,
                 Data_LE,
                 Data_TE,
                 PHIm = 60 * np.pi / 180,
                 f = 35,
                 v = 1.506e-5,
                 rho = 1.225,
                 CENTER = 0,
                 install_CENTER=0,
                 IS_PLOT=False,
                 ):
        """
        :param num_blade: 切分片条数量
        :param Data_R:  numpy.array   从翼根到翼展，指向y轴
        :param Data_LE: numpy.array   Z尽可能为正，
        :param Data_TE: numpy.array   要小于LE
        :param PHIm:
        :param rho:
        :param f:
        :param v:
        :param IS_PLOT:
        """

        # %% 几何预处理
        FUN_LE, R_new_LE, Z_new_LE = FROM_DATE_TO_FUN(Data_R, Data_LE,num_blade)
        FUN_TE, R_new_TE, Z_new_TE = FROM_DATE_TO_FUN(Data_R, Data_TE,num_blade)

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

        # %% 基本几何参数计算

        self.num_blade = num_blade
        self.r = R_new_LE
        self.del_R = min(Data_R)
        self.R = max(Data_R)
        # print("------最大翼长",self.R)
        self.B_base = max(Data_R) - min(Data_R)
        # print("------基准翼长", self.B_base)
        self.dr = abs(R_new_LE[0] - R_new_LE[1])  # 获得离散步长
        self.CENTER = CENTER # 处理对面翼的问题


        self.C = np.abs(Z_new_LE - Z_new_TE)





        self.CORE_F_rot2 = compute_F_rot2(Z_new_LE, Z_new_TE, self.r, rho, 1, self.dr)

        X_ROT = np.zeros(Z_new_LE.shape[0])



        # %% 累积变量
        self.S = np.sum(self.dr*self.C)
        # print("------面积：", self.S) # OK
        self.AR = self.B_base**2/self.S
        # print("------展弦比：", self.AR)
        self.r2 = BQS_compute_r2(R_new_LE, self.C, self.S, self.R)

        # print("----翼关键参数",self.r2)


        # print("------r2值：", self.r2)
        self.R2 = self.R * self.r2  # 计算有量纲面积二阶矩 # OK
        # print("------R2值：", self.R2)
        self.C2_R = BQS_trapz(R_new_LE, self.C ** 2 * R_new_LE)
        # print("------C2_R值：", self.C2_R)
        self.F_rot2 = compute_F_rot2(Z_new_LE, Z_new_TE, R_new_LE, rho, 1, self.dr)



        # %% 等效翼尖根弦长
        coefficients_LE = np.polyfit(Data_R, Data_LE, 1)
        slope, intercept = coefficients_LE # polyfit返回的系数，对于线性拟合，coefficients[0]是斜率，coefficients[1]是截距
        # 创建一个拟合线，用于绘图或计算
        C_ROOT_LE = slope * min(Data_R) + intercept
        C_TIP_LE = slope * max(Data_R) + intercept


        coefficients_TE = np.polyfit(Data_R, Data_TE, 1)
        slope, intercept = coefficients_TE # polyfit返回的系数，对于线性拟合，coefficients[0]是斜率，coefficients[1]是截距
        # 创建一个拟合线，用于绘图或计算
        C_ROOT_TE = slope * min(Data_R) + intercept
        C_TIP_TE = slope * max(Data_R) + intercept

        C_TIP = np.abs(C_TIP_LE - C_TIP_TE)
        C_ROOT = np.abs(C_ROOT_LE - C_ROOT_TE)
        self.C_mean = (C_TIP + C_ROOT)/2
        # print("------C_mean值：", self.C_mean)
        self.lamb = C_TIP / C_ROOT
        # print("------lamb值：", self.lamb)






        # %% 常量气动力参数
        self.rho = rho
        I = PHIm * 2 * 2  # OK
        self.Uref = self.R2 * I * f  # OK
        self.Re = self.C_mean * self.Uref / v  # 雷诺数 # OK
        self.Ro = self.R2 / self.C_mean  # OK

        # %% 虚拟压心(用来计算参考值)
        self.DUM_RC_R2_tr = 1.0768000587784088 * self.R2  # 主要用于计算旋转力和附加质量力
        self.DUM_XC_c_mean_tr = 0.38841422581129026 * self.C_mean  # 主要用于计算旋转力和附加质量力
        self.DUM_XC_c_mean_tr_ratio = 0.38841422581129026  # 主要用于计算每个片条上的平动力

        self.f_lma = 47.7 * self.lamb ** (-0.0019) - 46.7
        self.f_ARa = 1.294 - 0.590 * self.AR ** (-0.662)
        self.f_a = 0.776 + 1.911 * self.Re ** (-0.687)
        self.add_c = np.pi * self.rho * 0.25
        self.ADD1_trapz = BQS_trapz(R_new_LE, self.C ** 2 * R_new_LE)
        self.ADD2_trapz = BQS_trapz(R_new_LE, self.C ** 2 * (np.abs(Z_new_LE + Z_new_TE)/2 - X_ROT))



        self.f_r = 1.570 - -1.239*(1/self.R*BQS_trapz(R_new_LE,X_ROT)/self.C_mean);
        self.C_rot1 = 0.842 - 0.507 * self.Re ** (-0.1577)


        # %% 转动力和附加质量力的恒定压心
        self.P_S_cop_rot = 0.993 * self.R2
        self.P_C_cop_rot = 0.398 * self.C_mean
        self.P_S_cop_add = 1.078 * self.R2
        self.P_C_cop_add = 0.5 * self.C_mean

        # print("翼-准定长模型初始化完成")

    def MFD_Force_translation_Discrete_DICK(self,
                                            PHI_inst, PHI_b_dot_inst, theta_inst, theta_b_dot_inst,
                                            POS_x_B, POS_y_B, POS_z_B,
                                            theta_B_x, theta_B_y, theta_B_z,
                                            w_x_B, w_y_B, w_z_B,
                                            VX_ref_I, VY_ref_I, VZ_ref_I,
                                            INTERACTION_RATIO):
        """
        由于平动力主导了整个扑动过程，这里采用离散的方式计算片条上的力

        :param PHI_b_dot_inst:
        :param PHI_inst:
        :param theta_inst:
        :param theta_b_dot_inst:
        :param POS_x_B: 翼坐标店在机体坐标系下的位置（用来计算机体产生的附加速度）
        :param POS_y_B: 翼坐标店在机体坐标系下的位置（用来计算机体产生的附加速度）
        :param POS_z_B: 翼坐标店在机体坐标系下的位置（用来计算机体产生的附加速度）
        :param theta_B_x:
        :param theta_B_y:
        :param theta_B_z:
        :param w_x_B:
        :param w_y_B:
        :param w_z_B:
        :param VX_ref_I:
        :param VY_ref_I:
        :param VZ_ref_I:
        :param INTERACTION_RATIO:
        :return:
        """

        r_out = self.r[1:]
        r_in = self.r[:-1]

        C_out = self.C[1:]
        C_in = self.C[:-1]






        SUM_W_FX_tr, \
            SUM_W_FY_tr, \
            SUM_W_FZ_tr, \
            SUM_W_MX_tr, \
            SUM_W_MY_tr, \
            SUM_W_MZ_tr, \
            SUM_B_L_tr, \
            SUM_B_D_tr = MFD_Force_translation_Discrete_DICK_out(self.Re,
                                                                 self.AR,
                                                                 self.Ro,
                                                                 self.rho,
                                                                 self.num_blade,
                                                                 r_in, r_out,
                                                                 C_in, C_out,
                                                                 self.DUM_XC_c_mean_tr_ratio,
                                                                 self.dr,
                                            PHI_inst, PHI_b_dot_inst, theta_inst, theta_b_dot_inst,
                                            POS_x_B, POS_y_B, POS_z_B,
                                            theta_B_x, theta_B_y, theta_B_z,
                                            w_x_B, w_y_B, w_z_B,
                                            VX_ref_I, VY_ref_I, VZ_ref_I,
                                            INTERACTION_RATIO)

        return SUM_W_FX_tr, SUM_W_FY_tr, SUM_W_FZ_tr, SUM_W_MX_tr, SUM_W_MY_tr, SUM_W_MZ_tr, SUM_B_L_tr, SUM_B_D_tr

    def MFD_Force_rotation_REDICK(self, theta_inst,theta_b_dot_inst,PHI_b_dot_inst,INTERACTION_RATIO):
        # Re = self.Re
        # rho = self.rho
        #
        # f_alpha = 1 # np.sqrt(2) * np.cos(AOA)

        # f_r = 1.570
        # C_rot1 = 0.842 - 0.507 * Re ** (-0.1577)

        # F_rot1 = f_alpha * f_r * C_rot1 * rho * PHI_b_dot_inst * theta_b_dot_inst * self.C2_R
        F_rot2 = - self.CORE_F_rot2*theta_b_dot_inst**2
        # 这里获取法向力
        # FN_rot = F_rot1[0] + F_rot2
        FX_rot = F_rot2*INTERACTION_RATIO  # 加上 F_rot1[0] 会让现象非常奇怪



        # 生成升力和阻力
        FL_rot, FD_rot = GENE_FLD(FX_rot, theta_inst)


        W_FX_rot = FX_rot
        W_FY_rot = 0.0
        W_FZ_rot = 0.0


        # 转矩计算
        W_MX_rot, W_MY_rot, W_MZ_rot = FAST_TWM(W_FX_rot, W_FZ_rot, self.P_S_cop_rot, self.P_C_cop_rot)

        return W_FX_rot, W_FY_rot, W_FZ_rot, W_MX_rot, W_MY_rot, W_MZ_rot, FL_rot, FD_rot


    def AERO_SOLVER_FLY(self,
                        PHI_inst,PHI_b_dot_inst,
                        theta_inst,theta_b_dot_inst,
                        POS_x_B = 0,
                        POS_y_B = 0,
                        POS_z_B = 0 ,
                        theta_B_x = 0,
                        theta_B_y = 0,
                        theta_B_z = 0,
                        w_x_B = 0,
                        w_y_B = 0,
                        w_z_B = 0,
                        VX_ref_I = 0,
                        VY_ref_I = 0,
                        VZ_ref_I = 0,
                        INTERACTION_RATIO = 1):
        """
        :param PHI_inst: 扑动角
        :param PHI_b_dot_inst:

        :param theta_inst:  扑动角
        :param theta_b_dot_inst:

        :param POS_x_B: 翼根位置在机体坐标系的相对位置 x
        :param POS_y_B: 翼根位置在机体坐标系的相对位置 y
        :param POS_z_B: 翼根位置在机体坐标系的相对位置 z

        :param theta_B_x: 机体坐标下x角度
        :param theta_B_y: 机体坐标下y角度
        :param theta_B_z: 机体坐标下z角度

        :param w_x_B: 机体坐标下x角速度
        :param w_y_B: 机体坐标下y角速度
        :param w_z_B: 机体坐标下z角速度
        :param VX_ref_I: 机体，在惯性系下的速度
        :param VY_ref_I: 机体，在惯性系下的速度
        :param VZ_ref_I: 机体，在惯性系下的速度

        :param INTERACTION_RATIO: 串列翼修正系数
        :return:
        """
        # --------------------------------------------------------------------------------------------------------------
        # 平动力计算（这里使用每个片条分别计算的载荷）
        FX_W_tr_DICK, FY_W_tr_DICK, FZ_W_tr_DICK, \
            MX_W_tr_DICK, MY_W_tr_DICK, MZ_W_tr_DICK, \
            FL_B_tr_DICK, FD_B_tr_DICK = self.MFD_Force_translation_Discrete_DICK(PHI_inst,
                                                                                  PHI_b_dot_inst,
                                                                                  theta_inst,
                                                                                  theta_b_dot_inst,
                                                                                  POS_x_B, POS_y_B, POS_z_B,
                                                                                  theta_B_x, theta_B_y, theta_B_z,
                                                                                  w_x_B, w_y_B, w_z_B,
                                                                                  VX_ref_I, VY_ref_I, VZ_ref_I,
                                                                                  INTERACTION_RATIO)

        # --------------------------------------------------------------------------------------------------------------
        # 转动力计算
        FX_W_rot, FY_W_rot, FZ_W_rot, \
            MX_W_rot, MY_W_rot, MZ_W_rot, \
            FL_B_rot, FD_B_rot = self.MFD_Force_rotation_REDICK(theta_inst,
                                                                theta_b_dot_inst,
                                                                PHI_b_dot_inst,
                                                                INTERACTION_RATIO)


        FX_W_TOL = FX_W_tr_DICK + FX_W_rot
        FY_W_TOL = FY_W_tr_DICK + FY_W_rot
        FZ_W_TOL = FZ_W_tr_DICK + FZ_W_rot



        F_W_TOL = np.array([FX_W_TOL,FY_W_TOL,FZ_W_TOL])
        DRY_NP = R_xyz_NP(theta_x=0.0, theta_y=-theta_inst, theta_z = - PHI_inst + self.CENTER) # TMD一定加负号
        B_B_TOL = np.dot(DRY_NP.T, F_W_TOL)
        FX_B_TOL = B_B_TOL[0]
        FY_B_TOL = B_B_TOL[1]
        FZ_B_TOL = B_B_TOL[2]

        MX_W_TOL = MX_W_tr_DICK + MX_W_rot # + MX_add
        MY_W_TOL = MY_W_tr_DICK + MY_W_rot # + MY_add
        MZ_W_TOL = MZ_W_tr_DICK + MZ_W_rot # + MZ_add

        FL_B_TOL = FL_B_tr_DICK + FL_B_rot # + FL_add
        FD_B_TOL = FD_B_tr_DICK + FD_B_rot # + FD_add

        # --------------------------------------------------------------------------------------------------------------
        # 可视化环节
        IS_OB = True
        if IS_OB ==True:
            V_w_ref = GET_SPEED_ANYPOINT(theta_inst, PHI_inst,
                                         theta_b_dot_inst, PHI_b_dot_inst, self.DUM_XC_c_mean_tr, self.DUM_RC_R2_tr,
                                         POS_x_B, POS_y_B, POS_z_B,
                                         theta_B_x, theta_B_y, theta_B_z,
                                         w_x_B, w_y_B, w_z_B,
                                         VX_ref_I, VY_ref_I, VZ_ref_I)

            AOA, V2_inst = SSFAST_AOA(V_w_ref)
        else:
            AOA = 0.0
        return FX_B_TOL,FY_B_TOL,FZ_B_TOL,\
               MX_W_TOL,MY_W_TOL,MZ_W_TOL,\
               FL_B_TOL,FD_B_TOL,AOA, \
               # FX_W_tr_DICK, FY_W_tr_DICK, FZ_W_tr_DICK, \
               # MX_W_tr_DICK, MY_W_tr_DICK, MZ_W_tr_DICK, \
               # FL_B_tr_DICK, FD_B_tr_DICK, \
               # FX_W_rot, FY_W_rot, FZ_W_rot, \
               # MX_W_rot, MY_W_rot, MZ_W_rot, \
               # FL_B_rot, FD_B_rot



# %% 外挂加速
@njit
def MFD_Force_translation_NT_DICKINSON(S_AOA, V2_inst, dS, Re, AR, Ro, rho):
    """
    参见：DESIGN AND CONTROL OF A HUMMINGBIRD-SIZE FLAPPING WING MICRO AERIAL VEHICLE

    :param AOA:
    :param V2:
    :param dS:
    :param Re:
    :param AR:
    :param Ro:
    :param rho:
    :return:
    """
    CN_tr = 3.4*np.sin(S_AOA)
    CT_tr = 0.4*np.cos(2*S_AOA)**2

    F_W_z_tr = + CT_tr * 0.5 * rho * V2_inst * dS # T是Z轴正方向
    F_W_x_tr = - CN_tr * 0.5 * rho * V2_inst * dS # N_tr是X轴反方向

    return F_W_z_tr, F_W_x_tr

@njit
def MFD_Force_translation_Discrete_DICK_out(Re,AR,Ro,rho,num_blade,
                                            r_in,r_out,
                                            C_in,C_out,
                                            DUM_XC_c_mean_tr_ratio,dr,
                                            PHI_inst, PHI_b_dot_inst, theta_inst, theta_b_dot_inst,
                                            POS_x_B, POS_y_B, POS_z_B,
                                            theta_B_x, theta_B_y, theta_B_z,
                                            w_x_B, w_y_B, w_z_B,
                                            VX_ref_I, VY_ref_I, VZ_ref_I,
                                            INTERACTION_RATIO):
    """
    由于平动力主导了整个扑动过程，这里采用离散的方式计算片条上的力

    :param PHI_b_dot_inst:
    :param PHI_inst:
    :param theta_inst:
    :param theta_b_dot_inst:
    :param POS_x_B: 翼坐标店在机体坐标系下的位置（用来计算机体产生的附加速度）
    :param POS_y_B: 翼坐标店在机体坐标系下的位置（用来计算机体产生的附加速度）
    :param POS_z_B: 翼坐标店在机体坐标系下的位置（用来计算机体产生的附加速度）
    :param theta_B_x:
    :param theta_B_y:
    :param theta_B_z:
    :param w_x_B:
    :param w_y_B:
    :param w_z_B:
    :param VX_ref_I:
    :param VY_ref_I:
    :param VZ_ref_I:
    :param INTERACTION_RATIO:
    :return:
    """
    A_T_s_cop = (r_in+r_out)/2
    A_T_c_cop = (C_in+C_out)/2 * DUM_XC_c_mean_tr_ratio

    V_w_ref = A_GET_SPEED_ANYPOINT(theta_inst, PHI_inst,
                                   theta_b_dot_inst, PHI_b_dot_inst, A_T_c_cop, A_T_s_cop,
                                   POS_x_B, POS_y_B, POS_z_B,
                                   theta_B_x, theta_B_y, theta_B_z,
                                   w_x_B, w_y_B, w_z_B,
                                   VX_ref_I, VY_ref_I, VZ_ref_I)
    A_S_AOA, A_V2_inst = A_SSFAST_AOA(V_w_ref)  # print(i,"片条状态",S_AOA,V2_inst)

    A_dS = dr * (C_in + C_out) / 2
    FT_tr, FN_tr = MFD_Force_translation_NT_DICKINSON(A_S_AOA, A_V2_inst, A_dS, Re, AR, Ro, rho)
    FT_tr = FT_tr * INTERACTION_RATIO
    FN_tr = FN_tr * INTERACTION_RATIO

    S_MX_tr, S_MY_tr, S_MZ_tr = FAST_TWM(FN_tr, FT_tr, A_T_s_cop, A_T_c_cop)

    AR_B_L_tr = FT_tr * np.cos(theta_inst) - FN_tr * np.sin(theta_inst)
    AR_B_D_tr = FT_tr * np.sin(theta_inst) + FN_tr * np.cos(theta_inst)

    SUM_B_L_tr = np.sum(AR_B_L_tr)
    SUM_B_D_tr = np.sum(AR_B_D_tr)

    SUM_W_FX_tr = np.sum(FN_tr)
    SUM_W_FY_tr = 0.0
    SUM_W_FZ_tr = np.sum(FT_tr)
    SUM_W_MX_tr = np.sum(S_MX_tr)
    SUM_W_MY_tr = np.sum(S_MY_tr)
    SUM_W_MZ_tr = np.sum(S_MZ_tr)
    return SUM_W_FX_tr, SUM_W_FY_tr, SUM_W_FZ_tr, SUM_W_MX_tr, SUM_W_MY_tr, SUM_W_MZ_tr, SUM_B_L_tr, SUM_B_D_tr




def compute_F_rot2(xLE, xTE, r, rho, alpha_dot, dr):
    F_rot2 = 0
    for j in range(len(xLE)):
        x = np.linspace(xLE[j], xTE[j], 101)
        integrand = r[j] * x * np.abs(x)
        integral = np.trapz(integrand, x)
        F_rot2 += 2.67 * rho * alpha_dot ** 2 * integral * dr / 2
    return F_rot2


@njit
def BQS_compute_r2(r, c, S_base, R_Par):
    integrand = c[:] * r[:] * r[:]
    integral = np.trapz(integrand, r[:])
    r2 = np.sqrt(integral / (S_base * R_Par ** 2))
    return r2


def BQS_trapz(x, y, dim=None):
    if y.size == 0:
        z = np.zeros(1, dtype=y.dtype)
        return z

    if np.isscalar(x):
        z = x * np.sum((y[:-1] + y[1:]), axis=0) / 2
    else:
        z = np.diff(x, axis=0).T @ (y[:-1] + y[1:]) / 2

    siz = y.shape
    siz = (1,) * y.ndim + siz[1:]
    z = np.reshape(z, siz)

    if dim is not None and not np.isscalar(z):
        z = np.moveaxis(z, 0, dim)
    return z


@njit
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

    return np.dot(np.dot(Rz, Ry), Rx)

@njit
def FAST_TWM(FBX, FBZ, P_S_cop, P_C_cop):
    MX = FBZ * P_S_cop
    MY = - FBX * P_C_cop
    MZ = - FBX * P_S_cop
    return MX, MY, MZ








# @njit
def SSFAST_AOA(V_w_ref):
    # 直接简单定义（即使用cos来搞一下，也会遇到零的情况）
    if abs(V_w_ref[2])<10*eps:
        SIGN = np.sign(V_w_ref[0])
        AOA = SIGN*np.pi/2
        V2_inst = V_w_ref[2] ** 2
    else:
        AOA = np.arctan(V_w_ref[0] / V_w_ref[2])
        V2_inst = V_w_ref[2] ** 2 + V_w_ref[0] ** 2

    return AOA, V2_inst


@njit
def A_SSFAST_AOA(V_w_ref):
    A_AOA = np.arctan(V_w_ref[:, 0] / V_w_ref[:, 2])
    A_V2_inst = V_w_ref[:, 2]**2 + V_w_ref[:, 0]**2

    # 找到V_w_ref[:, 2]的绝对值小于10*eps的索引
    condition_indices = np.where(np.abs(V_w_ref[:, 2]) < 10*eps)[0]

    # 更新A_AOA和A_V2_inst在特定条件下的值
    A_AOA[condition_indices] = np.sign(V_w_ref[condition_indices, 0]) * np.pi / 2
    A_V2_inst[condition_indices] = V_w_ref[condition_indices, 2] ** 2

    return A_AOA, A_V2_inst








@njit
def FUNC_V_ABPW_DUMMY_REFINE(PHI_b_dot_inst, theta_inst, theta_b_dot_inst, L_SP, L_CH):
    v_x_w = - L_CH * theta_b_dot_inst - L_SP*np.cos(theta_inst)*PHI_b_dot_inst
    v_y_w = - L_CH * np.sin(theta_inst) * PHI_b_dot_inst
    v_z_w = - L_SP * np.sin(theta_inst) * PHI_b_dot_inst
    return np.array([v_x_w,v_y_w,v_z_w])


@njit
def GENE_FLD(FX_,theta_inst):
    # 生成升力和阻力（这个不要动，绑定了FX_rot）,就是有负号
    FL_ = -(FX_) * np.sin(theta_inst)
    FD_ = -(FX_) * np.cos(theta_inst)
    return FL_,FD_


@njit # 不可以，因为用了四元数库
def A_GET_SPEED_ANYPOINT(theta_inst,PHI_inst,
                       theta_dot_inst,PHI_dot_inst,A_Value_ch,A_Value_sp,
                       POS_x_B, POS_y_B, POS_z_B,
                       theta_B_x,theta_B_y,theta_B_z,
                       w_x_B, w_y_B, w_z_B,
                       VX_ref_I, VY_ref_I, VZ_ref_I):
    """
    计算翼面坐标系上任意点在惯性系下的速度分量
    :param theta_inst:
    :param PHI_inst:
    :param theta_dot_inst:
    :param PHI_dot_inst:

    :param Value_ch:
    :param Value_sp:

    :param POS_x_B:
    :param POS_y_B:
    :param POS_z_B:

    :param theta_B_x:
    :param theta_B_y:
    :param theta_B_z:

    :param w_x_B:
    :param w_y_B:
    :param w_z_B:
    :param VX_ref_I:
    :param VY_ref_I:
    :param VZ_ref_I:
    :return:
    """

    LEN = A_Value_ch.shape[0]


    A_V_theta_W = np.zeros((LEN,3))
    A_V_theta_W[:,0] = theta_dot_inst*A_Value_ch
    A_V_phi_A = np.zeros((LEN,3))
    A_V_phi_A[:, 0] = PHI_dot_inst*A_Value_sp

    ZERO_ANGLE = np.zeros(LEN)

    # 分量1
    POS_B = np.array([POS_x_B, POS_y_B, POS_z_B])
    angular_v= np.array([w_x_B, w_y_B, w_z_B])
    V_outer_1_B = np.cross(angular_v, POS_B)
    A_V_outer_1_B = FAST_TILE(V_outer_1_B,LEN)
    A_V_outer_1_W = A_quaternion_shift_numba(A_V_outer_1_B, ZERO_ANGLE, theta_inst, PHI_inst,LEN = LEN)

    # 分量2
    V_outer_2_I = np.array([VX_ref_I, VY_ref_I, VZ_ref_I])
    A_V_outer_2_I = FAST_TILE(V_outer_2_I, LEN)
    A_V_outer_2_B = A_quaternion_shift_numba(A_V_outer_2_I,theta_B_x,theta_B_y,theta_B_z,LEN = LEN)
    A_V_outer_2_W = A_quaternion_shift_numba(A_V_outer_2_B,ZERO_ANGLE,theta_inst,PHI_inst,LEN = LEN)

    # 分量3
    A_V_phi_W = A_quaternion_shift_numba(A_V_phi_A,ZERO_ANGLE,theta_inst,ZERO_ANGLE,LEN = LEN)

    # 翼面坐标系总速度
    A_V_tol_W = A_V_theta_W + A_V_phi_W + A_V_outer_1_W + A_V_outer_2_W
    return A_V_tol_W



@njit
def FAST_TILE(vector,LEN):
    result = np.ones((LEN,3))
    result[:, 0] = result[:,0] * vector[0]
    result[:, 1] = result[:,1] * vector[1]
    result[:, 2] = result[:,2] * vector[2]
    return result



@njit # 不可以，因为用了四元数库
def GET_SPEED_ANYPOINT(theta_inst,PHI_inst,
                       theta_dot_inst,PHI_dot_inst,Value_ch,Value_sp,
                       POS_x_B, POS_y_B, POS_z_B,
                       theta_B_x,theta_B_y,theta_B_z,
                       w_x_B, w_y_B, w_z_B,
                       VX_ref_I, VY_ref_I, VZ_ref_I):
    """
    计算翼面坐标系上任意点在惯性系下的速度分量
    :param theta_inst:
    :param PHI_inst:
    :param theta_dot_inst:
    :param PHI_dot_inst:
    :param Value_ch:
    :param Value_sp:
    :param POS_x_B:
    :param POS_y_B:
    :param POS_z_B:
    :param theta_B_x:
    :param theta_B_y:
    :param theta_B_z:
    :param w_x_B:
    :param w_y_B:
    :param w_z_B:
    :param VX_ref_I:
    :param VY_ref_I:
    :param VZ_ref_I:
    :return:
    """

    # 生成每个速度
    V_theta_W = np.array([theta_dot_inst*Value_ch,0.0,0.0])

    V_phi_A = np.array([PHI_dot_inst*Value_sp,0.0,0.0])

    POS_B = np.array([POS_x_B, POS_y_B, POS_z_B])
    angular_v= np.array([w_x_B, w_y_B, w_z_B])
    V_outer_1_B = np.cross(angular_v, POS_B)

    V_outer_2_I = np.array([VX_ref_I, VY_ref_I, VZ_ref_I])

    # 外部速度1
    V_outer_2_B = quaternion_shift_numba(V_outer_2_I,theta_B_x,theta_B_y,theta_B_z)
    V_outer_2_W = quaternion_shift_numba(V_outer_2_B,0.0,theta_inst,PHI_inst)

    # 外部速度2
    V_outer_1_W = quaternion_shift_numba(V_outer_1_B,0.0,theta_inst,PHI_inst)

    # 驱动器速度
    V_phi_W = quaternion_shift_numba(V_phi_A,0.0,theta_inst,0.0)

    # 翼面坐标系总速度
    V_tol_W = V_theta_W + V_phi_W + V_outer_1_W + V_outer_2_W
    return V_tol_W



# %% 校验专用
def IS_unique(TIMELINE_CL):
    """
    功能：数据处理-主要用于验证模型有效性过程的

    :param TIMELINE_CL:
    :return:
    """
    # 假设 TIMELINE_CL 是一个numpy数组
    unique_values, indices, counts = np.unique(TIMELINE_CL, return_index=True, return_counts=True)
    # 找出重复的值
    duplicate_indices = indices[counts > 1]
    duplicate_values = unique_values[counts > 1]

    FLAG = True

    LIST_index = []

    if duplicate_indices.size > 0:
        print("------存在重复数据:")
        for value, index in zip(duplicate_values, duplicate_indices):
            # print(f"------值 {value} 在索引 {index} 处重复出现。")
            LIST_index.append(index)
        FLAG = False
    else:
        print("------没有重复数据。")

    return FLAG,LIST_index


def CLEAN_getdate_INPUT(FILE_NAME_input,Uref,S,rho,T,t_values,QS_REF,FILE_NAME_output,
                        IS_OUT_DATA = True,IS_PLOT_DATA = True, IS_CL = True):
    """
    这个主要用在校验过程，没有


    :param FILE_NAME: 要处理的文件名字
    :param Uref: 动压-参考速度
    :param S: 动压-参考面积
    :param rho: 密度
    :param T: 周期时间
    :param t_values: 需要插值的位置
    :param QS_REF: 准定常模型计算结果
    :return:
    """

    DYNAMICS_Q = 0.5 * rho * Uref ** 2 * S

    # 使用 numpy 的 loadtxt 函数读取空格分隔的数据
    CL_array = np.loadtxt(FILE_NAME_input)

    O_TIMELINE_CL = CL_array[:, 0]
    O_TIMELINE_CL[0] = 0
    TIMELINE_CL = O_TIMELINE_CL * (T / (max(O_TIMELINE_CL) - min(O_TIMELINE_CL)))
    O_CL_DATA = CL_array[:, 1] * DYNAMICS_Q



    FLAG, LIST_index = IS_unique(TIMELINE_CL)
    print("存在问题的ID",LIST_index)

    if FLAG== True:
        print("数据是干净的")
    else:
        for i in range(len(LIST_index)):
            ID_1 = LIST_index[0]
            TIMELINE_CL = np.delete(TIMELINE_CL, ID_1)
            O_CL_DATA = np.delete(O_CL_DATA, ID_1)

    FLAG, LIST_index = IS_unique(TIMELINE_CL)
    print("数据是否清理干净",FLAG)


    FUN_exp_CL = interp1d(TIMELINE_CL, O_CL_DATA, kind='linear')
    EXP_CL_NEW = FUN_exp_CL(t_values)

    DATA_OUT_CX = np.zeros((t_values.shape[0], 5))
    DATA_OUT_CX[:, 0] = t_values / T
    DATA_OUT_CX[:, 1] = np.abs(QS_REF)
    DATA_OUT_CX[:, 2] = EXP_CL_NEW

    if IS_PLOT_DATA == True:
        plt.plot(t_values / T, QS_REF)
        plt.plot(t_values / T, EXP_CL_NEW)
        plt.show()

    if IS_OUT_DATA==True:
        # 将numpy数组转换为pandas DataFrame
        df = pd.DataFrame(DATA_OUT_CX)
        # 将DataFrame保存到Excel文件
        df.to_excel(FILE_NAME_output, index=False, header=False)



    if IS_CL==False:
        QS_REF = np.abs(QS_REF)

    MAX_QS_CX = np.max(QS_REF)
    MAX_EXP_CX = np.max(EXP_CL_NEW)
    MAX_PERCENT_ERROR_CX = np.abs((MAX_QS_CX - MAX_EXP_CX) / MAX_EXP_CX * 100)

    MEAN_QS_CX = np.mean(QS_REF)
    MEAN_EXP_CX = np.mean(EXP_CL_NEW)
    MEAN_PERCENT_ERROR_CX = np.abs((MEAN_QS_CX - MEAN_EXP_CX) / MEAN_EXP_CX * 100)

    correlation_matrix = np.corrcoef(QS_REF, EXP_CL_NEW)
    TREND_CX = correlation_matrix[0, 1]

    return DATA_OUT_CX, MAX_QS_CX, MAX_EXP_CX,MAX_PERCENT_ERROR_CX, MEAN_QS_CX, MEAN_EXP_CX, MEAN_PERCENT_ERROR_CX, TREND_CX
