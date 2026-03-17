import numpy as np

class PIDNN:
    """
        Adaptive PID Neural Network Controller
    """
    def __init__(self, P_VALUE, I_VALUE, D_VALUE, learning_rate, tolerance, timestep, OB_STEP = 2):
        """

        :param initial_constants:
        :param learning_rate:
        :param max_weight_change:
        :param tolerance:
        :param timestep:
        """

        # 网络学习率
        self.eta = learning_rate
        # 误差容差
        self.tol = tolerance


        # 除零容差
        self.div_by_zero_tol = 1e-20
        # 时间步长
        self.timestep = timestep

        # 隐藏层连接权重
        self.hidden_weights = np.array([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]], dtype=float)
        # 输出层连接权重(就是当前PID参数值)
        self.CURRENT_PID_VALUE = np.array([P_VALUE, I_VALUE, D_VALUE], dtype=float)

        # 根据[t, t - 1]索引 (相当于超级小buffer)
        # 系统反馈向量
        self.y = np.zeros([OB_STEP], dtype=float)
        # 参考向量
        self.r = np.zeros([OB_STEP], dtype=float)

        # Feedback input vector
        self.input_y = np.zeros([OB_STEP], dtype=float)
        # Reference input vector
        self.input_r = np.zeros([OB_STEP], dtype=float)

        # 隐藏层输入
        self.hidden_input_p = np.zeros([OB_STEP], dtype=float)
        self.hidden_input_i = np.zeros([OB_STEP], dtype=float)
        self.hidden_input_d = np.zeros([OB_STEP], dtype=float)

        # 隐藏层输出
        self.hidden_output_p = np.zeros([OB_STEP], dtype=float)
        self.hidden_output_i = np.zeros([OB_STEP], dtype=float)
        self.hidden_output_d = np.zeros([OB_STEP], dtype=float)

        # 动作输出
        self.action = np.zeros([OB_STEP], dtype=float)

        self.OB_STEP = OB_STEP


    def threshold_div_by_zero(self, value):
        # 防止除零情况发生
        if np.fabs(value) < self.div_by_zero_tol:
            if value >= 0:
                return self.div_by_zero_tol
            else:
                return -self.div_by_zero_tol
        return value

    def p_transfer_function(self, v):
        # 比例传递函数
        return v

    def i_transfer_function(self, v, accumulated_v):
        # 积分传递函数
        return v * self.timestep + accumulated_v

    def d_transfer_function(self, v, past_v):
        # 微分传递函数
        return (v - past_v) / self.timestep

    def predict(self, reference, feedback):
        # Update weights
        self.learn(feedback)

        # Update variable history
        self.y[1] = self.y[0]
        self.r[1] = self.r[0]
        self.y[0] = feedback
        self.r[0] = reference

        self.input_y[1] = self.input_y[0]
        self.input_r[1] = self.input_r[0]
        self.hidden_input_p[1] = self.hidden_input_p[0]
        self.hidden_input_i[1] = self.hidden_input_i[0]
        self.hidden_input_d[1] = self.hidden_input_d[0]
        self.hidden_output_p[1] = self.hidden_output_p[0]
        self.hidden_output_i[1] = self.hidden_output_i[0]
        self.hidden_output_d[1] = self.hidden_output_d[0]

        self.action[1] = self.action[0]

        # Calculate input neuron outputs
        self.input_y[0] = self.p_transfer_function(feedback)
        self.input_r[0] = self.p_transfer_function(reference)

        # Calculate hidden P neurons outputs
        self.hidden_input_p[0] = (self.input_y[0] * self.hidden_weights[0][0] + self.input_r[0] * self.hidden_weights[1][0])
        self.hidden_output_p[0] = self.p_transfer_function(self.hidden_input_p[0])

        # Calculate hidden I neurons outputs
        self.hidden_input_i[0] = (self.input_y[0] * self.hidden_weights[0][1] + self.input_r[0] * self.hidden_weights[1][1])
        self.hidden_output_i[0] = self.i_transfer_function(self.hidden_input_i[0], self.hidden_output_i[1])

        # Calculate hidden D neurons outputs
        self.hidden_input_d[0] = (self.input_y[0] * self.hidden_weights[0][2] + self.input_r[0] * self.hidden_weights[1][2])
        self.hidden_output_d[0] = self.d_transfer_function(self.hidden_input_d[0], self.hidden_input_d[1])

        # Calculate output neuron outputs
        self.action[0] = (self.hidden_output_p[0] * self.CURRENT_PID_VALUE[0] + self.hidden_output_i[0] * self.CURRENT_PID_VALUE[1] + self.hidden_output_d[0] * self.CURRENT_PID_VALUE[2])

        return self.action[0]

    def learn(self, feedback):
        # Backprop
        delta_r = self.r[0] - self.y[0]
        delta_y = feedback - self.y[0]

        if delta_r >= self.tol: # 当误差较大的时候进行更新
            delta_output_weights = self.backprop(delta_r, delta_y)
            for idx in range(delta_output_weights.shape[0]): # 对三个P,I,D参数进行循环
                delta_output_weights[idx] = np.clip(delta_output_weights[0], -self.CURRENT_PID_VALUE[idx] * self.eta, +self.CURRENT_PID_VALUE[idx] * self.eta)
                self.CURRENT_PID_VALUE[idx] = self.CURRENT_PID_VALUE[idx] + delta_output_weights[idx]

    def backprop(self, delta_r, delta_y):
        delta_v = self.threshold_div_by_zero(self.action[0] - self.action[1])

        # Output layer weight changes
        delta_output_weights = np.zeros([3], dtype=float)
        delta_output_weights[0] = (-2 * delta_r * delta_y * self.hidden_output_p[0] / delta_v)
        delta_output_weights[1] = (-2 * delta_r * delta_y * self.hidden_output_i[0] / delta_v)
        delta_output_weights[2] = (-2 * delta_r * delta_y * self.hidden_output_d[0] / delta_v)

        return delta_output_weights