import numpy as np
import math
import pandas as pd


class v2rChannels:
    def __init__(self):
        self.h_bs = 5   
        self.h_ms = 1.5 
        self.decorrelation_distance = 6  
        self.shadow_std = 3    

    def get_path_loss(self, position_A, position_B):
        d1 = abs(position_A[0] - position_B[0])
        d2 = abs(position_A[1] - position_B[1])
        distance = math.hypot(d1, d2)
        d3D = math.sqrt(distance ** 2 + (self.h_bs - self.h_ms) ** 2) / 1000
        fc = 2
        PL = 28 + 22 * np.log10(d3D) + 20 * np.log10(fc)
        return PL

    def get_shadowing(self, delta_distance, shadowing):
        return np.exp(-1 * (delta_distance / self.decorrelation_distance)) * shadowing \
               + math.sqrt(1 - np.exp(-2 * (delta_distance / self.decorrelation_distance))) * np.random.normal(0, 3)


class Task:
    def __init__(self, q, inten, c, T):
        self.c_off = c 
        self.inten = inten
        self.q_off = q 
        self.T = T 


class vehicleC:
    def __init__(self, position, velocity, compute_resource, task):
        self.position = position
        self.velocity = velocity
        self.f_local = compute_resource
        self.task = task


class RSU:
    def __init__(self, position, compute_resource):
        self.position = position
        self.f_RSU = compute_resource


class EnvCore(object):
    def __init__(self):
        self.agent_num = 10  
        self.obs_dim = 15  
        self.action_dim = 48  
        self.n_RB_R = 12   
        self.time = 0

        # Vehicle
        self.vehicleC = []
        self.delta_distance_C = []
        self.n_vehC = self.agent_num
        self.local_limit = 1
        self.data = np.random.randint(5, 20) * 1e6

        # 信道
        self.V2Rchannels = v2rChannels()
        # 阴影衰落
        self.V2R_Shadowing = []
        # 路径损失
        self.V2R_pathloss = []
        # 路损和阴影衰落的和
        self.V2R_channels_abs = []
        # 快衰落信道
        self.V2R_channels_with_fastfading = []
        # 干扰
        self.V2R_interference_all = []
        self.RSU = [RSU([200, 0], 10)]

        # 噪声和增益
        self.sig2_dB = -90
        self.vehAntGain = 3
        self.vehNoiseFigure = 9
        self.sig2 = 10 ** (self.sig2_dB / 10)
        self.time_fast = 0.001
        self.time_slow = 0.1
        self.bandwidth = int(20e6)
        self.power_dB = [23, 10, 5, 0]
        self.id_data = {}

    def load_highD(self):
        # 读取文件并处理数据
        df = pd.read_csv('envs/02_tracks.csv', nrows=50000)
        grouped = df.groupby('id')
        
        for id, group in grouped:
            # 提取 x 和 y 列的值
            x_values = group['x'].values
            y_values = group['y'].values
            v_values = group['xVelocity'].values
            
            # 将结果存储到字典中
            self.id_data[id] = {'x': x_values, 'y': y_values, 'v': v_values}

    # 增加1个任务车辆信息
    def add_new_vehicleC(self, position, velocity, compute_resource, task):
        self.vehicleC.append(vehicleC(position, velocity, compute_resource, task))

    # 增加多个任务车辆信息
    def add_new_vehiclesC_by_number(self, n, dataC, limit):    # n是车辆数量
        self.load_highD()

        for i in range(1, n+1):
            dat = dataC
            dat = np.random.randint(5, 20) * 1e6
            cv = 1.8 * dat
            limit_local = limit
            task = Task(dat, 1.8, cv, limit_local)

            start_position = [self.id_data[i]['x'][0], self.id_data[i]['y'][0]]
            velocity = self.id_data[i]['v'][0]
            compute_resource = 0.5
            self.add_new_vehicleC(start_position, abs(velocity), compute_resource, task)

        self.V2R_Shadowing = np.random.normal(0, 4, [len(self.vehicleC), 1])
        self.delta_distance_C = np.asarray([c.velocity * self.time_slow for c in self.vehicleC])

    def new_random_game(self, n_VehC = 0, data = 0, local_limit = 0):
        self.vehicleC = []
        if n_VehC > 0:
            self.n_vehC = n_VehC
        if data > 0:
            self.data = data
        if local_limit > 0:
            self.local_limt = local_limit
        self.add_new_vehiclesC_by_number(self.n_vehC, self.data, self.local_limit)
        self.renew_Channel()
        self.renew_channels_fastfading()
        self.RSU = [RSU([200, 0], 10)]

    # 更新任务车位置
    def renew_vC_positions(self):
        self.time = self.time + 2
        for i in range(self.n_vehC):
            self.vehicleC[i].position = [self.id_data[i+1]['x'][self.time], self.id_data[i+1]['y'][self.time]]
            self.vehicleC[i].velocity = self.id_data[i+1]['v'][self.time]
            self.vehicleC[i].task.q_off = np.random.randint(5, 20) * 1e6
            # self.vehicleC[i].task.q_off = 8 * 1e6
        if self.time == 40:
            self.time = 0

    # 更新慢衰落
    def renew_Channel(self):
        self.V2R_pathloss = np.zeros(len(self.vehicleC))      # 路径损耗
        self.V2R_channels_abs = np.zeros(len(self.vehicleC))

        for i in range(len(self.vehicleC)):
            self.V2R_Shadowing[i] = self.V2Rchannels.get_shadowing(self.delta_distance_C[i], self.V2R_Shadowing[i])
            # 路径损耗
            self.V2R_pathloss[i] = self.V2Rchannels.get_path_loss(self.vehicleC[i].position, self.RSU[0].position)
        self.V2R_Shadowing = self.V2R_Shadowing.reshape(-1)
        self.V2R_pathloss = self.V2R_pathloss.reshape(-1)
        self.V2R_channels_abs = self.V2R_pathloss + self.V2R_Shadowing

    # 更新快衰落
    def renew_channels_fastfading(self):
        # V2R链路 RSU链路 Rayleigh fading
        V2R_channels_with_fastfading = np.repeat(self.V2R_channels_abs[:, np.newaxis], self.n_RB_R, axis=1)
        h_rayleigh = np.abs(np.random.normal(0, 1, V2R_channels_with_fastfading.shape)
                   + 1j * np.random.normal(0, 1, V2R_channels_with_fastfading.shape)) / math.sqrt(2)
        #a Rayleigh fading channel gain
        self.V2R_channels_with_fastfading = V2R_channels_with_fastfading - 20 * np.log10(h_rayleigh)

    def get_state(self):
        V2R_fast = np.zeros((1, self.n_RB_R))  # 初始化,每个RSU每个信道的fastfading
        result = []
        for idx in range(len(self.vehicleC)):
            V2R_fast = (self.V2R_channels_with_fastfading[idx, :] - self.V2R_channels_abs[idx] + 10) / 35
            test = np.concatenate((np.reshape(V2R_fast, -1), np.asarray([self.vehicleC[idx].task.c_off, self.vehicleC[idx].task.q_off,
                                                                self.vehicleC[idx].task.T])))
            result = np.concatenate((result, test))
        return result

    def reset(self):
        self.new_random_game()
        V2R_fast = np.zeros((1, self.n_RB_R))  # 初始化,每个RSU每个信道的fastfading
        result = self.get_state()
        sub_agent_obs = []
        for i in range(self.agent_num):
            sub_obs = result[i * self.obs_dim:(i + 1) * self.obs_dim]
            sub_obs = sub_obs.reshape(self.obs_dim,)
            sub_agent_obs.append(sub_obs)
        return sub_agent_obs

    def step(self, actions):
        sub_agent_obs = []
        sub_agent_reward = []
        sub_agent_done = []
        sub_agent_info = []
        reward_list = -self.dataTrans(actions)
        self.renew_vC_positions()
        self.renew_Channel()
        self.renew_channels_fastfading()
        result = self.get_state()
        for i in range(self.agent_num):
            ob = result[i * self.obs_dim:(i + 1) * self.obs_dim]
            ob = ob.reshape(self.obs_dim, )
            sub_agent_obs.append(ob)
            sub_agent_reward.append([reward_list[i]])
            sub_agent_done.append(False)
            sub_agent_info.append({})

        return [sub_agent_obs, sub_agent_reward, sub_agent_done, sub_agent_info]

    def dataTrans(self, actions):
        channel_selection = np.zeros(len(self.vehicleC))
        power_selection = np.zeros(len(self.vehicleC))
        signal = np.zeros(len(self.vehicleC))
        V2R_interference = np.zeros(len(self.vehicleC)) + self.sig2
        Rate = np.zeros(len(self.vehicleC))

        delay_Trans = np.zeros(len(self.vehicleC))

        action = np.zeros(len(self.vehicleC))
        j = 0
        for sub in actions:
            index = np.where(sub == 1)
            action[j] = index[0]
            j = j + 1

        for i in range(len(self.vehicleC)):
            channel_selection[i] = action[i] % self.n_RB_R  # 信道选择
            power_selection[i] = action[i] // self.n_RB_R
        channel_selection = channel_selection.astype(int)
        power_selection = power_selection.astype(int)

        for i in range(len(self.vehicleC)):
            signal[i] = 10 ** ((self.power_dB[power_selection[i]] - self.V2R_channels_with_fastfading[i][
                channel_selection[i]]
                                + 2 * self.vehAntGain - self.vehNoiseFigure) / 10)

            for k in range(len(self.vehicleC)):
                if k != i and channel_selection[k] == channel_selection[i]:
                    V2R_interference[i] += 10 ** (
                            (self.power_dB[power_selection[i]]
                             - self.V2R_channels_with_fastfading[i][channel_selection[i]]
                             + 2 * self.vehAntGain - self.vehNoiseFigure) / 10)
            Rate[i] = np.multiply(self.bandwidth, np.log2(1 + np.divide(signal[i], V2R_interference[i])))

            delay_Trans[i] = np.divide(self.vehicleC[i].task.q_off, Rate[i])  # 传输时延

        return delay_Trans




class EnvCore_Eval(object):

    def __init__(self):
        self.agent_num = 10  
        self.obs_dim = 15  
        self.action_dim = 48  
        self.n_RB_R = 12   
        self.time = 0

        # Vehicle
        self.vehicleC = []
        self.delta_distance_C = []
        self.n_vehC = self.agent_num
        self.velocityC = 10
        self.local_limit = 1
        self.data = 10e6
        self.k = 10 ** (-28)  # 本地处理时的能量系数k

        # 信道
        self.V2Rchannels = v2rChannels()
        # 阴影衰落
        self.V2R_Shadowing = []
        # 路径损失
        self.V2R_pathloss = []
        # 路损和阴影衰落的和
        self.V2R_channels_abs = []
        # 快衰落信道
        self.V2R_channels_with_fastfading = []
        # 干扰
        self.V2R_interference_all = []
        self.RSU = [RSU([200, 0], 10)]

        # 噪声和增益
        self.sig2_dB = -90
        self.vehAntGain = 3
        self.vehNoiseFigure = 9
        self.sig2 = 10 ** (self.sig2_dB / 10)
        self.time_fast = 0.001
        self.time_slow = 0.1
        self.bandwidth = int(20e6)
        self.power_dB = [23, 10, 5, 0]
        self.id_data = {}

    def load_highD(self):
        # 读取文件并处理数据
        df = pd.read_csv('envs/02_tracks.csv', nrows=50000)
        grouped = df.groupby('id')
        
        for id, group in grouped:
            # 提取 x 和 y 列的值
            x_values = group['x'].values
            y_values = group['y'].values
            v_values = group['xVelocity'].values
            
            # 将结果存储到字典中
            self.id_data[id] = {'x': x_values, 'y': y_values, 'v': v_values}

    def add_new_vehicleC(self, position, velocity, compute_resource, task):
        self.vehicleC.append(vehicleC(position, velocity, compute_resource, task))

    def add_new_vehiclesC_by_number(self, n, dataC, limit): 
        self.load_highD()

        for i in range(1, n+1):
            dat = dataC
            dat = np.random.randint(5, 20) * 1e6
            cv = 1.8 * dat
            limit_local = limit
            task = Task(dat, 1.8, cv, limit_local)

            start_position = [self.id_data[i]['x'][0], self.id_data[i]['y'][0]]
            velocity = self.id_data[i]['v'][0]
            compute_resource = 0.5
            self.add_new_vehicleC(start_position, abs(velocity), compute_resource, task)

        self.V2R_Shadowing = np.random.normal(0, 4, [len(self.vehicleC), 1])
        self.delta_distance_C = np.asarray([c.velocity * self.time_slow for c in self.vehicleC])

    def new_random_game(self, n_VehC = 0, data = 0, local_limit = 0):
        self.vehicleC = []
        if n_VehC > 0:
            self.n_vehC = n_VehC
        if data > 0:
            self.data = data
        if local_limit > 0:
            self.local_limt = local_limit
        self.add_new_vehiclesC_by_number(self.n_vehC, self.data, self.local_limit)
        self.renew_Channel()
        self.renew_channels_fastfading()
        self.RSU = [RSU([200, 0], 10)]

    # 更新任务车位置
    def renew_vC_positions(self):
        self.time = self.time + 2
        for i in range(self.n_vehC):
            self.vehicleC[i].position = [self.id_data[i+1]['x'][self.time], self.id_data[i+1]['y'][self.time]]
            self.vehicleC[i].velocity = self.id_data[i+1]['v'][self.time]
        if self.time == 40:
            self.time = 0

    # 更新慢衰落
    def renew_Channel(self):
        self.V2R_pathloss = np.zeros(len(self.vehicleC))      # 路径损耗
        self.V2R_channels_abs = np.zeros(len(self.vehicleC))

        for i in range(len(self.vehicleC)):
            self.V2R_Shadowing[i] = self.V2Rchannels.get_shadowing(self.delta_distance_C[i], self.V2R_Shadowing[i])
            self.V2R_pathloss[i] = self.V2Rchannels.get_path_loss(self.vehicleC[i].position, self.RSU[0].position)
        self.V2R_Shadowing = self.V2R_Shadowing.reshape(-1)
        self.V2R_pathloss = self.V2R_pathloss.reshape(-1)
        self.V2R_channels_abs = self.V2R_pathloss + self.V2R_Shadowing

    # 更新快衰落
    def renew_channels_fastfading(self):
        V2R_channels_with_fastfading = np.repeat(self.V2R_channels_abs[:, np.newaxis], self.n_RB_R, axis=1)
        h_rayleigh = np.abs(np.random.normal(0, 1, V2R_channels_with_fastfading.shape)
                   + 1j * np.random.normal(0, 1, V2R_channels_with_fastfading.shape)) / math.sqrt(2)
        #a Rayleigh fading channel gain
        self.V2R_channels_with_fastfading = V2R_channels_with_fastfading - 20 * np.log10(h_rayleigh)

    def get_state(self):
        V2R_fast = np.zeros((1, self.n_RB_R)) 
        result = []
        for idx in range(len(self.vehicleC)):
            V2R_fast = (self.V2R_channels_with_fastfading[idx, :] - self.V2R_channels_abs[idx] + 10) / 35
            test = np.concatenate((np.reshape(V2R_fast, -1), np.asarray([self.vehicleC[idx].task.c_off, self.vehicleC[idx].task.q_off,
                                                                self.vehicleC[idx].task.T])))
            result = np.concatenate((result, test))
        return result

    def reset(self):
        self.new_random_game()
        V2R_fast = np.zeros((1, self.n_RB_R)) 
        result = self.get_state()
        sub_agent_obs = []
        for i in range(self.agent_num):
            sub_obs = result[i * self.obs_dim:(i + 1) * self.obs_dim]
            sub_obs = sub_obs.reshape(self.obs_dim,)
            sub_agent_obs.append(sub_obs)
        return sub_agent_obs

    def step(self, actions):
        sub_agent_obs = []
        sub_agent_reward = []
        sub_agent_done = []
        sub_agent_info = []
        reward_list = -self.dataTrans(actions)
        self.renew_vC_positions()
        self.renew_Channel()
        self.renew_channels_fastfading()
        result = self.get_state()
        for i in range(self.agent_num):
            ob = result[i * self.obs_dim:(i + 1) * self.obs_dim]
            ob = ob.reshape(self.obs_dim, )
            sub_agent_obs.append(ob)
            sub_agent_reward.append([reward_list[i]])
            sub_agent_done.append(False)
            sub_agent_info.append({})

        return [sub_agent_obs, sub_agent_reward, sub_agent_done, sub_agent_info]

    def dataTrans(self, actions):
        channel_selection = np.zeros(len(self.vehicleC))
        power_selection = np.zeros(len(self.vehicleC))

        signal = np.zeros(len(self.vehicleC))
        V2R_interference = np.zeros(len(self.vehicleC)) + self.sig2
        Rate = np.zeros(len(self.vehicleC))

        delay_Trans = np.zeros(len(self.vehicleC))

        action = np.zeros(len(self.vehicleC))
        j = 0
        for sub in actions:
            index = np.where(sub == 1)
            action[j] = index[0]
            j = j + 1

        for i in range(len(self.vehicleC)):
            channel_selection[i] = action[i] % self.n_RB_R  # 信道选择
            power_selection[i] = action[i] // self.n_RB_R
        channel_selection = channel_selection.astype(int)
        power_selection = power_selection.astype(int)

        for i in range(len(self.vehicleC)):
            signal[i] = 10 ** ((self.power_dB[power_selection[i]] - self.V2R_channels_with_fastfading[i][
                channel_selection[i]]
                                + 2 * self.vehAntGain - self.vehNoiseFigure) / 10)

            for k in range(len(self.vehicleC)):
                if k != i and channel_selection[k] == channel_selection[i]: 
                    V2R_interference[i] += 10 ** (
                            (self.power_dB[power_selection[i]]
                             - self.V2R_channels_with_fastfading[i][channel_selection[i]]
                             + 2 * self.vehAntGain - self.vehNoiseFigure) / 10)
            Rate[i] = np.multiply(self.bandwidth, np.log2(1 + np.divide(signal[i], V2R_interference[i])))

            delay_Trans[i] = np.divide(self.vehicleC[i].task.q_off, Rate[i])  # 传输时延

        return delay_Trans
