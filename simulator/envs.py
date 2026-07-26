import os, sys, random, time
import logging
import numpy as np

from .objects import Driver, Node
from .utilities import ids_1dto2d, get_neighbor_list, ids_2dto1d, datetime_range

from .objects import *
from .utilities import *
# from algorithm import *

# current_time = time.strftime("%Y%m%d_%H-%M")
# log_dir = "/nfs/private/linkaixiang_i/data/dispatch_simulator/experiments/"+current_time + "/"
# mkdir_p(log_dir)
# logging.basicConfig(filename=log_dir +'logger_env.log', level=logging.INFO)


# 配置logger并设置等级为DEBUG
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# 配置控制台Handler并设置等级为DEBUG
logger_ch = logging.StreamHandler()
logger_ch.setLevel(logging.DEBUG)
logger_ch.setFormatter(logging.Formatter(
    '%(asctime)s[%(levelname)s][%(lineno)s:%(funcName)s]||%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'))
# 将Handler加入logger
logger.addHandler(logger_ch)
RANDOM_SEED = 0  # unit test use this random seed.


def prn_obj(obj):
    print('\n'.join(['%s:%s' % item for item in obj.__dict__.items()]))

class CityReal:
    '''A real city is consists of M*N grids '''
    '''一个真实的城市存在于M*N的网格之中'''
    def __init__(self, mapped_matrix_int, order_num_dist, idle_driver_dist_time, idle_driver_location_mat, order_time_dist, order_price_dist,
                 l_max, M, N, n_side, probability, real_orders="", onoff_driver_location_mat="",
                 global_flag="global", time_interval=10):
        """
        :param mapped_matrix_int: 2D matrix: each position is either -100 or grid id from order in real data.
        2D矩阵：在实际数据中，每个位置都是-100或网格id。
        :param order_num_dist: 144 [{node_id1: [mu, std]}, {node_id2: [mu, std]}, ..., {node_idn: [mu, std]}]
                            node_id1 is node the index in self.nodes   node_id1就是代表自身节点
                            order_num_dist是长度为144的列表。每个元素都是一个字典。
                            order_num_dist[0][22] = [mean, variance] 表示的是在t=0时刻，网格编号为22的订单数量的均值和方差
        :param idle_driver_dist_time: [[mu1, std1], [mu2, std2], ..., [mu144, std144]] mean and variance of idle drivers in
        大小为144 x网格数的二维列表。  idle_driver_location_mat[0][grid_matrix_id] = 100 indicate mean of idle drivers at grid _grid_matrix_id_ at time 0.
       表示的是t=0时刻，网格数等于100，空闲司机数量的均值
        the city at each time  城市每个时间戳闲散驾驶员均值和方差
        :param idle_driver_location_mat: 144 x num_valid_grids matrix.    144个时间戳乘以地图中有效网格数
        :param order_time_dist: [ 0.27380797,..., 0.00205766] The probs of order duration = 1 to 9    订单持续时间  如果等于9，则持续时间为1.5小时
        :param order_price_dist: [[10.17, 3.34],   # mean and std of order's price, order durations = 10 minutes.
                                   [15.02, 6.90],  # mean and std of order's price, order durations = 20 minutes.
                                   ...,]   订单价格
        :param onoff_driver_location_mat: 144 x 504 x 2: 144 total time steps, num_valid_grids = 504.
        mean and std of online driver number - offline driver number   在线驾驶员-离线驾驶员的数量
        onoff_driver_location_mat[t] = [[-0.625       2.92350389]  <-- Corresponds to the grid in target_node_ids
                                        [ 0.09090909  1.46398452]    在线司机对应目标节点中的网络  在t=0时刻，更新的均值和标准差
                                        [ 0.09090909  2.36596622]
                                        [-1.2         2.05588586]...]
        :param M:
        :param N:
        :param n_side:
        :param time_interval:
        :param l_max: The max-duration of an order
        :return:
        """



        # City.__init__(self, M, N, n_side, time_interval)
        self.M = M  # row numbers    行数    矩阵的大小mapping_matrix_int
        self.N = N  # column numbers   列数
        self.nodes = [Node(i) for i in range(M * N)]  # a list of nodes: node id start from 0  列表节点  从0开始
        self.drivers = {}  # driver[driver_id] = driver_instance  , driver_id start from 0   司机的距离
        self.n_drivers = 0  # total idle number of drivers. online and not on service.  总的空闲司机数，在线加上离线的司机数量
        self.n_offline_drivers = 0  # total number of offline drivers.   总的离线司机数
        self.construct_map_simulation(M, N, n_side)  #  创建一个地图模拟器
        self.city_time = 0  #开始时间
        # self.idle_driver_distribution = np.zeros((M, N))
        self.n_intervals = 1440 // time_interval  #1440/时间间隔（10） =  144
        self.n_nodes = self.M * self.N    #节点数等于M*N
        self.n_side = n_side
        self.order_response_rate = 0   #订单响应率

        self.RANDOM_SEED = RANDOM_SEED  #随机种子

        #   从1开始，决定了结点可以计算的最大跨越层数，如l_max=7,每个节点可以计算7个时间步内所有可以到达的结点
        self.l_max = l_max  # Start from 1. The max number of layers an order can across.
        assert l_max <= M-1 and l_max <= N-1
        assert 1 <= l_max <= 9   # Ignore orders less than 10 minutes and larger than 1.5 hours  忽略订单小于10分钟或者大于1.5小时的订单

        self.target_grids = []   #目标网格
        self.n_valid_grids = 0  # num of valid grid   有效网格数
        self.nodes = [None for _ in np.arange(self.M * self.N)]
        self.construct_node_real(mapped_matrix_int)
        self.mapped_matrix_int = mapped_matrix_int

        self.construct_map_real(n_side)
        self.order_num_dist = order_num_dist
        self.distribution_name = "Poisson"
        self.idle_driver_dist_time = idle_driver_dist_time
        self.idle_driver_location_mat = idle_driver_location_mat
        self.order_time_dist = order_time_dist[:l_max]/np.sum(order_time_dist[:l_max])
        self.order_price_dist = order_price_dist

        target_node_ids = []
        target_grids_sorted = np.sort(mapped_matrix_int[np.where(mapped_matrix_int > 0)])   # 所有有效网格编号从小到大排序
        for item in target_grids_sorted:
            x, y = np.where(mapped_matrix_int == item)
            target_node_ids.append(ids_2dto1d(x, y, M, N)[0])   # 应该是记录id从低到高的在地图中的位置，如果mappend_matrix_int是代表id的话
        self.target_node_ids = target_node_ids
        # store valid note id. Sort by number of orders emerged. descending.
        # 存储有效的记录id。按出现的订单数排序。下降。
        self.node_mapping = {}
        self.construct_mapping()

        self.real_orders = real_orders  # 4 weeks' data
        # [[92, 300, 143, 2, 13.2],...] origin grid, destination grid, start time, end time, price.


        self.p = probability   #每个订单出现的概率 sample probability
        self.time_keys = [int(dt.strftime('%H%M')) for dt in
                          datetime_range(datetime(2017, 9, 1, 0), datetime(2017, 9, 2, 0),
                                        timedelta(minutes=time_interval))]
        self.day_orders = []  # one day's order.

        self.onoff_driver_location_mat = onoff_driver_location_mat  # [在线，离线]司机概率分布

        # Stats
        self.all_grids_on_number = 0  # current online # drivers.
        self.all_grids_off_number = 0   #当前离线的司机数量


        self.out_grid_in_orders = np.zeros((self.n_intervals, len(self.target_grids)))
        self.global_flag = global_flag
        self.weights_layers_neighbors = [1.0, np.exp(-1), np.exp(-2)]

        #mine
        self.finished_order_num = 0
        self.day_orders_num = 0
        self.on_off_nums = np.ones((144, self.n_valid_grids))
        self.neighbors_list = []
        for idx, node_id in enumerate(self.target_grids):
            neighbor_indices = self.nodes[node_id].layers_neighbors_id[0]  # index in env.nodes
            neighbor_ids = [self.target_grids.index(self.nodes[item].get_node_index()) for item in neighbor_indices]
            neighbor_ids.append(idx)
            # index in env.target_grids == index in state
            self.neighbors_list.append(neighbor_ids)
        self.valid_action_mask = np.ones((self.n_valid_grids, 7))
        for grid_idx, grid_id in enumerate(self.target_grids):
            for neighbor_idx, neighbor in enumerate(self.nodes[grid_id].neighbors):
                if neighbor is None:
                    self.valid_action_mask[grid_idx, neighbor_idx] = 0
        #模拟整个城市

     #建造一张六边形网格地图
    def construct_map_simulation(self, M, N, n):
        """Connect node to its neighbors based on a simulated M by N map
            :param M: M row index matrix
            :param N: N column index matrix
            :param n: n - sided polygon   多边形
        """
        for idx, current_node in enumerate(self.nodes):
            if current_node is not None:

                i, j = ids_1dto2d(idx, M, N)
                # print(idx, i, j)
                current_node.set_neighbors(get_neighbor_list(i, j, M, N, n, self.nodes))

    def construct_mapping(self):   #构造印射
        """
        :return:
        """
        target_grid_id = self.mapped_matrix_int[np.where(self.mapped_matrix_int>0)]
        for g_id, n_id in zip(target_grid_id, self.target_grids):
            self.node_mapping[g_id] = n_id

    def construct_node_real(self, mapped_matrix_int):
        """ Initialize node, only valid node in mapped_matrix_in will be initialized.
        初始化节点，将只初始化映射的矩阵中的有效节点
        """
        row_inds, col_inds = np.where(mapped_matrix_int >= 0)

        target_ids = []  # start from 0. 
        for x, y in zip(row_inds, col_inds):
            node_id = ids_2dto1d(x, y, self.M, self.N)
            self.nodes[node_id] = Node(node_id)  # node id start from 0.
            target_ids.append(node_id)

        # 获得layers_neighbors(i层代表第i个时间步结点可以到达的最远结点)
        for x, y in zip(row_inds, col_inds):
            node_id = ids_2dto1d(x, y, self.M, self.N)
            self.nodes[node_id].get_layers_neighbors(self.l_max, self.M, self.N, self)  # 每一层是每个时间步内可以到达的结点（不往回走）


        self.target_grids = target_ids  # 有效结点
        self.n_valid_grids = len(target_ids)    # 有效结点数
        
    def construct_map_real(self, n_side):
        """Build node connection. 
        """
        for idx, current_node in enumerate(self.nodes):
            i, j = ids_1dto2d(idx, self.M, self.N)
            if current_node is not None:
                current_node.set_neighbors(get_neighbor_list(i, j, self.M, self.N, n_side, self.nodes))

    def initial_order_random(self, distribution_all, dis_paras_all):
        """ Initialize order distribution
        :param distribution: 'Poisson', 'Gaussian'
        :param dis_paras:     lambda,    mu, sigma
        """
        for idx, node in enumerate(self.nodes):
            if node is not None:
                node.order_distribution(distribution_all[idx], dis_paras_all[idx])

    # 获得一个时刻的state，[1] idle_driver_dist [2] order_dist
    def get_observation(self):
        next_state = np.zeros((2, self.M, self.N))
        for _node in self.nodes:
            if _node is not None:
                row_id, column_id = ids_1dto2d(_node.get_node_index(), self.M, self.N)
                next_state[0, row_id, column_id] = _node.idle_driver_num
                next_state[1, row_id, column_id] = _node.order_num

        return next_state

    def get_num_idle_drivers(self):
        """ Compute idle drivers
        :return:
        """
        temp_n_idle_drivers= 0
        for _node in self.nodes:
            if _node is not None:
                temp_n_idle_drivers += _node.idle_driver_num
        return temp_n_idle_drivers

    def get_observation_driver_state(self):
        """ Get idle driver distribution, computing #drivers from node.
        :return:
        """
        next_state = np.zeros((self.M, self.N))
        for _node in self.nodes:
            if _node is not None:
                row_id, column_id = ids_1dto2d(_node.get_node_index(), self.M, self.N)
                next_state[row_id, column_id] = _node.get_idle_driver_numbers_loop()

        return next_state
    def reset_randomseed(self, random_seed):
        self.RANDOM_SEED = random_seed

    def reset(self):
        """ Return initial observation: get order distribution and idle driver distribution
       返回初始观察：得到订单分布和空闲驱动分布
        """

        _M = self.M
        _N = self.N
        assert self.city_time == 0
        # initialization drivers according to the distribution at time 0
        num_idle_driver = self.utility_get_n_idle_drivers_real()
        self.step_driver_online_offline_control(num_idle_driver)

        # generate orders at first time step
        distribution_name = [self.distribution_name]*(_M*_N)
        distribution_param_dictionary = self.order_num_dist[self.city_time]
        distribution_param = [0]*(_M*_N)
        for key, value in distribution_param_dictionary.items():
            if self.distribution_name == 'Gaussian':
                mu, sigma = value
                distribution_param[key] = mu, sigma
            elif self.distribution_name == 'Poisson':
                mu = value[0]
                distribution_param[key] = mu
            else:
                print ("Wrong distribution")

        self.initial_order_random(distribution_name, distribution_param)
        self.step_generate_order_real()

        return self.get_observation()

    # 清空环境，生成订单和车辆
    def reset_clean(self, generate_order=1, ratio=1, city_time=""):
        """ 1. bootstrap oneday's order data.  引导一天的订单数据。
            2. clean current drivers and orders, regenerate new orders and drivers.
            can reset anytime  清理当前驱动程序和订单，重新生成新的订单和驱动程序。
可以随时重置
        :return:
        """
        if city_time != "":
            self.city_time = city_time

        # clean orders and drivers
        self.drivers = {}  # driver[driver_id] = driver_instance  , driver_id start from 0
        self.n_drivers = 0  # total idle number of drivers. online and not on service.
        self.n_offline_drivers = 0  # total number of offline drivers.
        # 把所有节点的订单，车辆啥的全都清空
        for node in self.nodes:
            if node is not None:
                node.clean_node()
        if city_time == 0:
            asd  =1
        # Generate one day's order.
        # 会根据真实订单有概率的选取出其中一部分来生成
        if generate_order == 1:
            self.utility_bootstrap_oneday_order()   # 生成一天的订单，结果存储到env.day_orders,144个[]，每个[]有当天的所有订单

        # Init orders of current time step   当前时间步长的初始顺序
        moment = self.city_time % self.n_intervals
        self.step_bootstrap_order_real(self.day_orders[moment])     # 把0时刻的订单分配给所有节点

        # Init current driver distribution  初始化空闲司机分配，该方法只在每天0时刻用到了
        if self.global_flag == "global":
            num_driver = self.utility_get_n_idle_drivers_real()
            num_driver = int(num_driver * ratio)
        else:
            num_driver = self.utility_get_n_idle_drivers_nodewise()
        num_idle_driver = int(num_driver*0.15)
        num_off_line_driver = num_driver - num_idle_driver
        self.step_driver_online_offline_control_new(num_idle_driver, True)
        self.step_driver_online_offline_control_new(num_off_line_driver, False)
        return self.get_observation()

    def utility_collect_offline_drivers_id(self):
        """count how many drivers are offline
        :return: offline_drivers: a list of offline driver id
        """
        count = 0 # offline driver num
        offline_drivers = []   # record offline driver id
        for key, _driver in self.drivers.items():
            if _driver.online is False:
                count += 1
                offline_drivers.append(_driver.get_driver_id())
        return offline_drivers

    def utility_get_n_idle_drivers_nodewise(self):
        """ compute idle drivers.
        :return:
        """
        time = self.city_time % self.n_intervals
        idle_driver_num = np.sum(self.idle_driver_location_mat[time])
        return int(idle_driver_num)


    def utility_add_driver_real_new(self, num_added_driver):
        curr_idle_driver_distribution = self.get_observation()[0]
        curr_idle_driver_distribution_resort = np.array(
            [int(curr_idle_driver_distribution.flatten()[index]) for index in
             self.target_node_ids])

        idle_driver_distribution = self.idle_driver_location_mat[self.city_time % self.n_intervals, :]

        idle_diff = idle_driver_distribution.astype(int) - curr_idle_driver_distribution_resort
        idle_diff[np.where(idle_diff <= 0)] = 0

        node_ids = np.random.choice(self.target_node_ids, size=[num_added_driver],
                                    p=idle_diff/float(np.sum(idle_diff)))

        n_total_drivers = len(self.drivers.keys())
        for ii, node_id in enumerate(node_ids):
            added_driver_id = n_total_drivers + ii
            self.drivers[added_driver_id] = Driver(added_driver_id)
            self.drivers[added_driver_id].set_position(self.nodes[node_id])
            self.nodes[node_id].add_driver(added_driver_id, self.drivers[added_driver_id])

        self.n_drivers += num_added_driver
    # 上线司机
    def utility_add_driver_real_new_offlinefirst(self, num_added_driver, flag_idle):

        # curr_idle_driver_distribution = self.get_observation()[0][np.where(self.mapped_matrix_int > 0)]
        curr_idle_driver_distribution = self.get_observation()[0]   #   15×17当前空闲司机分布
        curr_idle_driver_distribution_resort = np.array([int(curr_idle_driver_distribution.flatten()[index]) for index in
                                                         self.target_node_ids]) # 模拟器中当前时刻的空闲司机分布

        idle_driver_distribution = self.idle_driver_location_mat[self.city_time % self.n_intervals, :]  # 真实数据中当前时刻的空闲司机分布

        idle_diff = idle_driver_distribution.astype(int) - curr_idle_driver_distribution_resort # 真实数据空闲司机数 - 模拟器中当空闲司机分布
        idle_diff[np.where(idle_diff <= 0)] = 0 # 模拟器空闲司机更多的话，不管，设为0，某节点模拟器差真实数据司机越多，那么该节点越有可能选中去加一个司机

        if float(np.sum(idle_diff)) == 0:   # 如果当前模拟的所有节点空闲司机都比真实数据的空闲司机要多，那么返回
            return
        np.random.seed(self.RANDOM_SEED)
        # 总共要增加num_added_driver个司机，所有选出num_added_driver个节点，每个节点可能被选中的概率 = 该节点模拟器少于真实的差值/总差值
        node_ids = np.random.choice(self.target_node_ids, size=[num_added_driver],
                                    p=idle_diff/float(np.sum(idle_diff)))

        for ii, node_id in enumerate(node_ids):
            n_total_drivers = len(self.drivers.keys())  # 所有司机数
            added_driver_id = n_total_drivers  # 新增司机的id
            self.drivers[added_driver_id] = Driver(added_driver_id)  # 增加这个司机
            self.drivers[added_driver_id].set_position(self.nodes[node_id])  # 设定这个司机的位置
            self.nodes[node_id].add_driver(added_driver_id, self.drivers[added_driver_id])  # 这个节点添加上这个司机
            self.n_drivers += 1  # 总空闲司机数加1
            if not flag_idle:
                self.drivers[added_driver_id].set_offline()
                self.n_offline_drivers += 1    # 总离线司机数+1
                self.n_drivers -= 1  # 总司机数加1
            #
            # if self.nodes[node_id].offline_driver_num > 0:  # 如果该节点有下线的司机，那么就把一个下线的司机上线
            #     self.nodes[node_id].set_offline_driver_online()
            #     self.n_drivers += 1
            #     self.n_offline_drivers -= 1
            # else:
            #
            #     n_total_drivers = len(self.drivers.keys())  # 所有司机数
            #     added_driver_id = n_total_drivers   # 新增司机的id
            #     self.drivers[added_driver_id] = Driver(added_driver_id) # 增加这个司机
            #     if flag_idle :
            #         self.drivers[added_driver_id].set_offline()
            #     self.drivers[added_driver_id].set_position(self.nodes[node_id]) # 设定这个司机的位置
            #     self.nodes[node_id].add_driver(added_driver_id, self.drivers[added_driver_id])  # 这个节点添加上这个司机
            #     self.n_drivers += 1 # 总司机数加1

    # 上线目标结点的司机
    def utility_add_driver_real_nodewise(self, node_id, num_added_driver):

        while num_added_driver > 0:
            # 若目标结点中还有下线状态的司机
            if self.nodes[node_id].offline_driver_num > 0:
                self.nodes[node_id].set_offline_driver_online()
                self.n_drivers += 1
                self.n_offline_drivers -= 1
            # 否则增加一个司机
            # else:
            #     n_total_drivers = len(self.drivers.keys())
            #     added_driver_id = n_total_drivers
            #     self.drivers[added_driver_id] = Driver(added_driver_id)
            #     self.drivers[added_driver_id].set_position(self.nodes[node_id])
            #     self.nodes[node_id].add_driver(added_driver_id, self.drivers[added_driver_id])
            #     self.n_drivers += 1
            num_added_driver -= 1
    def utility_add_driver_real_future(self, node_id, num_added_driver):

        while num_added_driver > 0:
            # 若目标结点中还有下线状态的司机
            n_total_drivers = len(self.drivers.keys())
            added_driver_id = n_total_drivers
            self.drivers[added_driver_id] = Driver(added_driver_id)
            self.drivers[added_driver_id].set_position(self.nodes[node_id])
            self.nodes[node_id].add_driver(added_driver_id, self.drivers[added_driver_id])
            # self.n_drivers += 1

            num_added_driver -= 1

    def utility_add_driver_real_future_future(self, node_id, num_added_driver, start_time, start_node):

        while num_added_driver > 0:
            # 若目标结点中还有下线状态的司机
            n_total_drivers = len(self.drivers.keys())
            added_driver_id = n_total_drivers
            self.drivers[added_driver_id] = Driver(added_driver_id, start_node=start_node, start_time=start_time)
            self.drivers[added_driver_id].set_position(self.nodes[node_id])
            self.nodes[node_id].add_driver(added_driver_id, self.drivers[added_driver_id])
            # self.n_drivers += 1

            num_added_driver -= 1

    def utility_set_drivers_offline_real_nodewise(self, node_id, n_drivers_to_off):

        while n_drivers_to_off > 0:
            if self.nodes[node_id].idle_driver_num > 0:
                self.nodes[node_id].set_idle_driver_offline_random()
                self.n_drivers -= 1
                self.n_offline_drivers += 1
                n_drivers_to_off -= 1
                self.all_grids_off_number += 1
            else:
                break
    def utility_set_drivers_offline_real_forever(self, node_id, n_drivers_to_off):

        while n_drivers_to_off > 0:
            if self.nodes[node_id].idle_driver_num > 0:
                self.nodes[node_id].set_idle_driver_offline_random_forever()
                self.n_offline_drivers += 1
                n_drivers_to_off -= 1
            else:
                break

    def utility_set_drivers_offline_real_high(self, node_id, n_drivers_to_off):
        node_drive = self.nodes[node_id].drivers.items()
        idle_driver_num = self.nodes[node_id].idle_driver_num
        while n_drivers_to_off > 0:
            if idle_driver_num > 0:
                for key, item in node_drive:
                    if item.onservice is False and item.online is True:
                        idle_driver_num -= 1
                        removed_driver_id = key
                        # a = self.nodes[node_id].get_driver()
                        removed_driver = self.nodes[node_id].remove_driver_id(removed_driver_id)
                        # a = self.nodes[node_id].get_driver()
                        break
                for keyy in self.drivers:
                    if removed_driver == self.drivers[keyy]:
                        self.drivers.pop(keyy)
                        break
                # self.nodes[node_id].set_idle_driver_offline_random()

                # self.n_offline_drivers = self.n_offline_drivers - 1

                n_drivers_to_off -= 1
            else:
                break

    def utility_set_drivers_offline_real_new(self, n_drivers_to_off):


        curr_idle_driver_distribution = self.get_observation()[0]
        curr_idle_driver_distribution_resort = np.array([int(curr_idle_driver_distribution.flatten()[index])
                                                         for index in self.target_node_ids])

        # historical idle driver distribution
        idle_driver_distribution = self.idle_driver_location_mat[self.city_time % self.n_intervals, :]

        # diff of curr idle driver distribution and history
        idle_diff = curr_idle_driver_distribution_resort - idle_driver_distribution.astype(int)
        idle_diff[np.where(idle_diff <= 0)] = 0

        n_drivers_can_be_off = int(np.sum(curr_idle_driver_distribution_resort[np.where(idle_diff >= 0)]))
        if n_drivers_to_off > n_drivers_can_be_off:
            n_drivers_to_off = n_drivers_can_be_off

        sum_idle_diff = np.sum(idle_diff)
        if sum_idle_diff == 0:

            return
        np.random.seed(self.RANDOM_SEED)
        node_ids = np.random.choice(self.target_node_ids, size=[n_drivers_to_off],
                                    p=idle_diff / float(sum_idle_diff))

        for ii, node_id in enumerate(node_ids):
            if self.nodes[node_id].idle_driver_num > 0:
                self.nodes[node_id].set_idle_driver_offline_random()
                self.n_drivers -= 1
                self.n_offline_drivers += 1
                n_drivers_to_off -= 1
    # 生成一天的数据，该方方法生成一天的所有订单集合，只在每天0时刻重置环境时使用
    def utility_bootstrap_oneday_order(self):

        num_all_orders = len(self.real_orders)  # 订单总数
        # 从订单中根据概率为28分之1的概率选出该天出现的哪些订单
        # 即每个真实订单有28分支1的可能性出现，index_sampled_orders就是那
        index_sampled_orders = np.where(np.random.binomial(1, self.p, num_all_orders) == 1)
        one_day_orders = self.real_orders[index_sampled_orders]
        self.day_orders_num = len(one_day_orders)
        self.out_grid_in_orders = np.zeros((self.n_intervals, len(self.target_grids)))  # 144×255

        day_orders = [[] for _ in np.arange(self.n_intervals)]  # 144
        for iorder in one_day_orders:
            #  iorder: [92, 300, 143, 2, 13.2]  这个是一个选中的订单
            start_time = int(iorder[2]) # 订单开始时间
            node_mapping_keys = self.node_mapping.keys()    # 有效网格集合
            if iorder[0] not in self.node_mapping.keys() and iorder[1] not in self.node_mapping.keys(): # 如果订单的起始节点和结束节点不是有效的，那么跳过该订单
                continue
            if int(iorder[3])<0:
                continue
            start_node = self.node_mapping.get(iorder[0], -100)     # 开始节点编号（从0开始的）
            end_node = self.node_mapping.get(iorder[1], -100)       # 结束节点编号
            duration = int(iorder[3])                               # 持续时间
            price = iorder[4]                                       # 订单价格


            if start_node == -100:
                column_index = self.target_grids.index(end_node)
                self.out_grid_in_orders[(start_time + duration) % self.n_intervals, column_index] += 1
                continue

            day_orders[start_time].append([start_node, end_node, start_time, duration, price])
        self.day_orders = day_orders    # 一天中每个个时段内的所有订单

    def step_driver_status_control(self):
        # Deal with orders finished at time T=1, check driver status. finish order, set back to off service
        for key, _driver in self.drivers.items():
            _driver.status_control_eachtime(self)
        moment = self.city_time % self.n_intervals
        orders_to_on_drivers = self.out_grid_in_orders[moment, :]
        for idx, item in enumerate(orders_to_on_drivers):
            if item != 0:
                node_id = self.target_grids[idx]
                self.utility_add_driver_real_nodewise(node_id, int(item))

    # 控制结点司机上下线
    def step_driver_online_offline_nodewise(self):
        """ node wise control driver online offline
        :return:
        """
        moment = self.city_time % self.n_intervals  # 时间步
        curr_onoff_distribution = self.onoff_driver_location_mat[moment]    # 当前时刻的上下线概率分布

        self.all_grids_on_number = 0
        self.all_grids_off_number = 0
        for idx, target_node_id in enumerate(self.target_node_ids):
            curr_mu    = curr_onoff_distribution[target_node_id, 0]    # 上线概率
            curr_sigma = curr_onoff_distribution[target_node_id, 1]    # 下线概率
            # 取该正态分布下一个点四舍五入，正数则司机上线，负数则司机下线（即均值大于0更利于上线，小于0利于下线，等于0无影响相当于随机分布）
            on_off_number = np.round(np.random.normal(curr_mu, curr_sigma, 1)[0]).astype(int)# 对该结点决定要上下线司机的个数
            self.on_off_nums[moment - 1][idx] = on_off_number
            if on_off_number > 0:
                self.utility_add_driver_real_nodewise(target_node_id, on_off_number)
                self.all_grids_on_number += on_off_number
            elif on_off_number < 0:
                self.utility_set_drivers_offline_real_nodewise(target_node_id, abs(on_off_number))
            else:
                pass

    def step_driver_online_offline_control_new(self, n_idle_drivers, flag_idle):
        """ control the online offline status of drivers
        控制车辆的上下线
        :param n_idle_drivers: the number of idle drivers expected at current moment
        :return:
        """

        if flag_idle :
            self.utility_add_driver_real_new_offlinefirst(n_idle_drivers, flag_idle)
        else:
            self.utility_add_driver_real_new_offlinefirst(n_idle_drivers, flag_idle)
        # offline_drivers = self.utility_collect_offline_drivers_id()
        # self.n_offline_drivers = len(offline_drivers)
        # #   n_idle_drivers为该时间片目标需要的空闲司机数，self.n_drivers为该时间片已有的空闲司机数
        # if n_idle_drivers > self.n_drivers:
        #     # 如果空闲的司机数大于当前空闲司机数，那么上线需要的空闲司机数，这个值=n_idle_drivers - self.n_drivers
        #     self.utility_add_driver_real_new_offlinefirst(n_idle_drivers - self.n_drivers)
        #
        # elif n_idle_drivers < self.n_drivers:
        #     self.utility_set_drivers_offline_real_new(self.n_drivers - n_idle_drivers)
        # else:
        #     pass

    def step_driver_online_offline_control(self, n_idle_drivers):
        """ control the online offline status of drivers

        :param n_idle_drivers: the number of idle drivers expected at current moment
        :return:
        """

        offline_drivers = self.utility_collect_offline_drivers_id()
        self.n_offline_drivers = len(offline_drivers)
        if n_idle_drivers > self.n_drivers:
            # bring drivers online.
            while self.n_drivers < n_idle_drivers:
                if self.n_offline_drivers > 0:
                    for ii in np.arange(self.n_offline_drivers):
                        self.drivers[offline_drivers[ii]].set_online()
                        self.n_drivers += 1
                        self.n_offline_drivers -= 1
                        if self.n_drivers == n_idle_drivers:
                            break

                self.utility_add_driver_real_new(n_idle_drivers - self.n_drivers)

        elif n_idle_drivers < self.n_drivers:
            self.utility_set_drivers_offline_real_new(self.n_drivers - n_idle_drivers)
        else:
            pass

    # 根据真实数据控制模拟器中的车辆数量，该方法只在每天0时刻重置更新时用到生成司机数
    def utility_get_n_idle_drivers_real(self):
        """ control the number of idle drivers in simulator;
        :return:
        """
        # time = self.city_time % self.n_intervals
        # mean, std = self.idle_driver_dist_time[time]    # 获得当前时刻空闲司机总数的平均值和方差
        # np.random.seed(self.city_time)
        # xx = np.random.normal(mean, std, 1) # 10为均值，1为平均差的正太分布取出一个数取整，为空闲车辆数
        # return np.round(np.random.normal(mean, std, 1)[0]).astype(int)
        time = self.city_time % self.n_intervals
        mean = (self.day_orders_num/(25))*1.5# 平均一个司机25单，1是订单与司机的一个占比
        std = 1.5
        np.random.seed(self.city_time)
        xx = np.random.normal(mean, std, 1) # 10为均值，1为平均差的正太分布取出一个数取整，为空闲车辆数
        return np.round(mean).astype(int)

    def utility_set_neighbor_weight(self, weights):
        self.weights_layers_neighbors = weights

    def step_generate_order_real(self):
        # generate order at t + 1
        for node in self.nodes:
            if node is not None:
                node_id = node.get_node_index()
                # generate orders start from each node
                random_seed = node.get_node_index() + self.city_time
                node.generate_order_real(self.l_max, self.order_time_dist, self.order_price_dist,
                                         self.city_time, self.nodes, random_seed)

    # 把一个时刻的订单分配给所有节点
    def step_bootstrap_order_real(self, day_orders_t):
        for iorder in day_orders_t:
            start_node_id = iorder[0]
            end_node_id = iorder[1]
            start_node = self.nodes[start_node_id]

            # 如果订单目的结点是在目标网络内的话
            if end_node_id in self.target_grids:
                end_node = self.nodes[end_node_id]
            else:
                end_node = None
            start_node.add_order_real(self.city_time, end_node, iorder[3], iorder[4])   # iorder[3]订单结束时间，iorder[4]订单价格

    def step_bootstrap_order_fake(self, day_orders_t):
        for iorder in day_orders_t:
            p_o = np.zeros(255)
            p_o[iorder[0]]+=1

            return p_o

    def step_assign_order(self):

        reward = 0  # R_{t+1}
        all_order_num = 0
        finished_order_num = 0
        for node in self.nodes:
            if node is not None:
                node.remove_unfinished_order(self.city_time)
                reward_node, all_order_num_node, finished_order_num_node = node.simple_order_assign_real(self.city_time, self)
                reward += reward_node
                all_order_num += all_order_num_node
                finished_order_num += finished_order_num_node
        # if all_order_num != 0:
        #     self.order_response_rate = finished_order_num/float(all_order_num)
        # else:
        #     self.order_response_rate = -1
        if all_order_num != 0:
            self.order_response_rate = finished_order_num/float(all_order_num)
        else:
            self.order_response_rate = 0
        return reward

    # 分配订单，最后一步
    def step_assign_order_broadcast_neighbor_reward_update(self):
        """ Consider the orders whose destination or origin is not in the target region
        :param num_layers:
        :param weights_layers_neighbors: [1, 0.5, 0.25, 0.125]
        :return:
        """

        node_reward = np.zeros((len(self.nodes)))
        neighbor_reward = np.zeros((len(self.nodes)))

        # First round broadcast
        reward = 0  # R_{t+1}
        all_order_num = 0
        finished_order_num = 0
        # 第一轮分配订单：
        # 第一轮是将改结点的订单分配给自己网格内的空闲车辆
        for node in self.nodes:
            if node is not None:
                reward_node, all_order_num_node, finished_order_num_node = node.simple_order_assign_real(self.city_time, self)  # 结点奖励，结点所有定单数目，完结订单数目
                reward += reward_node
                all_order_num += all_order_num_node
                finished_order_num += finished_order_num_node
                node_reward[node.get_node_index()] += reward_node

        # Second round broadcast
        # 第二轮分配订单：
        # 第二轮是将该结点的订单分配给自己相邻网格内的空闲车辆
        for idx,node in enumerate(self.nodes):
            if node is not None:
                # 如果这个结点还有订单需要匹配
                if node.order_num != 0:
                    reward_node_broadcast, finished_order_num_node_broadcast \
                        = node.simple_order_assign_broadcast_update(self, neighbor_reward)
                    reward += reward_node_broadcast
                    finished_order_num += finished_order_num_node_broadcast

        node_reward = node_reward + neighbor_reward
        # if all_order_num != 0:
        #     self.order_response_rate = finished_order_num/float(all_order_num)
        # else:
        #     self.order_response_rate = -1
        if all_order_num != 0:
            self.order_response_rate = finished_order_num/float(all_order_num)
        else:
            self.order_response_rate = 0.0    # 如果这个结点没有订单那么不参与计算相应率
        self.finished_order_num+=finished_order_num
        return reward, [node_reward, neighbor_reward]
    def step_assign_order_broadcast_neighbor_reward_update_top(self):
        """ Consider the orders whose destination or origin is not in the target region
        :param num_layers:
        :param weights_layers_neighbors: [1, 0.5, 0.25, 0.125]
        :return:
        """

        node_reward = np.zeros((len(self.nodes)))
        neighbor_reward = np.zeros((len(self.nodes)))
        top = [79,87,108,126,142,157,158,159,176]
        top_num = [0,0,0,0,0,0,0,0,0]
        top_finish = [0,0,0,0,0,0,0,0,0]
        top = np.array(top)
        top_num = np.array(top_num)
        top_finish = np.array(top_finish)
        # First round broadcast
        reward = 0  # R_{t+1}
        all_order_num = 0
        finished_order_num = 0
        # 第一轮分配订单：
        # 第一轮是将改结点的订单分配给自己网格内的空闲车辆
        for node in self.nodes:
            if node is not None:
                idex = node.get_node_index()
                reward_node, all_order_num_node, finished_order_num_node = node.simple_order_assign_real(self.city_time, self)  # 结点奖励，结点所有定单数目，完结订单数目
                reward += reward_node
                all_order_num += all_order_num_node
                finished_order_num += finished_order_num_node
                node_reward[node.get_node_index()] += reward_node
                if idex in top:
                    ti = np.argwhere(top == idex)
                    top_num[ti] += all_order_num_node
                    top_finish[ti] += finished_order_num_node
        # Second round broadcast
        # 第二轮分配订单：
        # 第二轮是将该结点的订单分配给自己相邻网格内的空闲车辆
        for idx,node in enumerate(self.nodes):
            if node is not None:
                idex = node.get_node_index()
                # 如果这个结点还有订单需要匹配
                if node.order_num != 0:
                    reward_node_broadcast, finished_order_num_node_broadcast \
                        = node.simple_order_assign_broadcast_update(self, neighbor_reward)
                    reward += reward_node_broadcast
                    finished_order_num += finished_order_num_node_broadcast
                    if idex in top:
                        ti = [i for i, j in enumerate(top) if j == idex]
                        top_finish[ti] += finished_order_num_node_broadcast
        node_reward = node_reward + neighbor_reward
        # if all_order_num != 0:
        #     self.order_response_rate = finished_order_num/float(all_order_num)
        # else:
        #     self.order_response_rate = -1
        if all_order_num != 0:
            self.order_response_rate = finished_order_num/float(all_order_num)
        else:
            self.order_response_rate = 1.0    # 如果这个结点没有订单那么不参与计算相应率
        self.finished_order_num+=finished_order_num
        top_order = [top_num, top_finish]
        return reward, [node_reward, neighbor_reward], top_order
    # 分配订单，最后一步
    def step_assign_order_broadcast_neighbor_reward_update_future(self):
        """ Consider the orders whose destination or origin is not in the target region
        :param num_layers:
        :param weights_layers_neighbors: [1, 0.5, 0.25, 0.125]
        :return:
        """
        top = [79,87,108,126,142,157,158,159,176]
        top_num = [0,0,0,0,0,0,0,0,0]
        top_finish = [0,0,0,0,0,0,0,0,0]
        top = np.array(top)
        top_num = np.array(top_num)
        top_finish = np.array(top_finish)
        node_reward = np.zeros((len(self.nodes)))
        neighbor_reward = np.zeros((len(self.nodes)))
        rr_1 = np.zeros((144, 232))
        # First round broadcast
        reward = 0  # R_{t+1}
        all_order_num = 0
        finished_order_num = 0
        # 第一轮分配订单：
        # 第一轮是将改结点的订单分配给自己网格内的空闲车辆
        for node in self.nodes:
            if node is not None:
                idex = node.get_node_index()
                reward_1, reward_2, reward_node, all_order_num_node, finished_order_num_node = node.simple_order_assign_real_future(self.city_time, self)  # 结点奖励，结点所有定单数目，完结订单数目
                reward += reward_node
                all_order_num += all_order_num_node
                finished_order_num += finished_order_num_node
                node_reward[node.get_node_index()] += reward_2
                rr_1 += reward_1
                if idex in top:
                    ti = np.argwhere(top == idex)
                    top_num[ti] += all_order_num_node
                    top_finish[ti] += finished_order_num_node
        # Second round broadcast
        # 第二轮分配订单：
        # 第二轮是将该结点的订单分配给自己相邻网格内的空闲车辆
        for idx, node in enumerate(self.nodes):
            if node is not None:
                idex = node.get_node_index()
                # 如果这个结点还有订单需要匹配
                if node.order_num != 0:
                    reward_1, reward_2, reward_node_broadcast, finished_order_num_node_broadcast \
                        = node.simple_order_assign_broadcast_update_future(self, neighbor_reward)
                    reward += reward_node_broadcast
                    finished_order_num += finished_order_num_node_broadcast
                    rr_1 += reward_1
                    if idex in top:
                        ti = [i for i, j in enumerate(top) if j == idex]
                        top_finish[ti] += finished_order_num_node_broadcast

        node_reward = node_reward + neighbor_reward
        # if all_order_num != 0:
        #     self.order_response_rate = finished_order_num/float(all_order_num)
        # else:
        #     self.order_response_rate = -1
        if all_order_num != 0:
            self.order_response_rate = finished_order_num/float(all_order_num)
        else:
            self.order_response_rate = 1.0    # 如果这个结点没有订单那么不参与计算相应率

        top_order = [top_num, top_finish]

        return reward, [node_reward, neighbor_reward], rr_1, top_order

    def step_remove_unfinished_orders(self):
        for node in self.nodes:
            if node is not None:
                node.remove_unfinished_order(self.city_time)

    # 相当于初分配？就是初分配，这次分配返回的是个state，奖励，车辆状态啥都没改
    def step_pre_order_assigin(self, next_state):

        remain_drivers = next_state[0] - next_state[1]  # idle - order
        remain_drivers[remain_drivers < 0] = 0  # 每个结点剩下的司机数，

        remain_orders = next_state[1] - next_state[0]   # order - idle
        remain_orders[remain_orders < 0] = 0    # 每个结点剩下的订单数

        if np.sum(remain_orders) == 0 or np.sum(remain_drivers) == 0:   # 如果所有结点的司机数都大于订单 or 所有结点的订单数都大于司机
            context = np.array([remain_drivers, remain_orders])         # 返回初分配后的state
            return context

        remain_orders_1d = remain_orders.flatten()
        remain_drivers_1d = remain_drivers.flatten()

        # 如果两边都有剩余，那么就往邻居扩散
        for node in self.nodes:
            if node is not None:
                curr_node_id = node.get_node_index()
                if remain_orders_1d[curr_node_id] != 0:
                    for neighbor_node in node.neighbors:
                        if neighbor_node is not None:
                            neighbor_id = neighbor_node.get_node_index()
                            a = remain_orders_1d[curr_node_id]
                            b = remain_drivers_1d[neighbor_id]
                            remain_orders_1d[curr_node_id] = max(a-b, 0)
                            remain_drivers_1d[neighbor_id] = max(b-a, 0)
                        if remain_orders_1d[curr_node_id] == 0:
                            break

        context = np.array([remain_drivers_1d.reshape(self.M, self.N),
                   remain_orders_1d.reshape(self.M, self.N)])
        return context
    def pre_order_assigin(self, next_state):

        remain_drivers = next_state[0] - next_state[1]  # idle - order
        remain_drivers[remain_drivers < 0] = 0  # 每个结点剩下的司机数，

        remain_orders = next_state[1] - next_state[0]   # order - idle
        remain_orders[remain_orders < 0] = 0    # 每个结点剩下的订单数

        if np.sum(remain_orders) == 0 or np.sum(remain_drivers) == 0:   # 如果所有结点的司机数都大于订单 or 所有结点的订单数都大于司机
            context = np.array([remain_drivers, remain_orders])         # 返回初分配后的state
            return context

        remain_orders_1d = remain_orders.flatten()
        remain_drivers_1d = remain_drivers.flatten()

        order = np.zeros(self.n_valid_grids)
        driver = np.zeros(self.n_valid_grids)

        # 如果两边都有剩余，那么就往邻居扩散
        for node in self.nodes:
            if node is not None:
                curr_node_id = node.get_node_index()
                if remain_orders_1d[curr_node_id] != 0:
                    for neighbor_node in node.neighbors:
                        if neighbor_node is not None:
                            neighbor_id = neighbor_node.get_node_index()
                            a = remain_orders_1d[curr_node_id]
                            b = remain_drivers_1d[neighbor_id]
                            remain_orders_1d[curr_node_id] = max(a-b, 0)
                            remain_drivers_1d[neighbor_id] = max(b-a, 0)
                        if remain_orders_1d[curr_node_id] == 0:
                            break

        driver = remain_drivers_1d[self.target_grids]
        order = remain_orders_1d[self.target_grids]
        driver = np.reshape(driver,(1,self.n_valid_grids,1))
        order = np.reshape(order,(1,self.n_valid_grids,1))
        context = np.array([driver, order])
        return context

    def step_dispatch_invalid(self, dispatch_actions):
        """ If a
        :param dispatch_actions:
        :return:
        """
        save_remove_id = []
        for action in dispatch_actions:

            # action包括调度起始结点，目标结点，执行调度的司机数目
            start_node_id, end_node_id, num_of_drivers = action
            start_node = self.nodes[start_node_id]
            end_node = self.nodes[end_node_id]
            if self.nodes[start_node_id] is None or num_of_drivers == 0:
                continue  # not a feasible action
            # 如果起始节点没有足够的司机执行调度，那么就该节点所有司机执行
            if self.nodes[start_node_id].get_driver_numbers() < num_of_drivers:
                num_of_drivers = self.nodes[start_node_id].get_driver_numbers()
            # 如果目标结点是非法节点，
            if end_node_id < 0:
                for _ in np.arange(num_of_drivers):
                    self.nodes[start_node_id].set_idle_driver_offline_random()
                    self.n_drivers -= 1
                    self.n_offline_drivers += 1
                    self.all_grids_off_number += 1
                continue

            # 如果目标结点是None
            if self.nodes[end_node_id] is None:
                for _ in np.arange(num_of_drivers):
                    self.nodes[start_node_id].set_idle_driver_offline_random()
                    self.n_drivers -= 1
                    self.n_offline_drivers += 1
                    self.all_grids_off_number += 1
                continue

            # 如果目标结点不在开始结点的邻居里
            if self.nodes[end_node_id] not in self.nodes[start_node_id].neighbors:

                a=self.nodes[end_node_id]
                bb = self.nodes[start_node_id]
                b=self.nodes[start_node_id].neighbors
                c=end_node_id
                d=start_node_id
                raise ValueError('City:step(): not a feasible dispatch')


            for _ in np.arange(num_of_drivers):
                # t = 1 dispatch start, idle driver decrease
                remove_driver_id = self.nodes[start_node_id].remove_idle_driver_random()    # 随机选取一位司机
                save_remove_id.append((end_node_id, remove_driver_id))
                self.drivers[remove_driver_id].set_position(None)
                self.drivers[remove_driver_id].set_offline_for_start_dispatch() # 把这个司机下线？？？？为什么？！！！
                self.n_drivers -= 1

        return save_remove_id

    def step_add_dispatched_drivers(self, save_remove_id):
        # drivers dispatched at t, arrived at t + 1
        for destination_node_id, arrive_driver_id in save_remove_id:
            self.drivers[arrive_driver_id].set_position(self.nodes[destination_node_id])
            self.drivers[arrive_driver_id].set_online_for_finish_dispatch()
            self.nodes[destination_node_id].add_driver(arrive_driver_id, self.drivers[arrive_driver_id])
            self.n_drivers += 1

    def step_increase_city_time(self):
        self.city_time += 1
        # set city time of drivers
        for driver_id, driver in self.drivers.items():
            driver.set_city_time(self.city_time)

    def step(self, dispatch_actions, generate_order=1):
        info = []
        '''**************************** T = 1 ****************************'''
        # Loop over all dispatch action, change the driver distribution
        # 执行action，改变司机的位置分布

        save_remove_id = self.step_dispatch_invalid(dispatch_actions)   # （动作目标结点，司机id），这些司机都被设为离线，且position=None
        # When the drivers go to invalid grid, set them offline.
        # 分配订单（Dispatch orders）返回奖励
        # reward为这一次分配订单
        reward, reward_node = self.step_assign_order_broadcast_neighbor_reward_update()

        '''**************************** T = 2 ****************************'''
        # increase city time t + 1
        # 所有司机的时间加1
        self.step_increase_city_time()
        # 完成订单的司机重新变回空闲司机
        self.step_driver_status_control()  # drivers finish order become available again.

        # drivers dispatched at t, arrived at t + 1, become available at t+1
        # 重定位司机，将t时刻的司机位置更新成t+1时刻司机的位置
        self.step_add_dispatched_drivers(save_remove_id)

        # generate order at t + 1
        # generate_order=1就代表使用模拟订单，否则使用真实订单，我们只用到了真实订单
        if generate_order == 1:
            self.step_generate_order_real() # 模拟订单
        else:
            moment = self.city_time % self.n_intervals
            self.step_bootstrap_order_real(self.day_orders[moment])
        
        # offline online control;
        # 司机的上下线状态控制，通过正态分布决定
        self.step_driver_online_offline_nodewise()
        # 移除那些未完成的订单
        self.step_remove_unfinished_orders()
        # state包括两维，每一维都有结点的二维坐标，两个维度分布代表司机数量和订单数量
        # get states S_{t+1}  [driver_dist, order_dist]
        next_state = self.get_observation()

        context = self.step_pre_order_assigin(next_state)
        info = [reward_node, context]
        return next_state, reward, info
  #输出下一个状态  得到奖励

    ### MINE ###
    def ma_step(self, dispatch_actions, generate_order=1):
        info = []
        '''**************************** T = 1 ****************************'''
        # Loop over all dispatch action, change the driver distribution
        # 执行action，改变司机的位置分布

        save_remove_id = self.step_dispatch_invalid(dispatch_actions)  # （动作目标结点，司机id），这些司机都被设为离线，且position=None
        # When the drivers go to invalid grid, set them offline.
        # 分配订单（Dispatch orders）返回奖励
        # reward为这一次分配订单
        reward, reward_node = self.step_assign_order_broadcast_neighbor_reward_update()

        '''**************************** T = 2 ****************************'''
        # increase city time t + 1
        # 所有司机的时间加1
        self.step_increase_city_time()
        # 完成订单的司机重新变回空闲司机
        self.step_driver_status_control()  # drivers finish order become available again.

        # drivers dispatched at t, arrived at t + 1, become available at t+1
        # 重定位司机，将t时刻的司机位置更新成t+1时刻司机的位置
        self.step_add_dispatched_drivers(save_remove_id)

        # generate order at t + 1
        # generate_order=1就代表使用模拟订单，否则使用真实订单，我们只用到了真实订单
        if generate_order == 1:
            self.step_generate_order_real()  # 模拟订单
        else:
            moment = self.city_time % self.n_intervals
            self.step_bootstrap_order_real(self.day_orders[moment])

        # offline online control;
        # 司机的上下线状态控制，通过正态分布决定
        self.step_driver_online_offline_nodewise()
        # 移除那些未完成的订单
        self.step_remove_unfinished_orders()
        # state包括两维，每一维都有结点的二维坐标，两个维度分布代表司机数量和订单数量
        # get states S_{t+1}  [driver_dist, order_dist]
        next_o_state = self.get_order_distribution()
        next_d_state = self.get_driver_distribution()
        next_state = self.get_observation()
        context = self.step_pre_order_assigin(next_state)
        info = [reward_node, context]
        return next_o_state, next_d_state, next_state, reward, info

    def get_order_distribution(self):
        next_state = np.zeros((1, self.M, self.N))
        for _node in self.nodes:
            if _node is not None:
                row_id, column_id = ids_1dto2d(_node.get_node_index(), self.M, self.N)
                next_state[0, row_id, column_id] = _node.order_num

        return next_state

    def get_driver_distribution(self):
        next_state = np.zeros((1, self.M, self.N))
        for _node in self.nodes:
            if _node is not None:
                row_id, column_id = ids_1dto2d(_node.get_node_index(), self.M, self.N)
                next_state[0, row_id, column_id] = _node.idle_driver_num

        return next_state

    def utility_bootstrap_old_order(self, old_orders, p):

        num_all_orders = len(old_orders)  # 订单总数
        # 从订单中根据概率为28分之1的概率选出该天出现的哪些订单
        # 即每个真实订单有28分支1的可能性出现，index_sampled_orders就是那
        index_sampled_orders = np.where(np.random.binomial(1, p, num_all_orders) == 1)
        one_day_orders = old_orders[index_sampled_orders]
        day_orders_num = len(one_day_orders)
        out_grid_in_orders = np.zeros((self.n_intervals, len(self.target_grids)))  # 144×255

        day_orders = [[] for _ in np.arange(self.n_intervals)]  # 144
        for iorder in one_day_orders:
            #  iorder: [92, 300, 143, 2, 13.2]  这个是一个选中的订单
            start_time = int(iorder[2]) # 订单开始时间
            node_mapping_keys = self.node_mapping.keys()    # 有效网格集合
            if iorder[0] not in self.node_mapping.keys() and iorder[1] not in self.node_mapping.keys(): # 如果订单的起始节点和结束节点不是有效的，那么跳过该订单
                continue
            start_node = self.node_mapping.get(iorder[0], -100)     # 开始节点编号（从0开始的）
            end_node = self.node_mapping.get(iorder[1], -100)       # 结束节点编号
            duration = int(iorder[3])                               # 持续时间
            price = iorder[4]                                       # 订单价格


            if start_node == -100:
                # column_index = self.target_grids.index(end_node)
                # self.out_grid_in_orders[(start_time + duration) % self.n_intervals, column_index] += 1
                continue

            day_orders[start_time].append([start_node, end_node, start_time, duration, price])
        return day_orders    # 一天中每个个时段内的所有订单

    def step_bootstrap_old_order_real(self, day_orders_t):
        mat = np.zeros(shape=(1, 15, 17))

        for iorder in day_orders_t:
            start_node_id = iorder[0]
            end_node_id = iorder[1]
            start_i, start_j = ids_1dto2d(iorder[0], 15, 17)
            end_i, end_j = ids_1dto2d(iorder[1], 15, 17)
            mat[0, start_i, start_j] += 1.0
            mat[0, end_i, end_j] += 0.1

        return mat

    def test_step_driver_online_offline_nodewise(self, on_off_nums):
        """ node wise control driver online offline
        :return:
        """
        moment = self.city_time % self.n_intervals  # 时间步
        curr_onoff_distribution = self.onoff_driver_location_mat[moment]    # 当前时刻的上下线概率分布

        self.all_grids_on_number = 0
        self.all_grids_off_number = 0
        for idx, target_node_id in enumerate(self.target_node_ids):
            on_off_number = on_off_nums[moment-1][idx]
            # self.on_off_nums[moment - 1][idx] = on_off_number
            if on_off_number > 0:
                self.utility_add_driver_real_nodewise(target_node_id, on_off_number)
                self.all_grids_on_number += on_off_number
            elif on_off_number < 0:
                self.utility_set_drivers_offline_real_nodewise(target_node_id, abs(on_off_number))
            else:
                pass

    def test_ma_step(self, dispatch_actions, on_off_nums, generate_order=1):
        info = []
        '''**************************** T = 1 ****************************'''
        # Loop over all dispatch action, change the driver distribution
        # 执行action，改变司机的位置分布

        save_remove_id = self.step_dispatch_invalid(dispatch_actions)  # （动作目标结点，司机id），这些司机都被设为离线，且position=None
        # When the drivers go to invalid grid, set them offline.

        # 分配订单（Dispatch orders）返回奖励
        # reward为这一次分配订单
        reward, reward_node = self.step_assign_order_broadcast_neighbor_reward_update()

        '''**************************** T = 2 ****************************'''
        # increase city time t + 1
        # 所有司机的时间加1
        self.step_increase_city_time()
        # 完成订单的司机重新变回空闲司机
        self.step_driver_status_control()  # drivers finish order become available again.

        # drivers dispatched at t, arrived at t + 1, become available at t+1
        # 重定位司机，将t时刻的司机位置更新成t+1时刻司机的位置
        self.step_add_dispatched_drivers(save_remove_id)

        # generate order at t + 1
        # generate_order=1就代表使用模拟订单，否则使用真实订单，我们只用到了真实订单
        if generate_order == 1:
            self.step_generate_order_real()  # 模拟订单
        else:
            moment = self.city_time % self.n_intervals
            self.step_bootstrap_order_real(self.day_orders[moment])

        # offline online control;
        # 司机的上下线状态控制，通过正态分布决定
        self.test_step_driver_online_offline_nodewise(on_off_nums)
        # 移除那些未完成的订单
        self.step_remove_unfinished_orders()
        # state包括两维，每一维都有结点的二维坐标，两个维度分布代表司机数量和订单数量
        # get states S_{t+1}  [driver_dist, order_dist]
        next_o_state = self.get_order_distribution()
        next_d_state = self.get_driver_distribution()
        next_state = self.get_observation()
        context = self.step_pre_order_assigin(next_state)
        info = [reward_node, context]
        return next_o_state, next_d_state, next_state, reward, info

    def test_ma_step_top(self, dispatch_actions, on_off_nums, generate_order=1):
        info = []
        '''**************************** T = 1 ****************************'''
        # Loop over all dispatch action, change the driver distribution
        # 执行action，改变司机的位置分布

        save_remove_id = self.step_dispatch_invalid(dispatch_actions)  # （动作目标结点，司机id），这些司机都被设为离线，且position=None
        # When the drivers go to invalid grid, set them offline.

        # 分配订单（Dispatch orders）返回奖励
        # reward为这一次分配订单
        reward, reward_node, top_order = self.step_assign_order_broadcast_neighbor_reward_update_top()

        '''**************************** T = 2 ****************************'''
        # increase city time t + 1
        # 所有司机的时间加1
        self.step_increase_city_time()
        # 完成订单的司机重新变回空闲司机
        self.step_driver_status_control()  # drivers finish order become available again.

        # drivers dispatched at t, arrived at t + 1, become available at t+1
        # 重定位司机，将t时刻的司机位置更新成t+1时刻司机的位置
        self.step_add_dispatched_drivers(save_remove_id)

        # generate order at t + 1
        # generate_order=1就代表使用模拟订单，否则使用真实订单，我们只用到了真实订单
        if generate_order == 1:
            self.step_generate_order_real()  # 模拟订单
        else:
            moment = self.city_time % self.n_intervals
            self.step_bootstrap_order_real(self.day_orders[moment])

        # offline online control;
        # 司机的上下线状态控制，通过正态分布决定
        self.test_step_driver_online_offline_nodewise(on_off_nums)
        # 移除那些未完成的订单
        self.step_remove_unfinished_orders()
        # state包括两维，每一维都有结点的二维坐标，两个维度分布代表司机数量和订单数量
        # get states S_{t+1}  [driver_dist, order_dist]
        next_o_state = self.get_order_distribution()
        next_d_state = self.get_driver_distribution()
        next_state = self.get_observation()
        context = self.step_pre_order_assigin(next_state)
        info = [reward_node, context]
        return next_o_state, next_d_state, next_state, reward, info, top_order

    def test_hma_step(self, dispatch_actions, future, on_off_nums, generate_order=1):
        info = []
        '''**************************** T = 1 ****************************'''
        # Loop over all dispatch action, change the driver distribution
        # 执行action，改变司机的位置分布

        save_remove_id = self.step_dispatch_invalid(dispatch_actions)  # （动作目标结点，司机id），这些司机都被设为离线，且position=None
        # When the drivers go to invalid grid, set them offline.
        # if self.city_time < 144:
        #     for i in range(255):
        #         if future_online_1[i][self.city_time] > 0:
        #             self.utility_add_driver_real_future(i, future_online_1[i][self.city_time])

        if self.city_time < 144:
            target, start = np.nonzero(future)
            length = len(target)
            if length > 0:
                for i in range(length):
                    t = target[i]
                    s = start[i]
                    num = int(future[t][s])
                    start_time = round((future[t][s] - num)*1000)
                    self.utility_add_driver_real_future_future(t, num, start_time, s)


        # 分配订单（Dispatch orders）返回奖励
        # reward为这一次分配订单

        reward, reward_node, reward_1, top_order = self.step_assign_order_broadcast_neighbor_reward_update_future()
        '''**************************** T = 2 ****************************'''
        # increase city time t + 1
        # 所有司机的时间加1
        self.step_increase_city_time()
        # 完成订单的司机重新变回空闲司机
        self.step_driver_status_control()  # drivers finish order become available again.

        # drivers dispatched at t, arrived at t + 1, become available at t+1
        # 重定位司机，将t时刻的司机位置更新成t+1时刻司机的位置
        self.step_add_dispatched_drivers(save_remove_id)

        # generate order at t + 1
        # generate_order=1就代表使用模拟订单，否则使用真实订单，我们只用到了真实订单
        if generate_order == 1:
            self.step_generate_order_real()  # 模拟订单
        else:
            moment = self.city_time % self.n_intervals
            self.step_bootstrap_order_real(self.day_orders[moment])

        # offline online control;
        # 司机的上下线状态控制，通过正态分布决定
        self.test_step_driver_online_offline_nodewise(on_off_nums)
        # 移除那些未完成的订单
        self.step_remove_unfinished_orders()
        # state包括两维，每一维都有结点的二维坐标，两个维度分布代表司机数量和订单数量
        # get states S_{t+1}  [driver_dist, order_dist]
        next_state = self.get_observation()
        next_high = self.get_high_distribution()
        context = self.step_pre_order_assigin(next_state)
        con = self.pre_order_assigin(next_state)
        next_o_state = con[1]
        next_d_state = con[0]
        info = [reward_node, context]

        return next_o_state, next_d_state, next_state, next_high, reward, info, reward_1, top_order

    def test_step(self, dispatch_actions, on_off_nums, generate_order=1):
        info = []
        '''**************************** T = 1 ****************************'''
        # Loop over all dispatch action, change the driver distribution
        # 执行action，改变司机的位置分布

        save_remove_id = self.step_dispatch_invalid(dispatch_actions)  # （动作目标结点，司机id），这些司机都被设为离线，且position=None
        # When the drivers go to invalid grid, set them offline.


        # 分配订单（Dispatch orders）返回奖励
        # reward为这一次分配订单
        reward, reward_node = self.step_assign_order_broadcast_neighbor_reward_update()

        '''**************************** T = 2 ****************************'''
        # increase city time t + 1
        # 所有司机的时间加1
        self.step_increase_city_time()
        # 完成订单的司机重新变回空闲司机
        self.step_driver_status_control()  # drivers finish order become available again.

        # drivers dispatched at t, arrived at t + 1, become available at t+1
        # 重定位司机，将t时刻的司机位置更新成t+1时刻司机的位置
        self.step_add_dispatched_drivers(save_remove_id)

        # generate order at t + 1
        # generate_order=1就代表使用模拟订单，否则使用真实订单，我们只用到了真实订单
        if generate_order == 1:
            self.step_generate_order_real()  # 模拟订单
        else:
            moment = self.city_time % self.n_intervals
            self.step_bootstrap_order_real(self.day_orders[moment])

        # offline online control;
        # 司机的上下线状态控制，通过正态分布决定
        self.test_step_driver_online_offline_nodewise(on_off_nums)
        # 移除那些未完成的订单
        self.step_remove_unfinished_orders()
        # state包括两维，每一维都有结点的二维坐标，两个维度分布代表司机数量和订单数量
        # get states S_{t+1}  [driver_dist, order_dist]
        next_state = self.get_observation()

        context = self.step_pre_order_assigin(next_state)
        info = [reward_node, context]
        return next_state, reward, info

    def test_step_top(self, dispatch_actions, on_off_nums, generate_order=1):
        info = []
        '''**************************** T = 1 ****************************'''
        # Loop over all dispatch action, change the driver distribution
        # 执行action，改变司机的位置分布

        save_remove_id = self.step_dispatch_invalid(dispatch_actions)  # （动作目标结点，司机id），这些司机都被设为离线，且position=None
        # When the drivers go to invalid grid, set them offline.


        # 分配订单（Dispatch orders）返回奖励
        # reward为这一次分配订单
        reward, reward_node, top_order = self.step_assign_order_broadcast_neighbor_reward_update_top()

        '''**************************** T = 2 ****************************'''
        # increase city time t + 1
        # 所有司机的时间加1
        self.step_increase_city_time()
        # 完成订单的司机重新变回空闲司机
        self.step_driver_status_control()  # drivers finish order become available again.

        # drivers dispatched at t, arrived at t + 1, become available at t+1
        # 重定位司机，将t时刻的司机位置更新成t+1时刻司机的位置
        self.step_add_dispatched_drivers(save_remove_id)

        # generate order at t + 1
        # generate_order=1就代表使用模拟订单，否则使用真实订单，我们只用到了真实订单
        if generate_order == 1:
            self.step_generate_order_real()  # 模拟订单
        else:
            moment = self.city_time % self.n_intervals
            self.step_bootstrap_order_real(self.day_orders[moment])

        # offline online control;
        # 司机的上下线状态控制，通过正态分布决定
        self.test_step_driver_online_offline_nodewise(on_off_nums)
        # 移除那些未完成的订单
        self.step_remove_unfinished_orders()
        # state包括两维，每一维都有结点的二维坐标，两个维度分布代表司机数量和订单数量
        # get states S_{t+1}  [driver_dist, order_dist]
        next_state = self.get_observation()

        context = self.step_pre_order_assigin(next_state)
        info = [reward_node, context]
        return next_state, reward, info, top_order

    def ha_step(self, dispatch_actions, generate_order=1):
        info = []
        '''**************************** T = 1 ****************************'''
        # Loop over all dispatch action, change the driver distribution
        # 执行action，改变司机的位置分布

        save_remove_id = self.step_dispatch_invalid(dispatch_actions)  # （动作目标结点，司机id），这些司机都被设为离线，且position=None
        # When the drivers go to invalid grid, set them offline.
        # 分配订单（Dispatch orders）返回奖励
        # reward为这一次分配订单
        reward, reward_node = self.step_assign_order_broadcast_neighbor_reward_update()

        '''**************************** T = 2 ****************************'''
        # increase city time t + 1
        # 所有司机的时间加1
        self.step_increase_city_time()
        # 完成订单的司机重新变回空闲司机
        self.step_driver_status_control()  # drivers finish order become available again.

        # drivers dispatched at t, arrived at t + 1, become available at t+1
        # 重定位司机，将t时刻的司机位置更新成t+1时刻司机的位置
        self.step_add_dispatched_drivers(save_remove_id)

        # generate order at t + 1
        # generate_order=1就代表使用模拟订单，否则使用真实订单，我们只用到了真实订单
        if generate_order == 1:
            self.step_generate_order_real()  # 模拟订单
        else:
            moment = self.city_time % self.n_intervals
            self.step_bootstrap_order_real(self.day_orders[moment])

        # offline online control;
        # 司机的上下线状态控制，通过正态分布决定
        self.step_driver_online_offline_nodewise()
        # 移除那些未完成的订单
        self.step_remove_unfinished_orders()
        # state包括两维，每一维都有结点的二维坐标，两个维度分布代表司机数量和订单数量
        # get states S_{t+1}  [driver_dist, order_dist]
        next_o_state = self.get_order_distribution()
        next_d_state = self.get_driver_distribution()
        next_local_state = self.get_local_distribution()
        next_state = self.get_observation()
        context = self.step_pre_order_assigin(next_state)
        info = [reward_node, context]
        return next_o_state, next_d_state, next_state, next_local_state, reward, info

    def hma_step(self, dispatch_actions, future_online_1, generate_order=1):
        info = []
        '''**************************** T = 1 ****************************'''
        # Loop over all dispatch action, change the driver distribution
        # 执行action，改变司机的位置分布

        save_remove_id = self.step_dispatch_invalid(dispatch_actions)  # （动作目标结点，司机id），这些司机都被设为离线，且position=None
        # When the drivers go to invalid grid, set them offline.
        if self.city_time < 144:
            for i in range(255):
                if future_online_1[i][self.city_time] > 0:
                    self.utility_add_driver_real_future(i, future_online_1[i][self.city_time])
        # 分配订单（Dispatch orders）返回奖励
        # reward为这一次分配订单
        reward, reward_node = self.step_assign_order_broadcast_neighbor_reward_update()

        '''**************************** T = 2 ****************************'''
        # increase city time t + 1
        # 所有司机的时间加1
        self.step_increase_city_time()
        # 完成订单的司机重新变回空闲司机
        self.step_driver_status_control()  # drivers finish order become available again.

        # drivers dispatched at t, arrived at t + 1, become available at t+1
        # 重定位司机，将t时刻的司机位置更新成t+1时刻司机的位置
        self.step_add_dispatched_drivers(save_remove_id)

        # generate order at t + 1
        # generate_order=1就代表使用模拟订单，否则使用真实订单，我们只用到了真实订单
        if generate_order == 1:
            self.step_generate_order_real()  # 模拟订单
        else:
            moment = self.city_time % self.n_intervals
            self.step_bootstrap_order_real(self.day_orders[moment])

        # offline online control;
        # 司机的上下线状态控制，通过正态分布决定
        self.step_driver_online_offline_nodewise()
        # 移除那些未完成的订单
        self.step_remove_unfinished_orders()
        # state包括两维，每一维都有结点的二维坐标，两个维度分布代表司机数量和订单数量
        # get states S_{t+1}  [driver_dist, order_dist]

        next_state = self.get_observation()
        next_high = self.get_high_distribution()
        context = self.step_pre_order_assigin(next_state)
        con = self.pre_order_assigin(next_state)
        next_o_state = con[1]
        next_d_state = con[0]
        info = [reward_node, context]
        return next_o_state, next_d_state, next_state, next_high, reward, info

    def get_local_distribution(self):
        grid_ids = self.target_node_ids
        local_distribution = np.zeros((1, 232, 7))
        counts = 0
        for idx in range(232):
            valid_qvalues = []
            for grid_valid_idx in self.neighbors_list[idx]:
                value = (self.nodes[grid_ids[grid_valid_idx]].idle_driver_num) - (self.nodes[grid_ids[grid_valid_idx]].order_num)
                valid_qvalues.append(value)
            local_distribution[0, idx, self.valid_action_mask[idx] > 0] = valid_qvalues

        # local_distribution = np.delete(local_distribution, 6, axis=2)
        return local_distribution

    def get_high_distribution(self):
        high_distribution = np.zeros((1, 232),dtype='float32')
        grid_ids = self.target_node_ids

        for idx in range(232):
            high_distribution[0, idx] = (self.nodes[grid_ids[idx]].idle_driver_num) - (
                self.nodes[grid_ids[idx]].order_num)
            # if high_distribution[0, idx]>0:
            #     high_distribution[0, idx]=1.000
            # else:
            #     high_distribution[0, idx]=-1.000

        # high_distribution = high_distribution.astype(float)
        return high_distribution

    def get_online_driver_hma(self):
        nums = 0
        for key in self.drivers:
            if self.drivers[key].online is True:
                nums += 1

        return nums

    # def get_q_reward(self, valid_prob, next_o_state, next_o_o_state, next_d_state, node_reward, gamma):
    #     targets = []
    #
    #     for idx in np.arange(self.n_valid_grid):
    #         grid_prob = valid_prob[0][idx][self.valid_action_mask[idx]>0]
    #         neighbor_grid_ids = self.neighbors_list[idx]
    #         best_grid = np.argmax(grid_prob)
    #         curr_grid_target = node_reward[neighbor_grid_ids][best_grid] + gamma * qvalue_next[neighbor_grid_ids][best_grid]
    #         targets.append(curr_grid_target)
    #
    #     return np.array(targets).reshape([-1, 232])

    #   new

    def new_step(self, dispatch_actions, on_off_nums, generate_order):
        info = []
        '''**************************** T = 1 ****************************'''
        # Loop over all dispatch action, change the driver distribution
        # 执行action，改变司机的位置分布

        save_remove_id = self.step_dispatch_invalid(dispatch_actions)  # （动作目标结点，司机id），这些司机都被设为离线，且position=None
        # When the drivers go to invalid grid, set them offline.
        # 分配订单（Dispatch orders）返回奖励
        # reward为这一次分配订单
        reward, reward_node = self.step_assign_order_broadcast_neighbor_reward_update()

        '''**************************** T = 2 ****************************'''
        # increase city time t + 1
        # 所有司机的时间加1
        self.step_increase_city_time()
        # 完成订单的司机重新变回空闲司机
        self.step_driver_status_control()  # drivers finish order become available again.

        # drivers dispatched at t, arrived at t + 1, become available at t+1
        # 重定位司机，将t时刻的司机位置更新成t+1时刻司机的位置
        self.step_add_dispatched_drivers(save_remove_id)

        # generate order at t + 1
        # generate_order=1就代表使用模拟订单，否则使用真实订单，我们只用到了真实订单
        if generate_order == 1:
            self.step_generate_order_real()  # 模拟订单
        else:
            moment = self.city_time % self.n_intervals
            self.step_bootstrap_order_real(self.day_orders[moment])

        # offline online control;
        # 司机的上下线状态控制，通过正态分布决定
        self.step_driver_online_offline_nodewise()
        # 移除那些未完成的订单
        self.step_remove_unfinished_orders()
        # state包括两维，每一维都有结点的二维坐标，两个维度分布代表司机数量和订单数量
        # get states S_{t+1}  [driver_dist, order_dist]
        next_state = self.get_observation()

        context = self.step_pre_order_assigin(next_state)
        info = [reward_node, context]
        return next_state, reward, info

    def reset_clean_new(self, generate_order=1, ratio=1, city_time=""):
            """ 1. bootstrap oneday's order data.  引导一天的订单数据。
                2. clean current drivers and orders, regenerate new orders and drivers.
                can reset anytime  清理当前驱动程序和订单，重新生成新的订单和驱动程序。
    可以随时重置
            :return:
            """
            if city_time != "":
                self.city_time = city_time

            # clean orders and drivers
            self.drivers = {}  # driver[driver_id] = driver_instance  , driver_id start from 0
            self.n_drivers = 0  # total idle number of drivers. online and not on service.
            self.n_offline_drivers = 0  # total number of offline drivers.
            # 把所有节点的订单，车辆啥的全都清空
            for node in self.nodes:
                if node is not None:
                    node.clean_node()
            if city_time == 0:
                asd = 1
            # Generate one day's order.
            # 会根据真实订单有概率的选取出其中一部分来生成
            if generate_order == 1:
                node_feature = self.utility_bootstrap_oneday_order_new()  # 生成一天的订单，结果存储到env.day_orders,144个[]，每个[]有当天的所有订单

            # Init orders of current time step   当前时间步长的初始顺序
            moment = self.city_time % self.n_intervals
            self.step_bootstrap_order_real(self.day_orders[moment])  # 把0时刻的订单分配给所有节点

            # Init current driver distribution  初始化空闲司机分配，该方法只在每天0时刻用到了
            if self.global_flag == "global":
                num_driver = self.utility_get_n_idle_drivers_real()
                num_driver = int(num_driver * ratio)
            else:
                num_driver = self.utility_get_n_idle_drivers_nodewise()
            num_idle_driver = int(num_driver * 0.15)
            num_off_line_driver = num_driver - num_idle_driver
            self.step_driver_online_offline_control_new(num_idle_driver, True)
            self.step_driver_online_offline_control_new(num_off_line_driver, False)
            return self.get_observation(), node_feature,

    def utility_bootstrap_oneday_order_new(self):
        node_feature = np.zeros((145, self.n_valid_grids, 3))   # 该网格作为目的地的次数、该网格内订单价值总和、该网格内订单价值时长总和
        num_all_orders = len(self.real_orders)  # 订单总数
        # 从订单中根据概率为28分之1的概率选出该天出现的哪些订单
        # 即每个真实订单有28分支1的可能性出现，index_sampled_orders就是那
        index_sampled_orders = np.where(np.random.binomial(1, self.p, num_all_orders) == 1)
        one_day_orders = self.real_orders[index_sampled_orders]
        self.day_orders_num = len(one_day_orders)
        self.out_grid_in_orders = np.zeros((self.n_intervals, len(self.target_grids)))  # 144×255

        day_orders = [[] for _ in np.arange(self.n_intervals)]  # 144
        for iorder in one_day_orders:
            #  iorder: [92, 300, 143, 2, 13.2]  这个是一个选中的订单
            start_time = int(iorder[2]) # 订单开始时间
            # node_mapping_keys = self.node_mapping.keys()    # 有效网格集合
            # if iorder[0] not in self.node_mapping.keys() and iorder[1] not in self.node_mapping.keys(): # 如果订单的起始节点和结束节点不是有效的，那么跳过该订单
            #     continue
            start_node = int(iorder[0])   # 开始节点编号（从0开始的）
            end_node = int(iorder[1])    # 结束节点编号
            if start_node not in self.target_grids or end_node not in self.target_grids: # 如果订单的起始节点和结束节点不是有效的，那么跳过该订单
                continue
            duration = int(iorder[3])                               # 持续时间
            price = iorder[4]                                       # 订单价格
            e = np.argwhere(self.target_grids == iorder[0])[0][0]
            s = np.argwhere(self.target_grids == iorder[1])[0][0]
            node_feature[start_time][e][0] += 1
            node_feature[start_time][s][1] += price
            node_feature[start_time][s][2] += duration
            if start_node == -100:
                column_index = self.target_grids.index(end_node)
                self.out_grid_in_orders[(start_time + duration) % self.n_intervals, column_index] += 1
                continue

            day_orders[start_time].append([start_node, end_node, start_time, duration, price])
        self.day_orders = day_orders    # 一天中每个个时段内的所有订单
        return node_feature

    def reset_clean_new_c(self, generate_order=1, ratio=1, city_time=""):
            """ 1. bootstrap oneday's order data.  引导一天的订单数据。
                2. clean current drivers and orders, regenerate new orders and drivers.
                can reset anytime  清理当前驱动程序和订单，重新生成新的订单和驱动程序。
    可以随时重置
            :return:
            """
            if city_time != "":
                self.city_time = city_time

            # clean orders and drivers
            self.drivers = {}  # driver[driver_id] = driver_instance  , driver_id start from 0
            self.n_drivers = 0  # total idle number of drivers. online and not on service.
            self.n_offline_drivers = 0  # total number of offline drivers.
            # 把所有节点的订单，车辆啥的全都清空
            for node in self.nodes:
                if node is not None:
                    node.clean_node()
            if city_time == 0:
                asd = 1
            # Generate one day's order.
            # 会根据真实订单有概率的选取出其中一部分来生成
            if generate_order == 1:
                node_feature, arrive_v = self.utility_bootstrap_oneday_order_new_c()  # 生成一天的订单，结果存储到env.day_orders,144个[]，每个[]有当天的所有订单

            # Init orders of current time step   当前时间步长的初始顺序
            moment = self.city_time % self.n_intervals
            self.step_bootstrap_order_real(self.day_orders[moment])  # 把0时刻的订单分配给所有节点

            # Init current driver distribution  初始化空闲司机分配，该方法只在每天0时刻用到了
            if self.global_flag == "global":
                num_driver = self.utility_get_n_idle_drivers_real()
                num_driver = int(num_driver * ratio)
            else:
                num_driver = self.utility_get_n_idle_drivers_nodewise()
            num_idle_driver = int(num_driver * 0.15)
            num_off_line_driver = num_driver - num_idle_driver
            self.step_driver_online_offline_control_new(num_idle_driver, True)
            self.step_driver_online_offline_control_new(num_off_line_driver, False)
            return self.get_observation(), node_feature, arrive_v

    def utility_bootstrap_oneday_order_new_c(self):
        arrive_v = np.zeros((145, self.n_valid_grids))   # 该网格作为目的地的次数、该网格内订单价值总和、该网格内订单价值时长总和
        node_feature = np.zeros((145, self.n_valid_grids, 3))
        num_all_orders = len(self.real_orders)  # 订单总数
        # 从订单中根据概率为28分之1的概率选出该天出现的哪些订单
        # 即每个真实订单有28分支1的可能性出现，index_sampled_orders就是那
        index_sampled_orders = np.where(np.random.binomial(1, self.p, num_all_orders) == 1)
        one_day_orders = self.real_orders[index_sampled_orders]
        self.day_orders_num = len(one_day_orders)
        self.out_grid_in_orders = np.zeros((self.n_intervals, len(self.target_grids)))  # 144×255

        day_orders = [[] for _ in np.arange(self.n_intervals)]  # 144
        for iorder in one_day_orders:
            #  iorder: [92, 300, 143, 2, 13.2]  这个是一个选中的订单
            start_time = int(iorder[2]) # 订单开始时间
            # node_mapping_keys = self.node_mapping.keys()    # 有效网格集合
            # if iorder[0] not in self.node_mapping.keys() and iorder[1] not in self.node_mapping.keys(): # 如果订单的起始节点和结束节点不是有效的，那么跳过该订单
            #     continue
            start_node = int(iorder[0])   # 开始节点编号（从0开始的）
            end_node = int(iorder[1])    # 结束节点编号
            if start_node not in self.target_grids or end_node not in self.target_grids: # 如果订单的起始节点和结束节点不是有效的，那么跳过该订单
                continue
            duration = int(iorder[3])                               # 持续时间
            price = iorder[4]                                       # 订单价格
            e = np.argwhere(self.target_grids == iorder[0])[0][0]
            s = np.argwhere(self.target_grids == iorder[1])[0][0]
            arrive_time = start_time + int(iorder[3])
            if arrive_time<145:
                arrive_v[arrive_time][e] += 1
            node_feature[start_time][e][0] += 1
            node_feature[start_time][s][1] += price
            node_feature[start_time][s][2] += duration
            if start_node == -100:
                column_index = self.target_grids.index(end_node)
                self.out_grid_in_orders[(start_time + duration) % self.n_intervals, column_index] += 1
                continue

            day_orders[start_time].append([start_node, end_node, start_time, duration, price])
        self.day_orders = day_orders    # 一天中每个个时段内的所有订单
        return node_feature, arrive_v
