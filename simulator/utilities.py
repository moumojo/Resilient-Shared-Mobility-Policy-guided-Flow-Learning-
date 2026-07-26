import numpy as np
import os
import errno



from datetime import datetime, timedelta

def datetime_range(start, end, delta):
    current = start
    while current < end:
        yield current
        current += delta


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

# 把二维数据转为一维数据
# 例如3×3的矩阵中，1行1列[0,0]—》[0],2行3列[1,2]->2×3-1=[5]
def ids_2dto1d(i, j, M, N):
    '''
    convert (i,j) in a M by N matrix to index in M*N list. (row wise)
    matrix: [[1,2,3], [4, 5, 6]]
    list: [0, 1, 2, 3, 4, 5, 6]
    index start from 0
    '''
    assert 0 <= i < M and 0 <= j < N
    index = i * N + j    #矩阵中的索引
    return index


def ids_1dto2d(ids, M, N):
    ''' inverse of ids_2dto1d(i, j, M, N)
        index start from 0
         逆矩阵
    '''
    i = ids // N
    j = ids - N * i
    return (i, j)


def get_neighbor_list(i, j, M, N, n, nodes):
    ''' n: n-sided polygon, construct for a 2d map  n边多边形，为二维地图构造
                 1
             6       2
               center
             5       3
                 4
    return index of neighbor 1, 2, 3, 4, 5,6 in the matrix
    '''

    neighbor_list = [None] * n
    if n == 6:
        # hexagonal
        # 如果是偶数列
        if j % 2 == 0:
            # 上边有节点，该节点作为0号邻居
            if i - 1 >= 0:
                neighbor_list[0] = nodes[ids_2dto1d(i-1, j,   M, N)]
            # 右边有节点，该节点作为1号邻居
            if j + 1 < N:
                neighbor_list[1] = nodes[ids_2dto1d(i,   j+1, M, N)]
            # 下边有节点且右边有节点（右下方有节点），该节点作为2号邻居
            if i + 1 < M and j + 1 < N:
                neighbor_list[2] = nodes[ids_2dto1d(i+1, j+1, M, N)]
            # 下边有节点，该节点作为3号邻居
            if i + 1 < M:
                neighbor_list[3] = nodes[ids_2dto1d(i+1, j,   M, N)]
            # 下边有节点且左边有节点（左下方有节点），该节点作为4号邻居
            if i + 1 < M and j - 1 >= 0:
                neighbor_list[4] = nodes[ids_2dto1d(i+1, j-1, M, N)]
            # 左边有节点，该节点作为5号邻居
            if j - 1 >= 0:
                neighbor_list[5] = nodes[ids_2dto1d(i,   j-1, M, N)]
        elif j % 2 == 1:
            # 上
            if i - 1 >= 0:
                neighbor_list[0] = nodes[ids_2dto1d(i-1, j,   M, N)]
            # 上右
            if i - 1 >= 0 and j + 1 < N:
                neighbor_list[1] = nodes[ids_2dto1d(i-1, j+1, M, N)]
            # 右
            if j + 1 < N:
                neighbor_list[2] = nodes[ids_2dto1d(i,   j+1, M, N)]
            # 下
            if i + 1 < M:
                neighbor_list[3] = nodes[ids_2dto1d(i+1, j,   M, N)]
            # 左
            if j - 1 >= 0:
                neighbor_list[4] = nodes[ids_2dto1d(i,   j-1, M, N)]
            # 上左
            if i - 1 >= 0 and j - 1 >= 0:
                neighbor_list[5] = nodes[ids_2dto1d(i-1, j-1, M, N)]
    # elif n == 4:
    #     # square
    #     if i - 1 >= 0:
    #         neighbor_list[0] = nodes[ids_2dto1d(i-1, j,   M, N)]
    #     if j + 1 < N:
    #         neighbor_list[1] = nodes[ids_2dto1d(i,   j+1, M, N)]
    #     if i + 1 < M:
    #         neighbor_list[2] = nodes[ids_2dto1d(i+1, j,   M, N)]
    #     if j - 1 >= 0:
    #         neighbor_list[3] = nodes[ids_2dto1d(i,   j-1, M, N)]

    return neighbor_list


def get_neighbor_index(i, j):
    """
                 1
             6       2
                center
             5       3
                 4
    return index of neighbor 1, 2, 3, 4, 5,6 in the matrix
    """
    neighbor_matrix_ids = []
    if j % 2 == 0:
        neighbor_matrix_ids = [[i - 1, j    ],
                               [i,     j + 1],
                               [i + 1, j + 1],
                               [i + 1, j    ],
                               [i + 1, j - 1],
                               [i    , j - 1]]
    elif j % 2 == 1:
        neighbor_matrix_ids = [[i - 1, j    ],
                               [i - 1, j + 1],
                               [i    , j + 1],
                               [i + 1, j    ],
                               [i    , j - 1],
                               [i - 1, j - 1]]

    return neighbor_matrix_ids


def get_layers_neighbors(i, j, l_max, M, N):
    """get neighbors of node layer by layer, todo BFS.
       i, j: center node location
       L_max: max number of layers
       layers_neighbors: layers_neighbors[0] first layer neighbor: 6 nodes: can arrived in 1 time step.
       layers_neighbors[1]: 2nd layer nodes id
       M, N: matrix rows and columns.
    """
    assert l_max >= 1
    layers_neighbors = []
    layer1_neighbor = get_neighbor_index(i, j)  #[[1,1], [0, 1], ...]
    temp = []
    for item in layer1_neighbor:
        x, y = item
        if 0 <= x <= M-1 and 0 <= y <= N-1:
            temp.append(item)
    layers_neighbors.append(temp)

    node_id_neighbors = []
    for item in layer1_neighbor:
        x, y = item
        if 0 <= x <= M-1 and 0 <= y <= N-1:
            node_id_neighbors.append(ids_2dto1d(x, y, M, N))

    layers_neighbors_set = set(node_id_neighbors)
    curr_ndoe_id = ids_2dto1d(i, j, M, N)
    layers_neighbors_set.add(curr_ndoe_id)

    t = 1
    while t < l_max:
        t += 1
        layer_neighbor_temp = []
        for item in layers_neighbors[-1]:
            x, y = item
            if 0 <= x <= M-1 and 0 <= y <= N-1:
                layer_neighbor_temp += get_neighbor_index(x, y)

        layer_neighbor = []  # remove previous layer neighbors
        for item in layer_neighbor_temp:
            x, y = item
            if 0 <= x <= M-1 and 0 <= y <= N-1:
                node_id = ids_2dto1d(x, y, M, N)
                if node_id not in layers_neighbors_set:
                    layer_neighbor.append(item)
                    layers_neighbors_set.add(node_id)
        layers_neighbors.append(layer_neighbor)

    return layers_neighbors


def get_driver_status(env):
    idle_driver_dist =list( np.zeros((env.M, env.N)))
    for driver_id, cur_drivers in env.drivers.iteritems():
        if cur_drivers.node is not None:
            node_id = cur_drivers.node.get_node_index()
            row, col = ids_1dto2d(node_id, env.M, env.N)
            if cur_drivers.onservice is False and cur_drivers.online is True:
                idle_driver_dist[row, col] += 1

    return idle_driver_dist

def debug_print_drivers(node):
    print("Status of all drivers in the node {}".format(node.get_node_index()))
    print("|{:12}|{:12}|{:12}|{:12}|".format("driver id", "driver location", "online", "onservice"))

    for driver_id, cur_drivers in node.drivers.iteritems():
        if cur_drivers.node is not None:
            node_id = cur_drivers.node.get_node_index()
        else:
            node_id = "none"
        print("|{:12}|{:12}|{:12}|{:12}|".format(driver_id, node_id, cur_drivers.online, cur_drivers.onservice))


### mine ###
# def ten_two(city_time):
#     a = binn(city_time)
#     a = list(a)
#     time = np.array(list(map(int, a)))
#     return time
# def binn(city_time):
#     n = city_time
#     # x = 2  # 转换为二进制，所以这里取x=2
#     b = []  # 存储余数
#     while True:  # 一直循环，商为0时利用break退出循环
#         s = n // 2  # 商
#         y = n % 2  # 余数
#         b = b + [y]  # 每一个余数存储到b中
#         if s == 0:
#             break  # 余数为0时结束循环
#         n = s
#     b.reverse()  # 使b中的元素反向排列
#     b = [str(i) for i in b]
#     while len(b) < 15:
#         b.append("0")
#     return b
# def get_dist(a,b):
#     x1, y1 = ids_1dto2d(a, 15, 17)
#     x2, y2 = ids_1dto2d(b, 15, 17)
#     # time = (abs((y1-y2))*3)//4 + (abs((x1-x2))*3//4)
#     time = int((abs((y1-y2))*3)/4 + (abs((x1-x2))*3/4))
#     return time

def get_dist(a, b):
    x1, y1 = ids_1dto2d(a, 15, 17)
    x2, y2 = ids_1dto2d(b, 15, 17)
    # time = (abs((y1-y2))*3)//4 + (abs((x1-x2))*3//4)
    if y1 == y2:
        dis = abs(x2-x1)
        return dis
    if y1 > y2:
        z1, z2 = x1, y1
        x1, y1 = x2, y2
        x2, y2 = z1, z2
    if y1%2==0:
        dis = abs(x2-x1)+abs(y2-y1)-min(abs(x2-x1), abs(y1-y1)/2 + 1)
        return dis
    if y1%2==1:
        dis = abs(x2-x1)+abs(y2-y1)-min(abs(x2-x1), abs(y1-y1)/2)
        return dis

