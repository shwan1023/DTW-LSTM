from sklearn.decomposition import PCA
import scipy.io as sio
import math
import matplotlib.pyplot as plt
import numpy as np
import dataStack
import startup


class SingleClusterAlgorithm:
    def __init__(self, threshold=20.0, max_iterations=100):
        self.center = None
        self.threshold = threshold
        self.max_iterations = max_iterations
        self.data_points = []  # 存储所有加入的数据点
        self.distance = []
        self.addDistance = 0
        self.times = 1

    def fit(self, data):
        self.times += 1
        if len(data) == 0:
            self.center = data
        else:
            self.center = data[np.random.randint(0, len(data))]
            for _ in range(self.max_iterations):
                cluster_points = [point for point in data]
                new_center = np.mean(cluster_points, axis=0)
                if np.linalg.norm(new_center - self.center) < self.threshold:
                    break
                self.center = new_center

    def is_inside_range(self, point):
        self.distance.append(np.linalg.norm(np.array(point) - np.array(self.center)))
        self.addDistance += np.linalg.norm(np.array(point) - np.array(self.center))
        return np.linalg.norm(np.array(point) - np.array(self.center)) <= self.threshold

    def add_data_point(self, point):
        self.data_points.append(point)

    def get_all_data_points(self):
        return self.data_points

    def getDistance(self):
        return self.distance

    def getCenter(self):
        return self.center

    def setThreshold(self,threshold):
        self.threshold = threshold

    def aveOfDistance(self):
        return self.addDistance / self.times

def plot_2d_projection(nums_Cluster, center, threshold, single_cluster):
    pca = PCA(n_components=2)
    u_2d = pca.fit_transform(nums_Cluster)
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.scatter(u_2d[:, 0], u_2d[:, 1], c='b', label='Data Points')
    center_2d = pca.transform([center])[0]
    plt.scatter(center_2d[0], center_2d[1], c='r', marker='x', label='Center')
    circle = plt.Circle((center_2d[0], center_2d[1]), threshold, color='g', fill=False, label='Cluster Range')
    plt.gca().add_artist(circle)
    plt.legend()
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title("Area and distance (Threshold = {})".format(threshold))

    plt.subplot(1, 2, 2)
    distanceOfSC = SingleClusterAlgorithm.getDistance(single_cluster)
    plt.plot(distanceOfSC, label='Distance')
    plt.axhline(y=threshold, color='red', linestyle='--')
    plt.legend()
    plt.xlabel('Time Window Index')
    plt.ylabel('Distance')
    plt.title("Distance over Time")
    plt.tight_layout()

    plt.savefig("figs\\area_and_distance_threshold_{}.png".format(threshold))


def kurtosis(data):
    mean = np.mean(data)
    n = len(data)
    fourth_moment = np.sum((data - mean) ** 4) / n
    variance = np.var(data)
    kurt = fourth_moment / (variance ** 2)
    return kurt


def plot_cluster_and_number_lines(single_cluster):
    data_points = np.array(single_cluster.get_all_data_points())
    center = single_cluster.center
    n_points, n_dimensions = data_points.shape
    fig, axs = plt.subplots(n_dimensions, 1, figsize=(10, 11))

    for i in range(n_dimensions):
        axs[i].hlines(1, min(data_points[:, i]) - 0.1, max(data_points[:, i]) + 0.1)
        axs[i].set_xlim(min(data_points[:, i]) - 0.1, max(data_points[:, i]) + 0.1)
        axs[i].set_ylim(0.9, 1.1)
        axs[i].plot(data_points[:, i], np.ones(n_points), '|', ms=15, color='blue')
        center_color = 'red'
        axs[i].plot(center[i], 1, 'x', ms=15, color=center_color, mew=3, mec=center_color)
        axs[i].set_yticks([])
        axs[i].spines['left'].set_color('none')
        axs[i].spines['right'].set_color('none')
        axs[i].spines['top'].set_color('none')
        axs[i].set_ylabel(f' Dimension{i + 1}')

    plt.xlabel('Distance')
    plt.suptitle("Time Window Positions and Relationship to Center")
    # plt.subplots_adjust(top=2)  # Adjust the top margin for the suptitle
    plt.tight_layout()
    plt.savefig("figs\\axes.png")
    plt.show()


# Use the function
# plot_cluster_and_number_lines(single_cluster)


def statusSpaceCreate(Fourier, originTemp, index, Length, deltaY):
    fourierAdd, originAdd = 0, 0
    # for i in range(index - Length, index):
    #     fourierAdd += Fourier[i]
    #     originAdd += originTemp[i]
    # fourierAdd / len(Fourier[index - Length: index])
    # originAdd / len(originTemp[index - Length: index])

    if len(originTemp[index - Length: index]) > 0 and len(originTemp[index - Length - 1: index - 1]) > 0:
        autoCorrelation = np.corrcoef(originTemp[index - Length: index], originTemp[index - Length - 1: index - 1])[
            0, 1]
    else:
        autoCorrelation = 0
    # movingAverage = fourierAdd
    originStd = np.std(originTemp[index - Length: index])
    fourierStd = np.std(Fourier[index - Length: index])
    origin_kurtosis = kurtosis(originTemp[index - Length: index])
    fourier_kurtosis = kurtosis(Fourier[index - Length: index])
    cov_matrix = np.cov(originTemp[index - Length: index], Fourier[index - Length: index])
    cov_origin_fourier = cov_matrix[0, 1]
    nums_Cluster.append([
        autoCorrelation,
        # movingAverage,
        originStd,
        fourierStd,
        origin_kurtosis,
        fourier_kurtosis,
        cov_origin_fourier,
        math.sqrt(np.sum(deltaY ** 2)) / len(deltaY)
    ])
    return nums_Cluster


def windowTest(Fourier, PekingTemp, Length, TrainLimit,threshold_pa):
    # Initialization
    originTemp = PekingTemp
    global nums_Cluster, nums_LegalTimes

    single_cluster = SingleClusterAlgorithm(threshold=threshold_pa, max_iterations=100)
    # Preallocation

    nums_Cluster = []
    nums_LegalTimes = []
    Seq = np.zeros(TrainLimit)

    # Main computation loop
    index = 1
    iterOfWindow = -1

    while index < TrainLimit:
        # print(len(originTemp))
        '''
        进入时间窗，首先先判断当前的时间窗特征空间
        '''
        if index - Length > 0 and index < len(originTemp):

            iterOfWindow += 1
            deltaY = np.abs(originTemp[index - Length: index] - Fourier[index - Length: index])

            nums_Cluster = statusSpaceCreate(Fourier, originTemp, index, Length, deltaY)
            # threshold = single_cluster.aveOfDistance()
            # single_cluster.setThreshold(num * threshold)
            single_cluster.fit(nums_Cluster)
            is_inside_cluster = single_cluster.is_inside_range(nums_Cluster[iterOfWindow])
        else:
            # 如果切片范围为空或超出了 originTemp 的范围，做适当处理
            # 这里可以选择直接跳过该次循环或进行其他处理
            index += 1
            continue

        # 特判
        # 除去聚类模型(消融实验)
        # if iterOfWindow == 0 or iterOfWindow == 1:

        if iterOfWindow == 0 or iterOfWindow == 1 or is_inside_cluster:
            '''
            如果时间窗序号 == 0 or 1，或者在类内，那么：
                seq直接取temp（不改变seq）
                第i个时间窗的res直接取第i-1个特征u
                时间窗仅向前移动一个时间步，这是因为没有进入修复模型
            '''
            Seq[index - Length:index] = (originTemp[index - Length:index])
            nums_LegalTimes.append(nums_Cluster[-1])
            index = index + 1
            if iterOfWindow != 0:
                single_cluster.add_data_point(nums_Cluster[iterOfWindow])
            continue
        else:
            '''

            进入修复模型：
                1、求出误差，并判定误差与界的关系
                2、存在界内则不变，否则调整参数a并进行修复，修复过程尽可能令seq靠近temp
                3、时间窗向前移动一个时间窗长度，这是因为已经进入修复模型，时间窗内数据无需再修复

            '''

            # 除去修复模型
            Seq[index - Length:index] = dataStack.dataStack(Seq, originTemp, Fourier, index, Length)

            #  除去堆栈模型
            # Seq[index - Length : index] = restore_model(Seq[index - Length : index],originTemp[index - Length : index],Fourier[index - Length : index])

            index += Length
            single_cluster.add_data_point(nums_Cluster[iterOfWindow])

    if len(Seq) != TrainLimit:
        Seq[index - Length:index] = (originTemp[index - Length:index])

    print(f"Lengths of Seq,PekingTemp,nums_Cluster,nums_LegalTimes equal to {len(Seq)},{len(PekingTemp)},{len(nums_Cluster)},{len(nums_LegalTimes)}")
    plot_2d_projection(nums_Cluster, single_cluster.center, single_cluster.threshold, single_cluster)
    plot_cluster_and_number_lines(single_cluster)

    return Seq

"""
main part
"""

[Fourier,PekingTemp,Length,TrainLimit,threshold_pa] = startup.initialize()
bestSeq = windowTest(Fourier,PekingTemp,Length,TrainLimit,threshold_pa)

sio.savemat("result\\processedTrainDataset.mat", {'Processed':bestSeq})