import numpy as np
import scipy.io as sio

def initialize():
    """
    初始化全局变量
    """
    # vars of Basis
    global Fourier,PekingTemp
    Fourier = load_txt_file('data/data1.txt')
    PekingTemp = load_txt_file('data/data2.txt')
    # dataAll = sio.loadmat("data\\Walmalt.mat")

    # Fourier = sio.loadmat("data\\data1.mat")
    # Fourier = Fourier['y_fit']
    # PekingTemp = sio.loadmat("data\\data2.mat")
    # PekingTemp = PekingTemp['volume']

    # dataAll = sio.loadmat("data\\ETTh1.mat")
    # Fourier = dataAll['y_fit']
    # PekingTemp = dataAll['y']


    # Parameters Initialization

    Length = 5
    TrainLimit = 4745
    threshold_pa = 1

    # data_array = load_txt_file("data\\data1.txt")

    return Fourier,PekingTemp,Length,TrainLimit,threshold_pa
    #vars of GM


def load_txt_file(file_path):
    """
    Load data from a text file and convert it into a NumPy array.
    Args:
        file_path (str): Path to the text file.
    Returns:
        numpy.ndarray: A NumPy array containing the data from the text file.
    """
    # 从文本文件中读取数据
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.read()

    # 将数据转换为 NumPy 数组
    data_array = np.fromstring(data, sep='\t')

    return data_array