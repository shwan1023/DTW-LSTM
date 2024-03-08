import scipy.io as sio
import math
import matplotlib.pyplot as plt
import numpy as np


# Machine Learning part
def objective_function_MSE(Seq, target):
    return np.mean((Seq - target) ** 2)

def gradient_of_objective_function_MSE(Seq, target):
    # 计算目标函数相对于参数a的梯度
    gradient = 2 * (Seq - target)
    return gradient

def restore_model(Seq_window, target_window, Fourier_window):
    a = np.ones_like(Seq_window)
    max_iterations = 10
    learning_rate = 0.001
    for i in range(max_iterations):
        for j in range(len(Seq_window)):
            gradient = gradient_of_objective_function_MSE(Seq_window[j], target_window[j])
            max_gradient_norm = 0.1
            gradient = np.clip(gradient, -max_gradient_norm, max_gradient_norm)
            a[j] = a[j] - learning_rate * gradient
            delta = abs(Fourier_window[j] - Seq_window[j])
            gamma = np.minimum(a[j] * delta, delta)
            Seq_window[j] = np.where(Seq_window[j] > Fourier_window[j], Seq_window[j] - gamma, Seq_window[j] + gamma)
    return Seq_window

def restore_model_mixed(Seq_window, originTemp_window, Fourier_window):
    Seq_window_origin = Seq_window.copy()
    Seq_window_combined = Seq_window.copy()
    Seq_window_Fourier = Seq_window.copy()
    Seq_window_origin = restore_model(Seq_window_origin, originTemp_window, originTemp_window,)
    Seq_window_combined = restore_model(Seq_window_combined, originTemp_window, Fourier_window)
    Seq_window_Fourier = restore_model(Seq_window_Fourier, Fourier_window, Fourier_window)
    Seq_window_mixed = 0 * Seq_window_origin + 0.225 * Seq_window_combined + 0.775 * Seq_window_Fourier
    return Seq_window_mixed



def dataStack(Seq,originTemp,Fourier,index,Length):
    # # Original
    # Seq[index - Length:index] = restore_model(Seq[index - Length:index],originTemp[index - Length:index],Fourier[index - Length:index],deltaY)
    # return Seq[index - Length:index]

    # 除去修复模型
    # return (Fourier[index - Length:index] + originTemp[index - Length:index]) / 2

    # Origin-Fourier 1.5578
    # Seq[index - Length:index] = restore_model(Seq[index - Length:index],
    #                                           originTemp[index - Length:index],
    #                                           restore_model_mixed(Seq[index - Length:index],
    #                                                               restore_model_mixed(Seq[index - Length:index],
    #                                                                                   originTemp[index - Length:index],
    #                                                                                   Fourier[index - Length:index],
    #                                                                                   deltaY),
    #                                                               Fourier[index - Length:index],
    #                                                               deltaY),
    #                                           deltaY)
    # return Seq[index - Length:index]

    # Origin-Origin 2.2241
    # Seq[index - Length:index] = restore_model(Seq[index - Length:index],
    #                                           originTemp[index - Length:index],
    #                                           restore_model_mixed(Seq[index - Length:index],
    #                                                               originTemp[index - Length:index],
    #                                                               restore_model_mixed(Seq[index - Length:index],
    #                                                                                   originTemp[index - Length:index],
    #                                                                                   Fourier[index - Length:index],
    #                                                                                   ),
    #                                                               ),
    #                                           )
    # return Seq[index - Length:index]

    # Fourier-Origin 1.5749
    # Seq[index - Length:index] = restore_model(Seq[index - Length:index],
    #                                           restore_model_mixed(Seq[index - Length:index],
    #                                                               originTemp[index - Length:index],
    #                                                               restore_model_mixed(Seq[index - Length:index],
    #                                                                                   originTemp[index - Length:index],
    #                                                                                   Fourier[index - Length:index],
    #                                                                                   deltaY),
    #                                                               deltaY),
    #                                           Fourier[index - Length:index],
    #                                           deltaY)
    # return Seq[index - Length:index]

    # 除去修复模型
    # return originTemp[index - Length:index] / 2 + Fourier[index - Length:index] / 2

    # Fourier-Fourier 1.2653
    Seq[index - Length:index] = restore_model(Seq[index - Length:index],
                                              restore_model_mixed(Seq[index - Length:index],
                                                                  restore_model_mixed(Seq[index - Length:index],
                                                                                      originTemp[index - Length:index],
                                                                                      Fourier[index - Length:index],
                                                                                      ),
                                                                  Fourier[index - Length:index],
                                                                  ),
                                              Fourier[index - Length:index],
                                           )
    return Seq[index - Length:index]

    # Fourier-Fourier-Fourier
    # Seq[index - Length:index] = restore_model(Seq[index - Length:index],
    #                                           Fourier[index - Length:index],
    #                                           restore_model_mixed(Seq[index - Length:index],
    #                                               Fourier[index - Length:index],
    #                                               restore_model_mixed(Seq[index - Length:index],
    #                                                   Fourier[index - Length:index],
    #                                                   originTemp[index - Length:index],
    #                                                   deltaY),
    #                                               deltaY),
    #                                           deltaY)
    # return Seq[index - Length:index]


    # p1
    # Seq[index - Length:index] = restore_model(Seq[index - Length:index],
    #                                           restore_model_mixed(Seq[index - Length:index],
    #                                                               restore_model_mixed(Seq[index - Length:index],
    #                                                                                   originTemp[index - Length:index],
    #                                                                                   Fourier[index - Length:index],
    #                                                                                   deltaY),
    #                                                               Fourier[index - Length:index],
    #                                                               deltaY),
    #                                           Fourier[index - Length: index], deltaY)


    # Seq[index - Length:index] = restore_model(Seq[index - Length:index],
    #                                           restore_model_mixed(Seq[index - Length:index],
    #                                                               originTemp[index - Length:index],
    #                                                               restore_model_mixed(Seq[index - Length:index],
    #                                                                                   originTemp[index - Length:index],
    #                                                                                   Fourier[index - Length:index],
    #                                                                                   deltaY),
    #                                                               deltaY),
    #                                           Fourier[index - Length: index], deltaY)

    # Seq[index - Length:index] = restore_model(Seq[index - Length:index],
    #                                           restore_model_mixed(Seq[index - Length:index],
    #                                                               originTemp[index - Length:index],
    #                                                               restore_model_mixed(Seq[index - Length:index],
    #                                                                                   originTemp[index - Length:index],
    #                                                                                   Fourier[index - Length:index],
    #                                                                                   deltaY), deltaY),
    #                                           Fourier[index - Length: index], deltaY)

    # Seq[index - Length:index] = restore_model(Seq[index - Length:index],
    #                                           originTemp[index - Length:index],
    #                                           restore_model_mixed(Seq[index - Length:index],
    #                                                               originTemp[index - Length:index],
    #                                                               restore_model_mixed(Seq[index - Length:index],
    #                                                                                   originTemp[index - Length:index],
    #                                                                                   Fourier[index - Length:index],
    #                                                                                   deltaY), deltaY),
    #                                           Fourier[index - Length: index], deltaY)


    # 3-layers:
    # Seq[index - Length:index] = restore_model(Seq[index - Length:index],
    #                                           restore_model_mixed(Seq[index - Length:index],
    #                                                               restore_model_mixed(Seq[index - Length:index],
    #                                                                                   restore_model_mixed(
    #                                                                                       Seq[index - Length:index],
    #                                                                                       originTemp[
    #                                                                                       index - Length:index],
    #                                                                                       Fourier[index - Length:index],
    #                                                                                       ),
    #                                                                                   Fourier[index - Length:index],
    #                                                                                   ),
    #                                                               Fourier[index - Length:index],
    #                                                               ),
    #                                           Fourier[index - Length:index],
    #                                           )
    # return Seq[index - Length:index]

    # 4
    # Seq[index - Length:index] = restore_model(Seq[index - Length:index],
    #                                           restore_model_mixed(Seq[index - Length:index],
    #                                                               restore_model_mixed(Seq[index - Length:index],
    #                                                                                   restore_model_mixed(
    #                                                                                       Seq[index - Length:index],
    #                                                                                       restore_model_mixed(
    #                                                                                           Seq[index - Length:index],
    #                                                                                           originTemp[
    #                                                                                           index - Length:index],
    #                                                                                           Fourier[
    #                                                                                           index - Length:index],
    #                                                                                           ),
    #                                                                                       Fourier[index - Length:index],
    #                                                                                       ),
    #                                                                                   Fourier[index - Length:index],
    #                                                                                   ),
    #                                                               Fourier[index - Length:index],
    #                                                               ),
    #                                           Fourier[index - Length:index],
    #                                           )
    # return Seq[index - Length:index]

    # 5
    # Seq[index - Length:index] = restore_model(Seq[index - Length:index],
    #                                           restore_model_mixed(Seq[index - Length:index],
    #                                                               restore_model_mixed(Seq[index - Length:index],
    #                                                                                   restore_model_mixed(
    #                                                                                       Seq[index - Length:index],
    #                                                                                       restore_model_mixed(
    #                                                                                           Seq[index - Length:index],
    #                                                                                           restore_model_mixed(Seq[
    #                                                                                                               index - Length:index],
    #                                                                                                               originTemp[
    #                                                                                                               index - Length:index],
    #                                                                                                               Fourier[
    #                                                                                                               index - Length:index],
    #                                                                                                               ),
    #                                                                                           Fourier[
    #                                                                                           index - Length:index],
    #                                                                                           ),
    #                                                                                       Fourier[index - Length:index],
    #                                                                                       ),
    #                                                                                   Fourier[index - Length:index],
    #                                                                                   ),
    #                                                               Fourier[index - Length:index],
    #                                                               ),
    #                                           Fourier[index - Length:index],
    #                                           )
    # return Seq[index - Length:index]

    # 6
    # Seq[index - Length:index] = restore_model(Seq[index - Length:index],
    #                                           restore_model_mixed(Seq[index - Length:index],
    #                                                               restore_model_mixed(Seq[index - Length:index],
    #                                                                                   restore_model_mixed(
    #                                                                                       Seq[index - Length:index],
    #                                                                                       restore_model_mixed(
    #                                                                                           Seq[index - Length:index],
    #                                                                                           restore_model_mixed(Seq[
    #                                                                                                               index - Length:index],
    #                                                                                                               restore_model_mixed(
    #                                                                                                                   Seq[
    #                                                                                                                   index - Length:index],
    #                                                                                                                   originTemp[
    #                                                                                                                   index - Length:index],
    #                                                                                                                   Fourier[
    #                                                                                                                   index - Length:index],
    #                                                                                                                   ),
    #                                                                                                               Fourier[
    #                                                                                                               index - Length:index],
    #                                                                                                               ),
    #                                                                                           Fourier[
    #                                                                                           index - Length:index],
    #                                                                                           ),
    #                                                                                       Fourier[index - Length:index],
    #                                                                                       ),
    #                                                                                   Fourier[index - Length:index],
    #                                                                                   ),
    #                                                               Fourier[index - Length:index],
    #                                                               )
    # return Seq[index - Length:index]

    # 7
    # Seq[index - Length:index] = restore_model(Seq[index - Length:index],
    #                                           restore_model_mixed(Seq[index - Length:index],
    #                                                               restore_model_mixed(Seq[index - Length:index],
    #                                                                                   restore_model_mixed(
    #                                                                                       Seq[index - Length:index],
    #                                                                                       restore_model_mixed(
    #                                                                                           Seq[index - Length:index],
    #                                                                                           restore_model_mixed(Seq[
    #                                                                                                               index - Length:index],
    #                                                                                                               restore_model_mixed(
    #                                                                                                                   Seq[
    #                                                                                                                   index - Length:index],
    #                                                                                                                   restore_model_mixed(
    #                                                                                                                       Seq[
    #                                                                                                                       index - Length:index],
    #                                                                                                                       originTemp[
    #                                                                                                                       index - Length:index],
    #                                                                                                                       Fourier[
    #                                                                                                                       index - Length:index],
    #                                                                                                                       ),
    #                                                                                                                   Fourier[
    #                                                                                                                   index - Length:index],
    #                                                                                                                   ),
    #                                                                                                               Fourier[
    #                                                                                                               index - Length:index],
    #                                                                                                               ),
    #                                                                                           Fourier[
    #                                                                                           index - Length:index],
    #                                                                                           ),
    #                                                                                       Fourier[index - Length:index],
    #                                                                                       ),
    #                                                                                   Fourier[index - Length:index],
    #                                                                                   )
    # return Seq[index - Length:index]
