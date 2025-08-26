import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def SSD(y, y_pred):
    return np.sum(np.square(y - y_pred), axis=1)  # axis 1 is the signal dimension


def MAD(y, y_pred):
    return np.max(np.abs(y - y_pred), axis=1)  # axis 1 is the signal dimension


def PRD(y, y_pred):
    N = np.sum(np.square(y_pred - y), axis=1)
    D = np.sum(np.square(y_pred - np.mean(y)), axis=1)

    PRD = np.sqrt(N / D) * 100

    return PRD


def COS_SIM(y, y_pred):
    # cos_sim = []
    # # 仅在最后一个轴的大小为1时进行压缩
    # if y.shape[-1] == 1:
    #     y = np.squeeze(y, axis=-1)
    # if y_pred.shape[-1] == 1:
    #     y_pred = np.squeeze(y_pred, axis=-1)
    #
    # for idx in range(len(y)):
    #     kl_temp = cosine_similarity(y[idx].reshape(1, -1), y_pred[idx].reshape(1, -1))
    #     cos_sim.append(kl_temp)
    #
    # cos_sim = np.array(cos_sim)

    cos_sim = []
    # 遍历每个样本
    for idx in range(len(y)):
        channel_sim = []  # 用于存储每个通道的余弦相似度

        # 遍历每个通道
        for channel in range(y.shape[2]):
            # 分别计算每个通道的余弦相似度
            sim = cosine_similarity(y[idx, :, channel].reshape(1, -1),
                                    y_pred[idx, :, channel].reshape(1, -1))
            channel_sim.append(sim)

        # 将各通道的相似度进行平均或其他处理（这里我们取平均值）
        # channel_sim_mean = np.mean(channel_sim)
        cos_sim.append(channel_sim)
    cos_sim = np.array(cos_sim)
    return cos_sim



def SNR(y1, y2, epsilon=1e-10):
    N = np.sum(np.square(y1), axis=1)
    D = np.sum(np.square(y2 - y1), axis=1)

    # 如果D中有小于等于0的值，将其设置为epsilon
    N = np.where(N <= epsilon, epsilon, N)
    D = np.where(D <= epsilon, epsilon, D)

    SNR = 10 * np.log10(N / D)

    return SNR



def SNR_improvement(y_in, y_out, y_clean):
    return SNR(y_clean, y_out) - SNR(y_clean, y_in)






