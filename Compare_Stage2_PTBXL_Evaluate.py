import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
import pickle
import Utils_Metrics_01 as metrics
import matplotlib.pyplot as plt
import time


def evaluate(model, test_loader, shots, device, foldername=""):
    ssd_total = 0
    mad_total = 0
    prd_total = 0
    cos_sim_total = 0
    snr_noise = 0
    snr_recon = 0
    snr_improvement = 0
    eval_points = 0

    restored_sig = []
    with tqdm(test_loader) as it:
        for batch_no, (clean_batch, noisy_batch, desc_batch) in enumerate(it, start=1):
            clean_batch, noisy_batch, desc_batch = clean_batch.to(device), noisy_batch.to(device), desc_batch.to(device)

            # # 画一下效果
            # if batch_no == 1:
            #     output1 = model.denoising(noisy_batch, desc_batch)
            #     output1 = output1.permute(0, 2, 1)  # B,L,1
            #     out_numpy1 = output1.cpu().detach().numpy()
            #     clean_batch1 = clean_batch.permute(0, 2, 1)
            #     clean_numpy1 = clean_batch1.cpu().detach().numpy()
            #     # print("在这停顿！")
            #     List = 10
            #     lead = 6
            #     Ori_Data = clean_numpy1[List, :, lead]
            #     Re_Data = out_numpy1[List, :, lead]
            #     No_Data = clean_numpy1[List, :, 1]
            #     # 定义采样频率和时间范围
            #     sampling_rate = 100
            #     start_time = 2  # 2秒
            #     end_time = 7  # 7秒
            #     # 计算索引范围
            #     start_index = start_time * sampling_rate
            #     end_index = end_time * sampling_rate
            #     # 提取数据
            #     extracted_ori_data = Ori_Data[start_index:end_index]
            #     extracted_re_data = Re_Data[start_index:end_index]
            #     extracted_no_data = No_Data[start_index:end_index]
            #     # 创建时间轴（时间单位为秒）
            #     time_axis = np.linspace(start_time, end_time, end_index - start_index)
            #     # 创建图形和子图
            #     plt.figure(figsize=(10, 8))
            #     # 绘制原始数据
            #     plt.subplot(3, 1, 1)  # 三行一列，这是第一个图
            #     plt.plot(time_axis, extracted_ori_data, label='Original Lead')
            #     plt.title('Original Data (2-7 seconds) Lead' + str(lead + 1))
            #     plt.xlabel('Time (seconds)')
            #     plt.ylabel('Amplitude')
            #     plt.grid(True)
            #     # 绘制重建数据
            #     plt.subplot(3, 1, 2)  # 三行一列，这是第二个图
            #     plt.plot(time_axis, extracted_re_data, label='Reconstructed Data')
            #     plt.title('Reconstructed Data (2-7 seconds) Lead' + str(lead + 1))
            #     plt.xlabel('Time (seconds)')
            #     plt.ylabel('Amplitude')
            #     plt.grid(True)
            #     # 绘制对比数据
            #     plt.subplot(3, 1, 3)  # 三行一列，这是第三个图
            #     plt.plot(time_axis, extracted_no_data, label='No Data')
            #     plt.title('No Data (2-7 seconds)')
            #     plt.xlabel('Time (seconds)')
            #     plt.ylabel('Amplitude')
            #     plt.grid(True)
            #     # 调整布局并显示图形
            #     plt.tight_layout()
            #     plt.show()
            #     # 画完了

            if shots > 1:
                output = 0
                for i in range(shots):
                    output += model.denoising(noisy_batch, desc_batch)
                output /= shots
            else:
                start = time.time()
                output = model.denoising(noisy_batch, desc_batch)  # B,1,L
                end = time.time()
                print(end-start)

            clean_batch = clean_batch.permute(0, 2, 1)
            noisy_batch = noisy_batch.permute(0, 2, 1)
            output = output.permute(0, 2, 1)  # B,L,1
            out_numpy = output.cpu().detach().numpy()
            clean_numpy = clean_batch.cpu().detach().numpy()
            noisy_numpy = noisy_batch.cpu().detach().numpy()
            # route4 = 'F:/Paper03_Data/PTBXL_100Hz/' + 'Ori_Lead8_Cond.npy'
            # pickle.dump(clean_numpy, open(route4, 'wb'), protocol=4)
            # route5 = 'F:/Paper03_Data/PTBXL_100Hz/' + 'Re_Lead8_Cond.npy'
            # pickle.dump(out_numpy, open(route5, 'wb'), protocol=4)
            # route6 = 'F:/Paper03_Data/PTBXL_100Hz/' + 'Ori_Lead2_Cond.npy'
            # pickle.dump(noisy_numpy, open(route6, 'wb'), protocol=4)
            # route4 = 'F:/Paper03_Data/PTBXL_100Hz/' + 'Ori_Unet_01.npy'
            # pickle.dump(clean_numpy, open(route4, 'wb'), protocol=4)
            # route5 = 'F:/Paper03_Data/PTBXL_100Hz/' + 'Re_Unet_02.npy'
            # pickle.dump(out_numpy, open(route5, 'wb'), protocol=4)
            # route6 = 'F:/Paper03_Data/PTBXL_100Hz/' + 'Ori_Lead2_Unet.npy'
            # pickle.dump(noisy_numpy, open(route6, 'wb'), protocol=4)
            # route4 = 'F:/Paper03_Data/PTBXL_100Hz/' + 'Ori_DiT_01.npy'
            # pickle.dump(clean_numpy, open(route4, 'wb'), protocol=4)
            # route5 = 'F:/Paper03_Data/PTBXL_100Hz/' + 'Re_DiT_02.npy'
            # pickle.dump(out_numpy, open(route5, 'wb'), protocol=4)
            # route6 = 'F:/Paper03_Data/PTBXL_100Hz/' + 'Ori_Lead2_DiT.npy'
            # pickle.dump(noisy_numpy, open(route6, 'wb'), protocol=4)


            eval_points += len(output)
            ssd_total += np.sum(metrics.SSD(clean_numpy, out_numpy))
            mad_total += np.sum(metrics.MAD(clean_numpy, out_numpy))
            prd_total += np.sum(metrics.PRD(clean_numpy, out_numpy))
            cos_sim_total += np.sum(metrics.COS_SIM(clean_numpy, out_numpy))
            # snr_noise += np.sum(metrics.SNR(clean_numpy, noisy_numpy))
            snr_recon += np.sum(metrics.SNR(clean_numpy, out_numpy))
            # snr_improvement += np.sum(metrics.SNR_improvement(noisy_numpy, out_numpy, clean_numpy))
            restored_sig.append(out_numpy)

            it.set_postfix(
                ordered_dict={
                    "ssd_total": ssd_total / eval_points,
                    "mad_total": mad_total / eval_points,
                    "prd_total": prd_total / eval_points,
                    "cos_sim_total": cos_sim_total / eval_points,
                    # "snr_in": snr_noise / eval_points,
                    "snr_out": snr_recon / eval_points,
                    # "snr_improve": snr_improvement / eval_points,
                },
                refresh=True,
            )

    restored_sig = np.concatenate(restored_sig)
    # np.save(foldername + '/CFDDPM_val.npy', restored_sig)

    print("ssd_total: ", ssd_total / eval_points)
    print("mad_total: ", mad_total / eval_points, )
    print("prd_total: ", prd_total / eval_points, )
    print("cos_sim_total: ", cos_sim_total / eval_points, )
    # print("snr_in: ", snr_noise / eval_points, )
    print("snr_out: ", snr_recon / eval_points, )
    # print("snr_improve: ", snr_improvement / eval_points, )