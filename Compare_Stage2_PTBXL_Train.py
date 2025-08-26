import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
import pickle
import Utils_Metrics_01 as metrics
import matplotlib.pyplot as plt
import time

def train(model, config, train_loader, device, valid_loader=None, valid_epoch_interval=5, foldername=""):
    optimizer = Adam(model.parameters(), lr=config["lr"])
    # ema = EMA(0.9)
    # ema.register(model)

    if foldername != "":
        output_path = foldername + "/model_"
        final_path = foldername + "/final.pth"

    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=50, gamma=.5
    )

    best_valid_loss = 1e10
    last_saved_epoch = 0  # 跟踪最近保存模型的 epoch 号

    for epoch_no in range(config["epochs"]):
        avg_loss = 0
        model.train()

        with tqdm(train_loader) as it:
            for batch_no, (clean_batch, noisy_batch, desc_batch) in enumerate(it, start=1):
                clean_batch, noisy_batch, desc_batch = clean_batch.to(device), noisy_batch.to(device), desc_batch.to(device)
                optimizer.zero_grad()

                start = time.time()

                loss = model(clean_batch, noisy_batch, desc_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.model.parameters(), 1.0)
                optimizer.step()

                end = time.time()
                print(end - start)

                avg_loss += loss.item()

                # ema.update(model)

                it.set_postfix(
                    ordered_dict={
                        "avg_epoch_loss": avg_loss / batch_no,
                        "epoch": epoch_no,
                    },
                    refresh=True,
                )

            lr_scheduler.step()
            current_lr = lr_scheduler.get_last_lr()[0]
            print(f"Epoch {epoch_no + 1}, Current learning rate: {current_lr}")

        if valid_loader is not None and (epoch_no + 1) % valid_epoch_interval == 0:
            model.eval()
            avg_loss_valid = 0
            with torch.no_grad():
                with tqdm(valid_loader) as it:
                    for batch_no, (clean_batch, noisy_batch, desc_batch) in enumerate(it, start=1):
                        clean_batch, noisy_batch, desc_batch = clean_batch.to(device), noisy_batch.to(device), desc_batch.to(device)
                        loss = model(clean_batch, noisy_batch, desc_batch)
                        avg_loss_valid += loss.item()

                        # # 画一下效果
                        # if batch_no == 1:
                        #     output = model.denoising(noisy_batch, desc_batch)
                        #     output = output.permute(0, 2, 1)  # B,L,1
                        #     out_numpy = output.cpu().detach().numpy()
                        #     clean_batch = clean_batch.permute(0, 2, 1)
                        #     clean_numpy = clean_batch.cpu().detach().numpy()
                        #     # print("在这停顿！")
                        #     List = 10
                        #     lead = 6
                        #     Ori_Data = clean_numpy[List, :, lead]
                        #     Re_Data = out_numpy[List, :, lead]
                        #     No_Data = clean_numpy[List, :, 1]
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

                        it.set_postfix(
                            ordered_dict={
                                "valid_avg_epoch_loss": avg_loss_valid / batch_no,
                                "epoch": epoch_no,
                            },
                            refresh=True,
                        )

            if best_valid_loss > avg_loss_valid / batch_no:
                best_valid_loss = avg_loss_valid / batch_no
                last_saved_epoch = epoch_no  # 更新最近保存模型的 epoch 号
                print("\n best loss is updated to ", avg_loss_valid / batch_no, "at", epoch_no, )

                if foldername != "":
                    torch.save(model.state_dict(), output_path + str(epoch_no) + '.pth')

        # 如果已经10个epoch没有保存新的模型权重，则存一个
        if (epoch_no - last_saved_epoch) >= 10:
            last_saved_epoch = epoch_no
            print("\n 10 epochs passed without saving, saving model at epoch", epoch_no)
            if foldername != "":
                torch.save(model.state_dict(), output_path + str(epoch_no) + '_backup.pth')

    torch.save(model.state_dict(), final_path)