success_confidences = []
fail_confidences = []

with open('logs/confidence.log', 'r') as f:
    lines = f.readlines()

current_confidences = []
for line in lines:
    line = line.strip()

    # 记录 confidence
    if line.startswith('confidence:'):
        conf_value = float(line.split(':')[1].strip())
        current_confidences.append(conf_value)

    # 每次遇到保存视频说明一个 episode 结束
    if 'Saved rollout MP4 at path' in line:
        episode_finished = True  # 标志位，可以省略

    # 检查 Success
    if 'Success:' in line:
        success = 'Success: True' in line
        if success:
            success_confidences.extend(current_confidences)
        else:
            fail_confidences.extend(current_confidences)
        current_confidences = []  # 清空缓存

# 计算平均值
import numpy as np

success_avg = np.mean(success_confidences) if success_confidences else 0
fail_avg = np.mean(fail_confidences) if fail_confidences else 0

print(f'Success average confidence: {success_avg}')
print(f'Failure average confidence: {fail_avg}')


# import matplotlib.pyplot as plt

# log_path = 'logs/difusion_15000.log'  # 改成你的 log 文件路径

# all_confidences = []
# current_confidences = []

# with open(log_path, 'r') as f:
#     for line in f:
#         line = line.strip()

#         # 解析 confidence
#         if line.startswith('confidence:'):
#             conf_value = float(line.split(':')[1].strip())
#             current_confidences.append(conf_value)

#         # 遇到 episode 结束
#         if 'Saved rollout MP4 at path' in line:
#             if current_confidences:
#                 all_confidences.append(current_confidences)
#                 current_confidences = []

# # 绘图
# plt.figure(figsize=(14, 8))
# for idx, confs in enumerate(all_confidences):
#     plt.plot(range(len(confs)), confs, alpha=0.5)

# plt.xlabel('Time Step')
# plt.ylabel('Confidence')
# plt.title('Confidence Curves per Episode')
# plt.grid(True)

# # 保存图片
# plt.savefig('confidence_curves.png', dpi=300)
# print('✅ 图片已保存为 confidence_curves.png')

# plt.show()

