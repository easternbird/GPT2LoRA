import matplotlib.pyplot as plt

# 假设的train_acc数据，每个rank有6个epoch的数据
train_acc_0 = [0.83628 ,0.91824 ,0.93672 ,0.95252 ,0.96684 ,0.97536]
train_acc_1 = [0.8482,0.9172,0.93496,0.9526,0.96268,0.97212]
train_acc_2 = [0.84136,0.91384,0.9368,0.95076,0.96356,0.97296]
train_acc_4 = [0.85144,0.91676,0.93636,0.95192,0.96556,0.97456]
train_acc_8 = [0.8174,0.90524,0.9306,0.94596,0.96144,0.97036]
train_acc_16 = [0.98624,0.98736,0.98908,0.99016,0.991,0.9926]



# epochs数据
epochs = list(range(1, 7))

# 创建图表和轴
plt.figure(figsize=(10, 6))

# 绘制每个rank对应的train_acc数据
plt.plot(epochs, train_acc_0, label='Without LoRA')
plt.plot(epochs, train_acc_1, label='LoRA Rank=1')
plt.plot(epochs, train_acc_2, label='LoRA Rank=2')
plt.plot(epochs, train_acc_4, label='LoRA Rank=4')
plt.plot(epochs, train_acc_8, label='LoRA Rank=8')
plt.plot(epochs, train_acc_16, label='LoRA Lock Rank=2')

# 自定义图形
plt.xlabel('Epoch')
plt.ylabel('Test Accuracy')
plt.title('Test Accuracy of GPT2 in IMDB fintuned with LoRA')
plt.legend()


plt.savefig('Test Accuracy.png', dpi=300)
# 显示图形
plt.show()

