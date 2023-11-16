import matplotlib.pyplot as plt
import numpy as np

# 假设有两组参数组合，每组有不同的参数
param_sets = [
    {'param1': 0.1, 'param2': 0.5},
    {'param1': 0.2, 'param2': 0.3}
]

# 存储每组参数的演化结果
results = []

# 模拟运行演化算法多次，记录每次的最优结果
for params in param_sets:
    # 在这里运行演化算法，得到每次运行的最优结果，假设使用以下方式生成伪随机结果
    iterations = np.arange(1, 11)
    best_results = np.random.rand(10)

    # 存储每组参数的演化结果
    results.append({'params': params, 'iterations': iterations, 'best_results': best_results})

# 绘制曲线
for result in results:
    plt.plot(result['iterations'], result['best_results'], label=str(result['params']))

# 设置图例位置为右上角
plt.legend(title='Parameter Sets', loc='lower right')

plt.title('Evolution of Best Results Over Iterations')
plt.xlabel('Iterations')
plt.ylabel('Best Results')
plt.grid(True)
plt.show()
