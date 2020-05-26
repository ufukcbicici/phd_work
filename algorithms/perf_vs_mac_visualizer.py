import matplotlib.pyplot as plt


class PerfvsMacGraph:
    def __init__(self):
        pass

    @staticmethod
    def draw(lambda_list, accuracy_list, mac_cost):
        fig, ax1 = plt.subplots()

        color = 'tab:red'
        ax1.set_xlabel('lambda (λ)')
        ax1.set_ylabel('Accuracy (%)', color=color)
        ax1.plot(lambda_list, accuracy_list, color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()

        color = 'tab:blue'
        ax2.set_ylabel('Mac Cost (%)', color=color)  # we already handled the x-label with ax1
        ax2.plot(lambda_list, mac_cost, color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.show()


lambda_list = [1.00, 0.99, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6]
accuracy_list = [92.7167, 92.6304, 92.5526, 92.5194, 92.5000, 92.4896, 92.4802, 92.4760, 92.4692, 92.4578]
mac_cost = [40.9209, 7.3044, 1.4018, 0.6435, 0.3748, 0.2569, 0.1891, 0.1452, 0.1315, 0.0973]

# accuracy_list = [99.4596, 99.4338, 99.4102, 99.4009, 99.4005, 99.4006, 99.4021, 99.4034, 99.4026, 99.4015]
# mac_cost = [31.9819, 3.0561, 0.3929, 0.2489, 0.2136, 0.1288, 0.0917, 0.0640, 0.0416, 0.0340]

PerfvsMacGraph.draw(lambda_list=lambda_list, accuracy_list=accuracy_list, mac_cost=mac_cost)


