import numpy as np

tower_count = 3
iteration_count = 500
momentum = 0.9

values = [np.random.uniform(low=-20, high=50, size=(iteration_count,)) for _ in range(tower_count)]

# Algorithm 1
mov_avg_1 = 0
for i in range(iteration_count):
    curr_avg = np.mean(np.array([values[k][i] for k in range(tower_count)]))
    if i == 0:
        mov_avg_1 = curr_avg
    else:
        mov_avg_1 = mov_avg_1 * momentum + curr_avg * (1.0 - momentum)

# Algorithm 2
mov_avg_2 = 0
moving_averages = [0] * tower_count
for i in range(iteration_count):
    for k in range(tower_count):
        curr_val = values[k][i]
        if i == 0:
            moving_averages[k] = curr_val
        else:
            moving_averages[k] = moving_averages[k] * momentum + curr_val * (1.0 - momentum)
mov_avg_2 = np.mean(np.array(moving_averages))


    # curr_avg = np.mean(np.array([values[k][i] for k in range(tower_count)]))
    # if i == 0:
    #     mov_avg = curr_avg
    # else:
    #     mov_avg = mov_avg * momentum + curr_avg * (1.0 - momentum)

print("X")
