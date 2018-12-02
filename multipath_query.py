from auxillary.path_query import execute_path_query

# DGX-1 192 H
# id_list = [4117, 4103, 4132, 4118, 4147, 4133, 4072, 4058, 4087, 4073, 4102, 4088, 4027, 4013, 4042, 4028, 4057, 4043,
#            3982, 3968, 3997, 3983, 4012, 3998, 3937, 3923, 3952, 3938, 3967, 3953, 3892, 3878, 3907, 3893, 3922, 3908,
#            3847, 3833, 3862, 3848, 3877, 3863, 3817, 3803, 3832, 3818, 3802, 3788, 3757, 3743, 3772, 3758, 3787, 3773,
#            3712, 3698, 3727, 3713, 3742, 3728, 3667, 3653, 3682, 3668, 3697, 3683]

# DGX-2 192 H
# id_list = [3076, 3062, 3091, 3077, 3106, 3092, 3031, 3017, 3046, 3032, 3061, 3047, 2986, 2972, 3001, 2987, 3016, 3002,
#            2941, 2927, 2956, 2942, 2971, 2957, 2896, 2882, 2911, 2897, 2926, 2912, 2851, 2837, 2866, 2852, 2881, 2867,
#            2806, 2792, 2821, 2807, 2836, 2822, 2776, 2762, 2791, 2777, 2761, 2747, 2716, 2702, 2731, 2717, 2746, 2732,
#            2671, 2657, 2686, 2672, 2701, 2687, 2626, 2612, 2641, 2627, 2656, 2642]

# DGX-3 128 H
id_list = [3034, 3020, 3049, 3035, 3064, 3050, 2989, 2975, 3004, 2990, 3019, 3005, 2944, 2930, 2959, 2945, 2974, 2960,
           2899, 2885, 2914, 2900, 2929, 2915, 2854, 2840, 2869, 2855, 2884, 2870, 2809, 2795, 2824, 2810, 2839, 2825,
           2764, 2750, 2779, 2765, 2794, 2780, 2734, 2720, 2749, 2735, 2719, 2705, 2674, 2660, 2689, 2675, 2704, 2690,
           2629, 2615, 2644, 2630, 2659, 2645, 2584, 2570, 2599, 2585, 2614, 2600]

dif_list = [id_list[i] - id_list[i + 1] for i in range(len(id_list) - 1)]
assert all(dif_list)

critical_rows = []
for i in range(int(len(id_list) / 2)):
    min_id = id_list[2 * i + 1]
    max_id = id_list[2 * i]
    rows = execute_path_query(min_id=min_id, max_id=max_id, do_print=False)
    unweighted_averages = [row for row in rows if row[2] == 0]
    weighted_averages = [row for row in rows if row[2] == 1]
    assert len(unweighted_averages) == len(weighted_averages)
    unweighted_averages_sorted = sorted(unweighted_averages, key=lambda row: row[4])
    weighted_averages_sorted = sorted(weighted_averages, key=lambda row: row[4])
    single_path_unweighted = unweighted_averages_sorted[0]
    single_path_weighted = weighted_averages_sorted[0]
    multi_path_unweighted = [row for row in unweighted_averages_sorted if row[4] <= 11000][-1]
    multi_path_weighted = [row for row in weighted_averages_sorted if row[4] <= 11000][-1]
    print("({0},{1})".format(min_id, max_id))
    print(single_path_unweighted)
    print(single_path_weighted)
    print(multi_path_unweighted)
    print(multi_path_weighted)
