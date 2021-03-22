import numpy as np
from algorithms.boostrap_mean_comparison import BootstrapMeanComparison

fashion_net_thin_baseline = np.array([91.80, 91.45, 91.60, 91.81, 92.02, 91.80, 91.71, 91.24, 91.50, 91.56])
fashion_net_thick_baseline = np.array([92.27, 92.60, 92.25, 92.11, 92.07, 92.24, 92.41, 91.92, 92.14, 92.50])
fashion_net_random_routing_baseline = np.array([92.00, 91.60, 91.80, 92.02, 91.75, 92.01, 92.04, 91.83, 92.11, 91.83])
fashion_net_cign_all_paths_baseline = np.array([91.88, 92.20, 91.80, 92.02, 91.77, 91.97, 92.12, 92.01, 91.92, 92.02])
fashion_net_with_no_annealing = np.array([92.23, 92.31, 92.12, 92.29, 92.40])
fashion_net_with_annealing = np.array([92.531, 92.522, 92.478, 92.172, 92.154])
fashion_net_cigj_random = np.array([92.87, 92.76, 92.61, 92.54, 92.60])

fashion_net_smoe_cign_v1 = 100.0 * np.array(
    [0.9296, 0.9286, 0.9297, 0.9304, 0.9293, 0.9298, 0.9299, 0.9295, 0.9297, 0.9287, 0.9292, 0.9286, 0.9288, 0.9295,
     0.928, 0.9281, 0.9271, 0.9267, 0.9291, 0.9283, 0.93, 0.9286, 0.9289, 0.9296, 0.9293, 0.9277, 0.9299, 0.9284, 0.928,
     0.9289, 0.9286, 0.9279, 0.9282, 0.9267, 0.9294, 0.9281, 0.9285, 0.9274, 0.9288, 0.9292, 0.9272, 0.9286, 0.9283,
     0.9292, 0.9283, 0.9261, 0.9272, 0.9294, 0.928, 0.9284, 0.9279, 0.9285, 0.9282, 0.9285, 0.9289, 0.93, 0.9281,
     0.9281, 0.9275, 0.9288, 0.9287, 0.9278, 0.9281, 0.9293, 0.9289, 0.9276, 0.9282, 0.9296, 0.9281, 0.928, 0.9281,
     0.9287, 0.9287, 0.9274, 0.9287, 0.9292, 0.9291, 0.9296, 0.9281, 0.9295, 0.9297, 0.9283, 0.9291, 0.9288, 0.9297,
     0.9294, 0.9291, 0.9299, 0.9295, 0.9295, 0.93, 0.9277, 0.9297, 0.9296, 0.9288, 0.9293, 0.928, 0.9287, 0.9285,
     0.9282, 0.9283, 0.9274, 0.928, 0.9285, 0.9293, 0.9275, 0.9287, 0.9287, 0.9279, 0.9278, 0.9289, 0.9273, 0.9294,
     0.9288, 0.9288, 0.9273, 0.9293, 0.9288, 0.9297, 0.9283, 0.9298, 0.9294, 0.9293, 0.9278, 0.9294, 0.9296, 0.9301,
     0.9298, 0.9294, 0.9296, 0.9258, 0.9253, 0.9264, 0.9248, 0.9272, 0.9257, 0.9264, 0.9261, 0.9254, 0.9265, 0.9256,
     0.9267, 0.9266, 0.9249, 0.9273, 0.9254, 0.9263, 0.9259, 0.9256, 0.9264, 0.9256, 0.9269, 0.9244, 0.9272, 0.9264,
     0.9255, 0.9254, 0.9258, 0.9259, 0.9269, 0.9263, 0.9257, 0.926, 0.9269, 0.9254, 0.9251, 0.9269, 0.9262, 0.9271,
     0.9265, 0.9254, 0.9268, 0.926, 0.9265, 0.9256, 0.9265, 0.926, 0.9254, 0.9263, 0.9225, 0.9259, 0.9283, 0.9279,
     0.9288, 0.9282, 0.9272, 0.927, 0.9273, 0.9275, 0.9281, 0.9269, 0.9272, 0.9283, 0.9276, 0.9285, 0.9288, 0.9259,
     0.9288, 0.9282, 0.9255, 0.9259, 0.9272, 0.9278, 0.9261, 0.9281, 0.9279, 0.9277, 0.9289, 0.928, 0.9272, 0.9284,
     0.9281, 0.9273, 0.928, 0.9286, 0.9265, 0.9283, 0.9278, 0.9292, 0.9278, 0.9275, 0.9283, 0.9298, 0.9289, 0.9285,
     0.9279, 0.9266, 0.9295, 0.9272, 0.927, 0.9291, 0.929, 0.9296, 0.9291, 0.9287, 0.9268, 0.9284, 0.9259, 0.9265,
     0.9283, 0.9285, 0.927, 0.9284, 0.9271, 0.9287, 0.9281, 0.9282, 0.9272, 0.9283, 0.9271, 0.928, 0.9286, 0.9267,
     0.9278, 0.9271, 0.9281, 0.9285, 0.9279, 0.9277, 0.9283, 0.9281, 0.9287, 0.9262, 0.9295, 0.9276, 0.9273, 0.9273,
     0.9281, 0.9283, 0.9279, 0.9276, 0.9282, 0.9277, 0.9279, 0.9281, 0.9281, 0.927, 0.9263, 0.9289, 0.9296, 0.9275,
     0.9284, 0.9297, 0.9281, 0.9286, 0.9287, 0.9282, 0.928, 0.9273, 0.9285, 0.9285, 0.9286, 0.9275, 0.9277, 0.9274,
     0.928, 0.9267, 0.9264, 0.9253, 0.9267, 0.9259, 0.9257, 0.9254, 0.9265, 0.9271, 0.9256, 0.9275, 0.9263, 0.9276,
     0.9277, 0.9261, 0.9269, 0.9275, 0.9274, 0.9268, 0.9275, 0.9275, 0.9267, 0.9272, 0.927, 0.9276, 0.9274, 0.9271,
     0.9277, 0.9268, 0.9279, 0.9266, 0.9267, 0.9268, 0.9252, 0.928, 0.9269, 0.9268, 0.9267, 0.9268, 0.9272, 0.9277,
     0.9273, 0.9277, 0.926, 0.927, 0.9268, 0.9268, 0.9269, 0.9271, 0.929, 0.9292, 0.9291, 0.9281, 0.9281, 0.9276,
     0.9278, 0.9261, 0.9273, 0.9274, 0.9273, 0.9262, 0.9271, 0.9278, 0.9277, 0.9273, 0.9271, 0.9283, 0.9258, 0.9271,
     0.9275, 0.9254, 0.9275, 0.9273, 0.9263, 0.9264, 0.9275, 0.9291, 0.927, 0.9275, 0.9281, 0.9281, 0.9269, 0.9247,
     0.9253, 0.9269, 0.9262, 0.925, 0.9258, 0.9263, 0.9263, 0.9252, 0.925, 0.9254, 0.9251, 0.9239, 0.9277, 0.9255,
     0.9274, 0.9252, 0.9244, 0.9258, 0.927, 0.9274, 0.9267, 0.9269, 0.9271, 0.9276, 0.9266, 0.9259, 0.9273, 0.9269,
     0.9282, 0.9275, 0.9272, 0.926, 0.9275, 0.927, 0.9264, 0.9266, 0.9265, 0.9258, 0.9258, 0.926, 0.9258, 0.9256,
     0.9259, 0.9256, 0.9259, 0.926, 0.9266, 0.9272, 0.9256, 0.926, 0.9253, 0.9264, 0.9261, 0.9252, 0.9259, 0.9252,
     0.9244, 0.9241, 0.9246, 0.924, 0.9246, 0.9265, 0.9261, 0.9262, 0.9261, 0.9253, 0.9254, 0.9263, 0.9258, 0.9264,
     0.9263, 0.9261, 0.9253, 0.9262, 0.9264, 0.9261, 0.9252, 0.9266, 0.925, 0.9262, 0.9253, 0.9234, 0.9254, 0.926,
     0.9252, 0.9255, 0.9244, 0.9266, 0.9258, 0.9254, 0.9248, 0.9242, 0.9256, 0.9264, 0.9267, 0.9247, 0.9267, 0.9251,
     0.9244, 0.9254, 0.9249, 0.9255, 0.9261, 0.9263, 0.9248, 0.9267, 0.9258, 0.9262, 0.9262, 0.9265, 0.9258, 0.9256,
     0.9249, 0.9253, 0.9258, 0.9258, 0.9271, 0.9261, 0.9261, 0.9253, 0.9248, 0.9256, 0.9249, 0.9249, 0.925, 0.9256,
     0.9255, 0.9243, 0.9257, 0.9267, 0.9258, 0.9259, 0.9256, 0.9272, 0.9272, 0.9256, 0.9264, 0.9255, 0.9256, 0.9257,
     0.9268, 0.9245, 0.9263, 0.9258, 0.9264, 0.9265, 0.9251, 0.9275, 0.9256, 0.9256, 0.9246, 0.9258, 0.9248, 0.9251,
     0.9237, 0.9257, 0.9261, 0.9244, 0.9253, 0.9261, 0.9249, 0.9236, 0.9251, 0.9261, 0.9251, 0.9249, 0.9274, 0.9281,
     0.9266, 0.9274, 0.9275, 0.9275, 0.9274, 0.9277, 0.9274, 0.9279, 0.928, 0.9291, 0.9275, 0.9271, 0.9273, 0.9268,
     0.9293, 0.9269, 0.9289, 0.9289])
fashion_net_smoe_cign_v2 = 100.0 * np.array(
    [0.9289, 0.9279, 0.9284, 0.9277, 0.9293, 0.928, 0.9289, 0.9283, 0.9292, 0.9298, 0.9262, 0.9275, 0.9269, 0.9279,
     0.9284, 0.9282, 0.9288, 0.9284, 0.9281, 0.9292, 0.928, 0.9267, 0.9281, 0.9279, 0.9266, 0.9291, 0.9272, 0.9277,
     0.9281, 0.9271, 0.9271, 0.926, 0.9283, 0.9275, 0.9268, 0.9292, 0.9271, 0.9276, 0.9258, 0.9264, 0.927, 0.9281,
     0.9287, 0.928, 0.929, 0.928, 0.9272, 0.9274, 0.9286, 0.929, 0.9266, 0.9281, 0.9271, 0.9267, 0.9271, 0.9268, 0.9267,
     0.9272, 0.9291, 0.9266, 0.927, 0.9275, 0.928, 0.9278, 0.9266, 0.9279, 0.9282, 0.9264, 0.9276, 0.9282, 0.9277,
     0.9276, 0.9281, 0.9272, 0.9298, 0.9271, 0.9266, 0.9276, 0.9272, 0.9274, 0.9273, 0.9269, 0.9276, 0.9283, 0.9278,
     0.927, 0.9279, 0.9279, 0.9285, 0.9283, 0.9274, 0.9295, 0.9265, 0.9283, 0.9286, 0.9288, 0.9284, 0.9295, 0.9285,
     0.9265, 0.9282, 0.9285, 0.9272, 0.928, 0.9264, 0.9273, 0.9283, 0.9268, 0.9282, 0.9272, 0.9279, 0.9282, 0.9278,
     0.9283, 0.9274, 0.9271, 0.9286, 0.9285, 0.9274, 0.9283, 0.9292, 0.9286, 0.9269, 0.9252, 0.9248, 0.9247, 0.9234,
     0.925, 0.925, 0.9248, 0.9262, 0.9244, 0.9253, 0.9246, 0.9246, 0.9253, 0.9238, 0.9245, 0.9241, 0.9252, 0.9231,
     0.9239, 0.9246, 0.9237, 0.9238, 0.9258, 0.9245, 0.925, 0.9241, 0.926, 0.9251, 0.9242, 0.924, 0.9236, 0.9255,
     0.9254, 0.9253, 0.9254, 0.924, 0.9245, 0.923, 0.925, 0.9254, 0.9248, 0.9259, 0.9246, 0.9238, 0.9245, 0.9247,
     0.9247, 0.9244, 0.9264, 0.9261, 0.9251, 0.9243, 0.9248, 0.925, 0.9247, 0.927, 0.9269, 0.9258, 0.9272, 0.9278,
     0.928, 0.9271, 0.9263, 0.9255, 0.9268, 0.9273, 0.9269, 0.9277, 0.9283, 0.9263, 0.9273, 0.9278, 0.9256, 0.9273,
     0.9261, 0.9246, 0.9274, 0.9279, 0.9263, 0.9263, 0.9265, 0.9267, 0.9258, 0.9276, 0.9257, 0.9264, 0.9275, 0.9268,
     0.9269, 0.9261, 0.9275, 0.926, 0.9265, 0.926, 0.9279, 0.926, 0.9257, 0.9267, 0.927, 0.9268, 0.9273, 0.927, 0.9268,
     0.9265, 0.9266, 0.9265, 0.9268, 0.9274, 0.9263, 0.9255, 0.9259, 0.9259, 0.9263, 0.9273, 0.9275, 0.9265, 0.9262,
     0.9263, 0.9268, 0.927, 0.9271, 0.9266, 0.927, 0.9275, 0.927, 0.9255, 0.9263, 0.9261, 0.9261, 0.9264, 0.9266,
     0.9252, 0.9256, 0.9262, 0.9256, 0.9261, 0.9271, 0.9268, 0.9268, 0.9264, 0.9267, 0.9271, 0.9269, 0.9282, 0.9265,
     0.9272, 0.9274, 0.9269, 0.9263, 0.9274, 0.9268, 0.9274, 0.9276, 0.927, 0.9259, 0.9269, 0.9272, 0.9268, 0.9267,
     0.927, 0.9273, 0.9269, 0.9272, 0.9273, 0.9271, 0.9275, 0.9281, 0.9272, 0.9268, 0.9268, 0.927, 0.9267, 0.9271,
     0.9265, 0.9252, 0.9257, 0.9258, 0.926, 0.9253, 0.9255, 0.9258, 0.9264, 0.9256, 0.9269, 0.9272, 0.9272, 0.9269,
     0.9269, 0.9261, 0.9262, 0.927, 0.9276, 0.9279, 0.9269, 0.9276, 0.9275, 0.9261, 0.9268, 0.9273, 0.9277, 0.9271,
     0.9269, 0.9255, 0.9269, 0.9264, 0.9277, 0.9278, 0.9269, 0.9271, 0.9273, 0.9263, 0.9264, 0.9278, 0.926, 0.927,
     0.9271, 0.9268, 0.9272, 0.927, 0.9265, 0.9267, 0.9274, 0.9272, 0.9271, 0.9271, 0.9271, 0.9274, 0.9277, 0.9273,
     0.9274, 0.9274, 0.9276, 0.9271, 0.9278, 0.9278, 0.9276, 0.9276, 0.927, 0.9264, 0.9265, 0.9266, 0.9275, 0.9272,
     0.9274, 0.9266, 0.9268, 0.9269, 0.9265, 0.9272, 0.9254, 0.927, 0.9268, 0.9261, 0.9265, 0.9237, 0.9242, 0.9257,
     0.9239, 0.9236, 0.9259, 0.9237, 0.9258, 0.925, 0.9242, 0.9276, 0.9275, 0.9235, 0.9245, 0.9253, 0.9249, 0.9253,
     0.9245, 0.9253, 0.9268, 0.9273, 0.9275, 0.9268, 0.9263, 0.9266, 0.9277, 0.927, 0.9276, 0.9271, 0.9263, 0.9257,
     0.9274, 0.9268, 0.9263, 0.9267, 0.9263, 0.9257, 0.9249, 0.925, 0.9259, 0.9248, 0.9235, 0.9255, 0.9251, 0.9246,
     0.9243, 0.9254, 0.9248, 0.9246, 0.9257, 0.9252, 0.9247, 0.9251, 0.9245, 0.9257, 0.9246, 0.9239, 0.9236, 0.924,
     0.9243, 0.9246, 0.9237, 0.9263, 0.9242, 0.9235, 0.9244, 0.9254, 0.9252, 0.925, 0.925, 0.924, 0.9245, 0.9234,
     0.9253, 0.9255, 0.9249, 0.9255, 0.925, 0.9254, 0.9249, 0.925, 0.9248, 0.9242, 0.9246, 0.9249, 0.9239, 0.9244,
     0.9254, 0.9255, 0.9247, 0.9245, 0.9256, 0.9263, 0.9252, 0.9254, 0.9236, 0.9242, 0.9255, 0.9246, 0.9246, 0.9257,
     0.9241, 0.9253, 0.9243, 0.9239, 0.9254, 0.9239, 0.9241, 0.9255, 0.9255, 0.9264, 0.9239, 0.9248, 0.9236, 0.9239,
     0.9256, 0.9259, 0.9248, 0.9247, 0.9234, 0.9234, 0.9246, 0.9247, 0.9236, 0.9261, 0.9243, 0.9257, 0.9259, 0.9245,
     0.9244, 0.9249, 0.9235, 0.9257, 0.9242, 0.9257, 0.9237, 0.9251, 0.9255, 0.9238, 0.9253, 0.9238, 0.9258, 0.9245,
     0.9246, 0.9259, 0.9258, 0.9263, 0.9246, 0.9229, 0.9232, 0.925, 0.9236, 0.9249, 0.9246, 0.9237, 0.9246, 0.9258,
     0.9241, 0.9242, 0.9244, 0.9232, 0.9247, 0.9276, 0.9278, 0.9273, 0.927, 0.9273, 0.9273, 0.9272, 0.9277, 0.9279,
     0.9276, 0.928, 0.9279, 0.9268, 0.9271, 0.9265, 0.9273, 0.9263, 0.9273, 0.927, 0.9273])
fashion_net_smoe_cign_v3 = 100.0 * np.array(
    [0.9269, 0.928, 0.9265, 0.9276, 0.9279, 0.9272, 0.9268, 0.9273, 0.927, 0.9267, 0.9273, 0.9273, 0.9273, 0.9271,
     0.9269, 0.9267, 0.9288, 0.9271, 0.9267, 0.928, 0.9275, 0.9271, 0.9271, 0.9262, 0.9258, 0.9248, 0.9262, 0.9255,
     0.9256, 0.9258, 0.9266, 0.9261, 0.9259, 0.9258, 0.9256, 0.9258, 0.9263, 0.9265, 0.9269, 0.9255, 0.9259, 0.9261,
     0.9261, 0.9265, 0.9273, 0.9262, 0.9278, 0.9268, 0.9269, 0.9266, 0.9263, 0.926, 0.9273, 0.9262, 0.9266, 0.9271,
     0.9256, 0.9262, 0.9252, 0.9264, 0.9263, 0.9261, 0.9274, 0.9273, 0.9266, 0.9279, 0.928, 0.9261, 0.9258, 0.9263,
     0.9269, 0.926, 0.927, 0.9276, 0.9278, 0.9269, 0.9282, 0.9262, 0.927, 0.9267, 0.9264, 0.9275, 0.9264, 0.9262,
     0.9259, 0.9276, 0.9261, 0.9269, 0.9267, 0.9277, 0.928, 0.9278, 0.9274, 0.9266, 0.9259, 0.9266, 0.9277, 0.9264,
     0.9262, 0.927, 0.9264, 0.927, 0.9267, 0.927, 0.9269, 0.9263, 0.9257, 0.9264, 0.9269, 0.9274, 0.9265, 0.9259,
     0.9257, 0.9261, 0.9258, 0.9249, 0.9254, 0.9256, 0.9256, 0.926, 0.9259, 0.926, 0.927, 0.9266, 0.9237, 0.9238,
     0.9233, 0.9239, 0.9246, 0.9234, 0.9242, 0.9236, 0.9244, 0.9229, 0.9235, 0.9241, 0.9244, 0.9233, 0.9232, 0.9231,
     0.9238, 0.9245, 0.9231, 0.9238, 0.9232, 0.9231, 0.9237, 0.9245, 0.9222, 0.9232, 0.9234, 0.9228, 0.9238, 0.9239,
     0.9249, 0.9235, 0.9249, 0.9236, 0.9236, 0.9237, 0.9232, 0.9242, 0.9242, 0.9241, 0.9244, 0.9239, 0.9236, 0.9243,
     0.9233, 0.9237, 0.924, 0.9242, 0.9249, 0.9241, 0.9237, 0.9239, 0.9231, 0.9233, 0.9261, 0.9262, 0.9275, 0.9271,
     0.9279, 0.9251, 0.9261, 0.9268, 0.9268, 0.9266, 0.9253, 0.9267, 0.9265, 0.9262, 0.9267, 0.9271, 0.9249, 0.9279,
     0.9249, 0.9269, 0.926, 0.9252, 0.9252, 0.9249, 0.9253, 0.9256, 0.9253, 0.925, 0.9255, 0.9275, 0.9274, 0.9265,
     0.9248, 0.9247, 0.9257, 0.9254, 0.9255, 0.9259, 0.9252, 0.9252, 0.9264, 0.9249, 0.9261, 0.9259, 0.9255, 0.9262,
     0.9258, 0.9253, 0.926, 0.925, 0.9248, 0.926, 0.9261, 0.9268, 0.9241, 0.9254, 0.9252, 0.9259, 0.9253, 0.9254,
     0.9259, 0.9254, 0.9249, 0.926, 0.925, 0.925, 0.9252, 0.9257, 0.9251, 0.9251, 0.9258, 0.9248, 0.9257, 0.9238,
     0.9252, 0.9255, 0.9268, 0.9265, 0.9264, 0.9252, 0.9262, 0.9268, 0.9258, 0.9245, 0.9256, 0.9258, 0.9254, 0.9258,
     0.9263, 0.9259, 0.9265, 0.9269, 0.9261, 0.9261, 0.9259, 0.9259, 0.9252, 0.9255, 0.9255, 0.9263, 0.9249, 0.9253,
     0.9268, 0.9255, 0.9262, 0.9269, 0.9273, 0.9254, 0.9253, 0.9265, 0.9264, 0.9262, 0.9263, 0.927, 0.9261, 0.9253,
     0.9265, 0.9251, 0.9254, 0.926, 0.9249, 0.9244, 0.9257, 0.9261, 0.9256, 0.926, 0.9262, 0.9251, 0.9265, 0.926,
     0.9271, 0.9268, 0.9271, 0.9261, 0.9266, 0.9262, 0.9269, 0.9265, 0.9267, 0.9264, 0.9268, 0.9254, 0.9274, 0.9264,
     0.9258, 0.927, 0.9272, 0.9266, 0.9267, 0.926, 0.9267, 0.9256, 0.9265, 0.9269, 0.9264, 0.9266, 0.9269, 0.927,
     0.9265, 0.9269, 0.9263, 0.9269, 0.9263, 0.9261, 0.9267, 0.9268, 0.9257, 0.9267, 0.9272, 0.9272, 0.9268, 0.9262,
     0.9267, 0.9269, 0.9269, 0.9271, 0.927, 0.9266, 0.9267, 0.9267, 0.9256, 0.9273, 0.9264, 0.9265, 0.9259, 0.9262,
     0.9272, 0.9263, 0.9259, 0.9265, 0.927, 0.9269, 0.9264, 0.9265, 0.9272, 0.9263, 0.9264, 0.9256, 0.9225, 0.9228,
     0.9224, 0.9229, 0.9227, 0.9245, 0.9234, 0.9229, 0.9244, 0.9231, 0.9236, 0.9232, 0.9265, 0.9234, 0.9244, 0.9231,
     0.9232, 0.9228, 0.9244, 0.9273, 0.9271, 0.9265, 0.9265, 0.9276, 0.9272, 0.9269, 0.9252, 0.9271, 0.9267, 0.9269,
     0.9262, 0.9254, 0.9267, 0.9265, 0.9257, 0.9235, 0.9238, 0.9235, 0.9224, 0.9229, 0.925, 0.9237, 0.9239, 0.9238,
     0.9231, 0.9243, 0.925, 0.9231, 0.9225, 0.9226, 0.9236, 0.9252, 0.9229, 0.9244, 0.9241, 0.9239, 0.9227, 0.924,
     0.9239, 0.9231, 0.9222, 0.9242, 0.9232, 0.923, 0.9222, 0.9249, 0.9238, 0.9234, 0.924, 0.9251, 0.9232, 0.9235,
     0.9241, 0.925, 0.9231, 0.9225, 0.9244, 0.9233, 0.924, 0.924, 0.9241, 0.9228, 0.9232, 0.9221, 0.9236, 0.9231,
     0.9235, 0.9247, 0.9232, 0.9239, 0.9229, 0.9235, 0.9223, 0.925, 0.9254, 0.9236, 0.9234, 0.9251, 0.924, 0.9251,
     0.9219, 0.9228, 0.9246, 0.9254, 0.9243, 0.924, 0.9226, 0.9243, 0.9236, 0.9228, 0.924, 0.9242, 0.9211, 0.923,
     0.9236, 0.9241, 0.9238, 0.9211, 0.9239, 0.9228, 0.924, 0.9236, 0.9224, 0.9234, 0.9229, 0.9223, 0.9215, 0.923,
     0.9249, 0.9243, 0.9238, 0.9227, 0.9234, 0.9213, 0.9229, 0.9233, 0.9243, 0.9223, 0.9223, 0.9229, 0.9226, 0.9223,
     0.923, 0.9232, 0.9242, 0.9233, 0.9229, 0.9208, 0.9223, 0.9221, 0.9233, 0.9235, 0.9231, 0.922, 0.9229, 0.9231,
     0.9221, 0.9233, 0.9223, 0.9238, 0.922, 0.9242, 0.9226, 0.9264, 0.9271, 0.9268, 0.9265, 0.9271, 0.9266, 0.926,
     0.9265, 0.9271, 0.9272, 0.9262, 0.9264, 0.9273, 0.9263, 0.9256, 0.9264, 0.9266, 0.9263, 0.9269, 0.927])

fashion_cigj_results = np.array([93.26, 93.12, 93.09, 93.04, 93.03])

print(np.mean(fashion_net_thin_baseline))
print(np.mean(fashion_net_thick_baseline))
print(np.mean(fashion_net_random_routing_baseline))
print(np.mean(fashion_net_cign_all_paths_baseline))
print(np.mean(fashion_net_with_no_annealing))
print(np.mean(fashion_net_with_annealing))
print(np.mean(fashion_net_smoe_cign_v1))
print(np.mean(fashion_net_smoe_cign_v2))
print(np.mean(fashion_net_smoe_cign_v3))
print(np.mean(fashion_net_cigj_random))
print(np.mean(fashion_cigj_results))

data_samples = {"fashion_net_thin_baseline": fashion_net_thin_baseline,
                "fashion_net_thick_baseline": fashion_net_thick_baseline,
                "fashion_net_random_routing_baseline": fashion_net_random_routing_baseline,
                "fashion_net_cign_all_paths_baseline": fashion_net_cign_all_paths_baseline,
                "fashion_net_with_no_annealing": fashion_net_with_no_annealing,
                "fashion_net_with_annealing": fashion_net_with_annealing,
                "fashion_net_smoe_cign_v1": fashion_net_smoe_cign_v1,
                "fashion_net_smoe_cign_v2": fashion_net_smoe_cign_v2,
                "fashion_net_smoe_cign_v3": fashion_net_smoe_cign_v3,
                "fashion_cigj_results": fashion_cigj_results,
                "fashion_net_cigj_random": fashion_net_cigj_random}

# baselines = ["fashion_net_thin_baseline", "fashion_net_thick_baseline", "fashion_net_random_routing_baseline",
#              "fashion_net_cign_all_paths_baseline"]
baselines = ["fashion_net_thin_baseline", "fashion_net_thick_baseline", "fashion_net_cigj_random",
             "fashion_net_with_annealing"]
# cigns = ["fashion_net_with_no_annealing", "fashion_net_with_annealing",
#          "fashion_net_smoe_cign_v1", "fashion_net_smoe_cign_v2", "fashion_net_smoe_cign_v3",
#          "fashion_cigj_results"]

cigns = ["fashion_cigj_results"]
for cign_method in cigns:
    print("**********CIGN method:{0}**********".format(cign_method))
    for baseline_method in baselines:
        print("Comparing {0} vs {1}".format(cign_method, baseline_method))
        cign_arr = data_samples[cign_method]
        baseline_arr = data_samples[baseline_method]
        p_value, reject_null_hypothesis = BootstrapMeanComparison.compare(x=cign_arr, y=baseline_arr,
                                                                          boostrap_count=100000)
        print("p-value:{0} Reject H0 for equal means:{1}".format(p_value, reject_null_hypothesis))

x = np.random.uniform(low=0.0, high=10.0, size=(1000,))
y = np.random.uniform(low=20.0, high=40.0, size=(1250,))

p_value, reject_null_hypothesis = BootstrapMeanComparison.compare(x=x, y=y, boostrap_count=10000)
