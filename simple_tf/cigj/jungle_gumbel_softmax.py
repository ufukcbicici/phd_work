from simple_tf.cigj.jungle_no_stitch import JungleNoStitch


class JungleGumbelSoftmax(JungleNoStitch):
    def __init__(self, node_build_funcs, h_funcs, grad_func, threshold_func, residue_func, summary_func, degree_list,
                 dataset):
        super().__init__(node_build_funcs, h_funcs, grad_func, threshold_func, residue_func, summary_func, degree_list,
                         dataset)
        # self.unitTestList = [self.test_stitching]


