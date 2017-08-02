class Runtime:
    node_scope_stack = []

    @staticmethod
    def push_node(node):
        Runtime.node_scope_stack.append(node)

    @staticmethod
    def pop_node():
        Runtime.node_scope_stack.pop()

    @staticmethod
    def get_context_name():
        ctx_name = ""
        for index in range(len(Runtime.node_scope_stack)):
            node = Runtime.node_scope_stack[index]
            if index == 0:
                ctx_name = node.indicatorText
            else:
                ctx_name = ctx_name + "/" + node.indicatorText
        return ctx_name
