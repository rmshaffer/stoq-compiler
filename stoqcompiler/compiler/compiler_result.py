class CompilerResult:
    def __init__(self, compiled_sequence, cost_by_step, total_elapsed_time):
        self.compiled_sequence = compiled_sequence
        self.cost_by_step = cost_by_step
        self.total_elapsed_time = total_elapsed_time