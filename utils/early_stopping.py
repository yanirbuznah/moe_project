class EarlyStopping:
    def __init__(self, tolerance=5, min_delta=0):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_score, validation_score, less_is_better=True):
        diff_score = train_score - validation_score
        if less_is_better:
            diff_score *= -1
        if diff_score > self.min_delta:
            self.counter += 1
            if self.counter >= self.tolerance:
                self.early_stop = True
        else:
            self.counter = 0
