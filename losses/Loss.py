

class Loss:
    def __init__(self):
        self.stat = 0.0

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def __repr__(self):
        return f'{self.__class__.__name__} with {self.stat:.2f}'