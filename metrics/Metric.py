class Metric:

    def compute(self):
        pass

    def reset(self):
        pass

    def get_name(self):
        return self.__class__.__name__
