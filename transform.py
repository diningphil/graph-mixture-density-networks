class ZincPreprocess:
    def __call__(self, data):
        # For regression problems
        if len(self.data.y.shape) == 1:
            self.data.y = self.data.y.unsqueeze(1)


class AlchemyFullPreprocess:
    def __call__(self, data):
        # For regression problems
        if len(self.data.y.shape) == 1:
            self.data.y = self.data.y.unsqueeze(1)

        # Normalize all target variables (just for stability purposes)
        mean = self.data.y.mean(0).unsqueeze(0)
        std = self.data.y.std(0).unsqueeze(0)
        self.data.y = (self.data.y - mean) / std
