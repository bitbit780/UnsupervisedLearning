
class DimReducerBase:
    def __init__(self, **params):
        self.model = self.get_model(**params)

    def get_model(self, **params):
        raise NotImplementedError
    
    def run(self, x_train, x_validation=None):
        x_train_analyzed = self.model.fit_transform(x_train)
        x_validation_analyzed = None
        if x_validation is not None and hasattr(self.model, 'transform'):
            x_validation_analyzed = self.model.transform(x_validation)
        
        return x_train_analyzed, x_validation_analyzed