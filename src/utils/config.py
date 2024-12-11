class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def override(self, attrs):
        if isinstance(attrs, dict):
            self.__dict__.update(**attrs)
        elif isinstance(attrs, (list, tuple, set)):
            for attr in attrs:
                self.override(attr)
        elif attrs is not None:
            raise NotImplementedError
        return self
model_params = AttrDict(
    depth=1,
    heads=3,
    dim_head=64,
    mlp_dim=64,
    dim=6
)
# model_params = AttrDict(
#     depth=3,
#     heads=3,
#     dim_head=64,
#     mlp_dim=64,
#     dim=6
# )