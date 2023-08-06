import inspect


class Config(dict):
    def __init__(self, name, **kwargs):
        super(Config, self).__init__()
        self.name = name
        self.__dict__.update(**kwargs)

    def __setattr__(self, key, value):
        print('\n__setattr__:', inspect.stack()[1][3], key, value)
        if inspect.stack()[1][3] == '__init__':
            return super().__setattr__(key, value)
        if key == 'name':
            raise AttributeError('name is unchangeable')
        return super().__setattr__(key, value)

    def __getattribute__(self, attr):
        try:
            return super().__getattribute__(attr)
        except AttributeError:
            return None

    def __setitem__(self, key, value):
        print('__setitem__:', inspect.stack()[1][3], key, value)
        return super().__setitem__(key, value)

    def update(self, **kwargs):
        print('update:', inspect.stack()[1], kwargs)
        return super().update(**kwargs)

    def update_class(self, **kwargs):
        for key, item in kwargs.items():
            self.__setattr__(key, item)


class BlockConfig(Config):
    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)

    def __setitem__(self, key, value):
        if inspect.stack()[1][3] == 'add_layer_config':
            return super().__setitem__(key, value)
        raise ValueError

    def add_layer_config(self, name, **class_kw):
        con_new = Config(name, **class_kw)
        self[name] = con_new

    def update_layer_config(self, name: str, **layer_kw):
        con = self.get(name)
        if con is None:
            raise KeyError(f'unknown layer {name}')
        con.update(**layer_kw)



if __name__ == '__main__':
    con = Config('foo')
    print('--------------------------------------------')
    con.y = 0
    print('--------------------------------------------')
    con['x'] = 9
    print('--------------------------------------------')
    assert con.k is None
    print('--------------------------------------------')
    con.update_class(f=8)

    block = BlockConfig('b', ee='ee')
    block.add_layer_config('k', u=5)
    block.w = 90
    block['k'].update(f=90)
    block.update_layer_config('k', o=90)