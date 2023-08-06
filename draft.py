import inspect


class Setup(dict):
    def __init__(self, name, **kwargs):
        super().__init__()
        self.name = name
        self.__dict__.update(**kwargs)

    def __setattr__(self, key, value):
        print('__setattr__:', inspect.stack()[1][3], key, value)
        return super().__setattr__(key, value)

    def __getattribute__(self, attr):
        print('__getattribute__:', inspect.stack()[1][3], attr)
        return super().__getattribute__(attr)

    def __setitem__(self, key, value):
        print('__setitem__:', inspect.stack()[1][3], key, value)
        return super().__setitem__(key, value)

    def __getitem__(self, item):
        print('__getitem__:', inspect.stack()[1][3], item)
        return super().__getitem__(item)


if __name__ == '__main__':
    con = Setup('foo')
    con.y = 0
    con['x'] = 9
    con.y
    con['x']
