# -*- coding:utf-8 -*-
"""
Author:
    LuJie, 597906300@qq.com
"""



class Param(object):
    """
    模型参数类
    """
    def __init__(self, epoch, batch_size, model_path, **kwargs):

        self.epoch = epoch

        self.batch_size = batch_size

        self.model_path = model_path

        for k, v in kwargs.items():
            self.__setattr__(k, v)

    def set(self, name, value):
        self.__setattr__(name, value)

    def contains(self, key):

        return hasattr(self, key)

    def to_dict(self):
        """
        对象转换成字典
        """
        _dict = {}

        for name in dir(self):

            value = getattr(self, name)

            if not name.startswith('__') and not callable(value):

                _dict[name] = value

        return _dict

