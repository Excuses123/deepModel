# -*- coding:utf-8 -*-
"""
Author:
    LuJie, 597906300@qq.com
"""


import pymysql
from sqlalchemy import create_engine


class MySql(object):
    """
    mysql工具类
    """

    def __init__(self, host, port, user, password, db, charset):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.db = db
        self.charset = charset

    def insert_df(self, data, table, if_exists='append', chunksize=10000):
        """engine模式插入dataframe"""
        data.to_sql(table, con=self.engine, index=False, if_exists=if_exists, chunksize=chunksize)

    def insert(self, sql):
        """执行单条数据插入"""
        self.cursor.execute(sql)
        self.connect.commit()

    def insertmany(self, sql, data):
        """执行多条数据插入"""
        self.cursor.executemany(sql, data)
        self.connect.commit()

    def execute(self, sql):
        self.cursor.execute(sql)

    def execute_engine(self, sql):
        self.engine.execute(sql)

    def open(self):
        self.connect = pymysql.connect(host=self.host,
                                       port=self.port,
                                       user=self.user,
                                       password=self.password,
                                       db=self.db,
                                       charset=self.charset)

        self.cursor = self.connect.cursor()

    def close(self):
        self.cursor.close()
        self.connect.close()

    def open_engine(self):
        db_info = {
            'user': self.user,
            'password': self.password,
            'host': self.host,
            'port': self.port,
            'database': self.db
        }

        self.engine = create_engine(
            'mysql+pymysql://%(user)s:%(password)s@%(host)s:%(port)d/%(database)s?charset=utf8' % db_info,
            encoding='utf-8')
