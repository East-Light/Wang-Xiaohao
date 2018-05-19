from __future__ import print_function
import mysql.connector


class DataBase(object):
    """

    """

    def __init__(self):
        pass

    def connect_db(self, ):
        """

        :return:
        """
        print("connecting...")
        connect = mysql.connector.connect(user='root',
                                          password='1203',
                                          host='120.24.90.180',
                                          database='securitydrive', # 数据库名是什么我不知道
                                          use_unicode=True)
        cur = connect.cursor()
        print("database connect success")
        return cur, connect

    def query_data_set(self, cur, query, params):
        """

        :param cur:
        :param query:
        :return:
        """
        cur.execute(query, params)
        return cur.fetchall()


if __name__ == '__main__':
    import random
    db = DataBase()
    cur, connect = db.connect_db()
    query = 'INSERT INTO Meal(user_id, food_code, meal_type, eat_time) VALUES (%s, %s, %s, %s)'
    food_code = [100000007, 100000009, 100000015, 100000019, 100000024, 100000030, 100000032, 100000067, 100000081, 100000083, 100000098, 100000101, 100000103]
    meal_time = [1, 2, 3]
    user_id = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    eat_time = ['2018-01-02', '2018-01-03', '2018-01-04', '2018-01-05', '2018-01-06', '2018-02-01', '2018-02-02', '2018-01-22']
    food_code_len = len(food_code); meal_time_len = len(meal_time); user_id_len = len(user_id); eat_time_len = len(eat_time)
    count = 1
    print("insert data ...")
    for i in range(1000):
        food_code_index = random.randint(0, food_code_len - 1)
        meal_time_index = random.randint(0, meal_time_len - 1)
        user_id_index = random.randint(0, user_id_len - 1)
        eat_time_index = random.randint(0, eat_time_len - 1)
        insert_data = [user_id[user_id_index], food_code[food_code_index], meal_time[meal_time_index], eat_time[eat_time_index]]
        cur.execute(query, insert_data)
        connect.commit()
        count += 1

    print("insert all data success ...\n insert %d data" % count)
