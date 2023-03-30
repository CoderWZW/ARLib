import os.path
from os import remove
from re import split


class FileIO(object):
    def __init__(self):
        pass

    @staticmethod
    def write_file(dir, file, content, op='w'):
        if not os.path.exists(dir):
            os.makedirs(dir)
        with open(dir + file, op) as f:
            f.writelines(content)

    @staticmethod
    def delete_file(file_path):
        if os.path.exists(file_path):
            remove(file_path)

    @staticmethod
    def load_data_set(file):
        data = []
        with open(file) as f:
            for line in f:
                items = split(' ', line.strip())
                user_id = items[0]
                item_id = items[1]
                weight = items[2]
                data.append([user_id, item_id, float(weight)])
        return data

