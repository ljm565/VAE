import json



class Config:
    def __init__(self, path):
        self.path = path
        with open(self.path, 'r') as f:
            data = json.load(f)
            self.__dict__.update(data)


    @property
    def dict(self):
        return self.__dict__



if __name__ == '__main__':
    config_path = './Practice/AE/config.json'
    config = Config(config_path)
    dic = config.dict
    print(config.data_path)
    print(config.spm_txt_path)
    