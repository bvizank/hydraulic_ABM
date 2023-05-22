class Test():
    def __init__(self, num):
        self.data_container = list()
        for i in range(10):
            self.data_container.append(0)

    def save_data(self, data, container):
        container.append(data)

    def collect_data(self, data):
        self.save_data(data, self.data_container)
