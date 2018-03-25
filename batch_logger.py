class BatchLogger(object):

    def __init__(self):
        super(BatchLogger, self).__init__()
        self.reset()

    def reset():
        self.losses, self.batch_count = [], 0
        self.interval_correct, self.interval_count = 0, 0
        self.all_correct, self.all_count = 0, 0
        self.t0 = self.t1 = time.time()

    def start():
        self.reset()
