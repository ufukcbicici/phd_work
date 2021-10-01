import time


# Simple profiler
class Profiler:

    def __init__(self):
        self.measurements = []
        self.measurementLabels = []
        self.measurements.append(time.time())

    def add_measurement(self, label):
        time_stamp = time.time()
        self.measurements.append(time_stamp)
        self.measurementLabels.append(label)

    def get_all_measurements(self):
        measurements_dict = {}
        assert len(self.measurements) > 1
        for idx in range(1, len(self.measurements)):
            interval = self.measurements[idx] - self.measurements[idx - 1]
            label = self.measurementLabels[idx - 1]
            measurements_dict[label] = interval
        return measurements_dict
