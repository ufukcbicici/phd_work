import threading


class ModelRunner(threading.Thread):
    def __init__(self, threadId, name, device, list_of_runIds, train_program_path, list_of_params, file_paths):
        threading.Thread.__init__(self)
        self.threadID = threadId
        self.name = name
        self.device = device
        self.listOfRunIds = list_of_runIds
        self.trainProgramPath = train_program_path
        self.listOfParams = list_of_params
        self.filePaths = file_paths