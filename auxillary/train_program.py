import json


class TrainProgram:
    def __init__(self, program_file):
        self.programFile = program_file

    @staticmethod
    def decode_json_element_for_parameter(parameter_name, json_element):
        for k, v in json_element.items():
            # Tokenize dictionary key. Look if the parameter name matches the key: All "," separated words must be
            # contained in the parameter name.
            tokens = k.split(",")
            all_tokens_inside = True
            for token in tokens:
                if not (token in parameter_name):
                    all_tokens_inside = False
                    break
            if all_tokens_inside:
                return v
        raise Exception("No suitable setting for parameter {0}".format(parameter_name))

    def set_train_program_element(self, element_name, keywords, skipwords, value):
        json_file = open(self.programFile)
        config_file = json_file.read()
        connection_map = json.loads(config_file)
        json_file.close()
        for entry in connection_map[element_name]:
            all_tokens_inside = True
            for token in keywords:
                if not (token in entry):
                    all_tokens_inside = False
                    break
            if not all_tokens_inside:
                continue
            contains_skip_words = False
            for token in skipwords:
                if token in entry:
                    contains_skip_words = True
                    break
            if contains_skip_words:
                continue
            connection_map[element_name][entry] = value
        json_file = open(self.programFile, "w")
        json.dump(connection_map, json_file, indent=2)
        json_file.close()

    def load_settings_for_property(self, property_name):
        with open(self.programFile) as f:
            config_file = f.read()
            connection_map = json.loads(config_file)
            if property_name not in connection_map:
                raise Exception("Train program does not contain information about property {0}".format(property_name))
        return connection_map[property_name]
