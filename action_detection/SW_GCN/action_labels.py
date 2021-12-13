import warnings

action_labels_whole = {
    "Assemble system": 11,
    "Consult sheets": 1,
    "No action": 0,
    "Picking in front": 5,
    "Picking left": 6,
    "Put down component": 10,
    "Put down measuring rod": 8,
    "Put down screwdriver": 4,
    "Put down subsystem": 13,
    "Take component": 9,
    "Take measuring rod": 7,
    "Take screwdriver": 3,
    "Take subsystem": 12,
    "Turn sheets": 2
}
#
# action_names = [
#     "Assemble system",
#     "Consult sheets",
#     "No action",
#     "Picking in front",
#     "Picking left",
#     "Put down component",
#     "Put down measuring rod",
#     "Put down screwdriver",
#     "Put down subsystem",
#     "Take component",
#     "Take measuring rod",
#     "Take screwdriver",
#     "Take subsystem",
#     "Turn sheets"]

action_labels_train = {
    "Assemble system": 0,
    "Picking in front": 1,
    "Picking left": 2,
    "Put down component": 3,
    "Put down subsystem": 4,
    "Take component": 5,
    "Take subsystem": 6,
}
action_names_train = [
    "Assemble system",
    "Picking in front",
    "Picking left",
    "Put down component",
    "Put down subsystem",
    "Take component",
    "Take subsystem",
]

# action_labels = {
#     "Assemble system": 0,
#     "Picking in front": 1,
#     "Picking left": 2,
#     "Put down component": 3,
#     "Put down subsystem": 4,
#     "Take component": 5,
#     "Take subsystem": 6,
# }
# action_names = [
#     "Assemble system",
#     "Picking in front",
#     "Picking left",
#     "Put down component",
#     "Put down subsystem",
#     "Take component",
#     "Take subsystem",
# ]

action_labels = {
    "Assemble system": 0,
    "Put down": 1,
    "Take": 2
}
action_names = [
    "Assemble system",
    "Put down",
    "Take"
]


def get_key_by_value(value):
    for string in action_labels:
        if action_labels[string] == value:
            return string

    warnings.warn("Requested action class not listed!")
    return "No action class"


def get_action_class_from_action_labels_train(action_label):
    """
    parse data
    summarize action classes
    :param action_label:
    :return:
    """
    if action_labels_train[action_label] in [1, 2, 5, 6]:
        return 2
    elif action_labels_train[action_label] in [3, 4]:
        return 1
    else:
        return 0


if __name__ == '__main__':
    for i in action_labels_train:
        print(get_action_class_from_action_labels_train(i))
