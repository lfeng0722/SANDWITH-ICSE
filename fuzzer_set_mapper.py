import numpy as np

def map_to_vehicle_coordinates(theta, movement_vector):
    R = np.array([
        [np.cos(theta), np.sin(theta)],
        [-np.sin(theta), np.cos(theta)]
    ])

    movement = np.array(movement_vector)

    vehicle_movement = R.dot(movement)

    dx_vehicle, dy_vehicle = vehicle_movement
    return dx_vehicle, dy_vehicle

def FSM_mapper(output_action):
    # x, y = map_to_vehicle_coordinates(heading, output_action)

    if output_action[1] != 3:
        return 'break'
    elif output_action[0] == 0 :
        return 'right_change_acc'
    elif output_action[0] == 1:
        return 'accelerate'
    elif output_action[0] == 2:
        return 'left_change_acc'
