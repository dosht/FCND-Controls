"""
PID Controller

components:
    follow attitude commands
    gps commands and yaw
    waypoint following
"""
import numpy as np
from frame_utils import euler2RM

DRONE_MASS_KG = 0.5
GRAVITY = -9.81
MOI = np.array([0.005, 0.005, 0.01])
MAX_THRUST = 10.0
MAX_TORQUE = 1.0
MAX_YAW = 2 * np.pi

class NonlinearController(object):

    def __init__(self,
                 g=GRAVITY,
                 m=DRONE_MASS_KG):
        """Initialize the controller object and control gains"""
        self.g = g
        self.m = m

    def trajectory_control(self, position_trajectory, yaw_trajectory, time_trajectory, current_time):
        """Generate a commanded position, velocity and yaw based on the trajectory

        Args:
            position_trajectory: list of 3-element numpy arrays, NED positions
            yaw_trajectory: list yaw commands in radians
            time_trajectory: list of times (in seconds) that correspond to the position and yaw commands
            current_time: float corresponding to the current time in seconds

        Returns: tuple (commanded position, commanded velocity, commanded yaw)

        """

        ind_min = np.argmin(np.abs(np.array(time_trajectory) - current_time))
        time_ref = time_trajectory[ind_min]


        if current_time < time_ref:
            position0 = position_trajectory[ind_min - 1]
            position1 = position_trajectory[ind_min]

            time0 = time_trajectory[ind_min - 1]
            time1 = time_trajectory[ind_min]
            yaw_cmd = yaw_trajectory[ind_min - 1]

        else:
            yaw_cmd = yaw_trajectory[ind_min]
            if ind_min >= len(position_trajectory) - 1:
                position0 = position_trajectory[ind_min]
                position1 = position_trajectory[ind_min]

                time0 = 0.0
                time1 = 1.0
            else:

                position0 = position_trajectory[ind_min]
                position1 = position_trajectory[ind_min + 1]
                time0 = time_trajectory[ind_min]
                time1 = time_trajectory[ind_min + 1]

        position_cmd = (position1 - position0) * \
                        (current_time - time0) / (time1 - time0) + position0
        velocity_cmd = (position1 - position0) / (time1 - time0)


        return (position_cmd, velocity_cmd, yaw_cmd)

    def lateral_position_control(self, local_position_cmd, local_velocity_cmd, local_position, local_velocity,
                               acceleration_ff = np.array([0.0, 0.0])):
        """Generate horizontal acceleration commands for the vehicle in the local frame

        Args:
            local_position_cmd: desired 2D position in local frame [north, east]
            local_velocity_cmd: desired 2D velocity in local frame [north_velocity, east_velocity]
            local_position: vehicle position in the local frame [north, east]
            local_velocity: vehicle velocity in the local frame [north_velocity, east_velocity]
            acceleration_cmd: feedforward acceleration command

        Returns: desired vehicle 2D acceleration in the local frame [north, east]
        """
        #TODO move to __init__
        k_p = np.array([4.5, 4.5])
        k_d = np.array([3.5, 3.5])

        position_err = local_position_cmd - local_position
        velocity_err = local_velocity_cmd - local_velocity
        p_term = k_p * position_err
        d_term = k_d * velocity_err

        acc_cmd = p_term + d_term + acceleration_ff
        # print(-acc_cmd)
        # return np.array([1.0, 1.0])
        return acc_cmd

    def roll_pitch_controller(self, acceleration_cmd, attitude, thrust_cmd):
        """ Generate the rollrate and pitchrate commands in the body frame

        Args:
            acceleration_cmd: 2-element numpy array (north_acceleration_cmd,east_acceleration_cmd) in m/s^2
            attitude: 3-element numpy array (roll, pitch, yaw) in radians
            thrust_cmd: vehicle thruts command in Newton

        Returns: 2-element numpy array, desired rollrate (p) and pitchrate (q) commands in radians/s
        """
        # #TODO move to __init__
        k_p_roll = 8
        k_p_pitch = 5

        (b_x_c, b_y_c) = acceleration_cmd * self.m / -thrust_cmd

        R = euler2RM(*attitude)

        b_x = R[0, 2]
        b_x_err = b_x_c - b_x
        b_x_c_dot = k_p_roll * b_x_err

        b_y = R[1, 2]
        b_y_err = b_y_c - b_y
        b_y_c_dot = k_p_pitch * b_y_err

        rot_mat1 = np.array([[R[1, 0], -R[0, 0]],
                             [R[1, 1], -R[0, 1]]]) / R[2, 2]

        rot_rate = np.matmul(rot_mat1,
                             np.array([b_x_c_dot, b_y_c_dot]).T)

        print(rot_rate)
        return rot_rate

    def altitude_control(self, z_c, z_dot_c, z, z_dot, attitude,
                         z_dot_dot_c=0.0):
        """Generate vertical acceleration (thrust) command

        Args:
            altitude_cmd: desired vertical position (+up)
            vertical_velocity_cmd: desired vertical velocity (+up)
            altitude: vehicle vertical position (+up)
            vertical_velocity: vehicle vertical velocity (+up)
            attitude: the vehicle's current attitude, 3 element numpy array (roll, pitch, yaw) in radians
            acceleration_ff: feedforward acceleration command (+up)

        Returns: thrust command for the vehicle (+up)
        """
        # z_c = 3
        # z_dot_c = 0

        # TODO move to __init__
        # TODO should be calculated?
        z_k_p = 8
        z_k_d = 2.3

        z_err = z_c - z
        z_err_dot = z_dot_c - z_dot

        p_term = z_k_p * z_err
        d_term = z_k_d * z_err_dot

        R = euler2RM(*attitude)
        b_z = R[2, 2]

        u_1_bar = p_term + d_term + z_dot_dot_c
        # thrust = (u_1_bar - self.g) / b_z
        thrust = u_1_bar * self.m / b_z
        return bounded_thrust(thrust)

        # return 5.5

    def body_rate_control(self, body_rate_cmd, body_rate):
        """ Generate the roll, pitch, yaw moment commands in the body frame

        Args:
            body_rate_cmd: 3-element numpy array (p_cmd,q_cmd,r_cmd) in radians/second^2
            body_rate: 3-element numpy array (p,q,r) in radians/second^2

        Returns: 3-element numpy array, desired roll moment, pitch moment, and yaw moment commands in Newtons*meters
        """
        # (p_c, q_c, r_c) = np.array([0.0, 0.0, 0.0])
        (p_c, q_c, r_c) = body_rate_cmd
        # body_rate_cmd = np.array([0.0, 0.0, 0.0])

        #TODO move to __init__
        k_p_p = 20
        k_p_q = 20
        k_p_r = 5

        (p, q, r) = body_rate
        p_err = p_c - p
        q_err = q_c - q
        r_err = r_c - r

        u_bar_p = k_p_p * p_err
        u_bar_q = k_p_q * q_err
        u_bar_r = k_p_r * r_err

        moment_c = np.array([u_bar_p, u_bar_q, u_bar_r]) * MOI

        # return np.array([0.0, 0.0, 0.0])
        return bounded_moment(moment_c)

    def yaw_control(self, yaw_cmd, yaw):
        """ Generate the target yawrate

        Args:
            yaw_cmd: desired vehicle yaw in radians
            yaw: vehicle yaw in radians

        Returns: target yawrate in radians/sec
        """

        k_p_yaw = 5.0
        yaw_cmd = np.mod(yaw_cmd, MAX_YAW)

        yaw_err = yaw_cmd - yaw
        r_c = k_p_yaw * bounded_yaw(yaw_err)
        # print(f"yaw: {yaw}, yaw_cmd: {yaw_cmd} yaw_err: {yaw_err}, r_c: {r_c}")
        # return 10
        return r_c


def bounded_thrust(value):
    print(f"thrust value: {value}")
    return np.clip(value, 0.001, MAX_THRUST)


def bounded_moment(value):
    return np.clip(value, -MAX_TORQUE, MAX_TORQUE)


def bounded_yaw(value):
    if -np.pi < value < np.pi:
        return value

    correction = MAX_YAW * (-1 if value > 0 else 1)
    return value + correction
