import numpy as np
import copy


class Detection:
    """ Object Detection Data Class

    This Class contains an object detection report that was obtained by a
    sensor for a single object.

    Attributes
    ----------
    time : float
        Detection time [sec]
    measurement : array_like
        Object measurement [m]
    measurement_noise : array_like
        Measurement noise covariance [m^2]
    sensor_index : int
        Sensor identifier
    object_id : int
        Object class identifier
    object_attribute : str or float
        Object attributes
    """

    def __init__(self, time : float, measurement : list):

        # conversion to row vector
        n_dim_state = len(measurement)
        vector = np.array(measurement).reshape(n_dim_state, 1)

        self.time : float = time
        self.measurement : np.ndarray = vector
        self.measurement_noise : np.ndarray = np.empty(0)
        self.sensor_index : int = []
        self.object_id : int = []
        self.object_attribute = []


class Filter:
    """ Kalman Filter Class

    This class is a discrete-time linear Kalman Filter used to track the
    position and velocities of target object.

    Attributes
    ----------
    state : array_like
        Kalman filter state [m][m/s]
    state_covariance : array_like
        State estimation error covariance
    state_transition_model : array_like
        Matrix of state transition
    process_noise : array_like
        Covariance of process noise
    measurement_model : array_like
        Measurement Matrix for state vector
    """

    def __init__(self):
        self.state : np.ndarray = np.empty(0)
        self.state_covariance : np.ndarray = np.empty(0)
        self.state_transition_model : np.ndarray = np.empty(0)
        self.process_noise : np.ndarray = np.empty(0)
        self.measurement_model : np.ndarray = np.empty(0)

    def initialize(self, detection_old : Detection, detection_new : Detection):
        """ Initialize Kalman filter

        Initialize state and state estimation error covariance of linear Kalman
        filter (using constant velocity model).


        Parameters
        ----------
        detection_old : Detection class
            Older detection for initiation
        detection_new : Detection class
            Newer detection for initiation

        Returns
        -------
        True


        """

        z_new = detection_new.measurement
        z_old = detection_old.measurement
        dt = detection_new.time - detection_old.time

        pos = z_new
        vel = (z_new - z_old) / dt
        self.state = np.vstack([pos, vel])

        R_new = detection_new.measurement_noise
        R_old = detection_old.measurement_noise
        P_11 = R_new
        P_12 = R_new / dt
        P_22 = (R_new + R_old) / (dt**2)
        self.state_covariance = np.block([[P_11, P_12], [P_12, P_22]])

        return True

    def predict(self):
        """ Prediction step

        Predict state and state estimation error covariance of linear Kalman
        filter.

        Parameters
        ----------

        Returns
        -------
        True

        """

        x = self.state
        P = self.state_covariance
        phi = self.state_transition_model
        Q = self.process_noise

        self.state = phi @ x
        self.state_covariance = phi @ P @ phi.T + Q

        return True


    def correct(self, detection):
        """ Correction step

        Correct state and state estimation error covariance of linear Kalman
        filter.

        Parameters
        ----------
        detection : Detection class
            Detection for correction

        Returns
        -------
        True

        """

        x = self.state
        P = self.state_covariance
        H = self.measurement_model
        z = detection.measurement
        R = detection.measurement_noise

        S = H @ P @ H.T + R # residual covariance
        inv_S = np.linalg.inv(S)
        K = P @ H.T @ inv_S # Kalman gain

        residual = z - (H @ x)
        self.state = x + (K @ residual)
        self.state_covariance = P - (K @ (H @ P))

        return True





