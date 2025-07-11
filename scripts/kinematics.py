import copy
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from scripts.utils import *


class RobotKinematics:
    
    def __init__(self):
        pass

    def forward_kinematics(self, robot, q):
        
        """compute forward kinematics (worldTtool)"""

        baseTn = self._forward_kinematics_baseTn(robot, q)

        return robot.worldTbase @ baseTn @ robot.nTtool

    def _forward_kinematics_baseTn(self, robot, q):
        
        """compute forward kinematics (baseTn)"""

        T = np.eye(4)
        DOF = len(q)

        for i in range(DOF):
            dh = robot.dh_table[i]
            T_link = RobotUtils.calc_dh_matrix(dh, q[i])
            T = T @ T_link

        return T

    def _inverse_kinematics_step_baseTn(self, robot, q_start, T_desired, use_orientation=True, k=0.8, n_iter=50):
        
        """compute inverse kinematics (T_desired must be expressed in baseTn)"""

        # don't override current joint positions
        q = copy.deepcopy(q_start)

        for _ in range(n_iter):
            # compute current pose baseTn
            T_current = self._forward_kinematics_baseTn(robot, q)

            # compute linear error
            err_lin = RobotUtils.calc_lin_err(T_current, T_desired)

            # decide whether to use full jacobian or not
            if use_orientation:
                err_ang = RobotUtils.calc_ang_err(T_current, T_desired)  # compute angular error
                error = np.concatenate((err_lin, err_ang))  # total error
                J_geom = RobotUtils.calc_geom_jac_0(robot, q)  # full jacobian
            else:
                error = err_lin  # total error
                J_geom = RobotUtils.calc_geom_jac_0(robot, q)[:3, :]  # take only the position part

            # stop if error is minimum
            if np.linalg.norm(error) < 1e-5:
                break

            # Damped Least Squares Right-Pseudo-Inverse
            J_pinv = RobotUtils.dls_right_pseudoinv(J_geom)

            # keep integrating resulting joint positions
            q += k * (J_pinv @ error)

        return q

    def inverse_kinematics(self, robot, q_start, desired_worldTtool, use_orientation=True, k=0.8, n_iter=50):
        
        """compute inverse kinematics (T_desired must be expressed in worldTtool)
        It is performed an interpolation both for linear and angular components"""

        # I compute ikine with baseTn
        desired_baseTn = (RobotUtils.inv_homog_mat(robot.worldTbase)
                          @ desired_worldTtool
                          @ RobotUtils.inv_homog_mat(robot.nTtool))

        # don't override current joint positions
        q = copy.deepcopy(q_start)

        # init interpolator
        n_steps = self._interp_init(self._forward_kinematics_baseTn(robot, q), desired_baseTn)

        for i in range(0, n_steps + 1):
            # current setpoint as baseTn
            T_desired_interp = self._interp_execute(i)

            # get updated joint positions
            q = self._inverse_kinematics_step_baseTn(robot, q, T_desired_interp, use_orientation, k, n_iter)

        # check final error
        current_worldTtool = self.forward_kinematics(robot, q)
        err_lin = RobotUtils.calc_lin_err(current_worldTtool, desired_worldTtool)
        lin_error_norm = np.linalg.norm(err_lin)
        assert lin_error_norm < 1e-2, (f"[ERROR] Large position error ({lin_error_norm:.4f}). Check target reachability (position/orientation)")

        return q

    def _interp_init(self, T_start, T_final):
        
        """Initialize interpolator parameters"""

        # init
        self.t_start = T_start[:3, 3]
        self.t_final = T_final[:3, 3]
        R_start = T_start[:3, :3]
        R_final = T_final[:3, :3]
        
        # step size for trajectory
        delta_trans = 0.01  # meters
        delta_rot = 0.05    # radians
        
        # linear distance
        trans_dist = RobotUtils.calc_distance(self.t_final, self.t_start)
        n_steps_trans = trans_dist / delta_trans
        
        # angular distance
        rotvec = R.from_matrix(R_final @ R_start.T).as_rotvec() # axis*angle
        ang_dist = np.linalg.norm(rotvec) # angle modulus
        n_steps_rot = ang_dist / delta_rot
        
        # total steps
        self.n_steps = int(np.ceil(max(n_steps_trans, n_steps_rot)))

        # Create SLERP object
        times = [0, 1]
        rotations = R.from_matrix([R_start, R_final])
        self.slerp = Slerp(times, rotations)

        return self.n_steps

    def _interp_execute(self, i):
        
        """Compute Cartesian pose setpoint for the current step"""

        # n_steps == 0 means Tgoal == Tinit
        # In this way I also avoid division by zero
        if self.n_steps == 0:
            s = 1.0
        else:
            s = i / self.n_steps  # compute current step

        t_interp = (1 - s) * self.t_start + s * self.t_final
        R_interp = self.slerp(s).as_matrix()

        # compute current setpoint
        T_interp = np.eye(4)
        T_interp[:3, :3] = R_interp
        T_interp[:3, 3] = t_interp

        return T_interp
    
    
