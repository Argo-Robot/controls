from scripts.kinematics import *
from scripts.dynamics import *



## kynematics ##

print("\n\nKINEMATICS EXAMPLE:\n\n")

# init
robot = Robot(robot_type="so100")
kin = RobotKinematics()

# get current joint positions
q_init = np.array([-np.pi / 2, -np.pi / 2, np.pi / 2, np.pi / 2, -np.pi / 2, np.pi / 2])
print("q_init_mechanical: ", np.rad2deg(q_init))

# convert from mechanical angle to dh angle
q_init_dh = robot.from_mech_to_dh(q_init)
print("\nq_init_dh_deg: ", np.rad2deg(q_init_dh))
print("q_init_dh_rad: ", q_init_dh)

# compute start pose
T_start = kin.forward_kinematics(robot, q_init_dh)
print("\nT_start = \n", T_start)

# Define goal pose
T_goal = T_start.copy()
T_goal[:3, 3] += np.array([0.0, 0.0, -0.1])
print("\nT_goal = \n", T_goal)

# IK with internal interpolation
q_final_dh = kin.inverse_kinematics(robot, q_init_dh, T_goal, use_orientation=True, k=0.8, n_iter=50)
T_final = kin.forward_kinematics(robot, q_final_dh)

print("\nFinal joint angles = ", q_final_dh)
print("\nFinal pose direct kinematics = \n", T_final)

print("\nerr_lin = ", RobotUtils.calc_lin_err(T_goal, T_final))
print("err_ang = ", RobotUtils.calc_ang_err(T_goal, T_final))

# convert from dh angle to mechanical angle
q_final_mech = robot.from_dh_to_mech(q_final_dh)
print("\nq_final_mech: ", np.rad2deg(q_final_mech))

# add gripper position
gripper_pose = np.deg2rad(0.0)
q_final_mech = np.append(q_final_mech, gripper_pose)

# raise an error in case joint limits are exceeded
robot.check_joint_limits(q_final_mech)


## dynamics ## 

print("\n\nDYNAMICS EXAMPLE:\n\n")

# init
dyn = RobotDynamics()

# init parameters
q = np.array([0.1, 0.2, 0.1, 0.2, 0.1]) 
qdot = np.array([0.7, 0.7, 0.7, 0.7, 0.7]) # --> -0.6120  -7.6320  -3.2212   0.4416   0.1600 e qddot = 0
qddot = np.array([0.7, 0.7, 0.7, 0.7, 0.7]) # --> -0.1444  -7.3537  -3.0630   0.4483   0.3428 e qdot != 0
Fext = np.array([6.0, 5.0, 4.0, 3.0, 2.0, 1.0]) # expressed wrt n-frame --> 3.4263  -8.6711  -5.0112  -2.1744   1.3428 e qdot, qddot != 0

# inverse dynamics
torques = dyn.inverse_dynamics(robot, q, qdot, qddot, Fext = Fext) 
print("torques with RNE: \n", torques)

# compute B, Cqdot, G for the whole robotic model
B, Cqdot, G = dyn.get_robot_model(robot, q, qdot)
print("\nB: \n", B)
print("\nCqdot: \n", Cqdot)
print("\nG: \n", G)

# transform force
f_ext_tool = np.array([0.0, 0.0, 2.0, 0.0, 1.0, 0.0])
tool_T_n = RobotUtils.inv_homog_mat(robot.nTtool)
f_ext_n = dyn.transform_force(f_ext_tool, tool_T_n)
print("\nforce expressed in n-frame: \n", f_ext_n)

# jacobians
q = np.array([0.1, 0.2, 0.1, 0.2, 0.1]) 
base_T_n = kin._forward_kinematics_baseTn(robot, q)
J0 = RobotUtils.calc_geom_jac_0(robot, q)
Jn = RobotUtils.calc_geom_jac_n(robot, q, base_T_n)
print("\ngeometric jacobian in base-frame: \n", J0)
print("\ngeometric jacobian in n-frame: \n", Jn)
