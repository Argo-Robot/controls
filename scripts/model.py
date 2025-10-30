from urchin import URDF
import numpy as np



class RobotLoader:

    """Parent class to define methods/attributes common to DH and URDF representations"""

    def __init__(self):

        self.link_mass = []
        self.link_com = []
        self.link_inertia = []
        self.mech_joint_limits_low = []
        self.mech_joint_limits_up = []
        self.worldTbase = None
        self.nTtool = None

    def get_n_joints(self):
        raise NotImplementedError

    def get_n_links(self):
        raise NotImplementedError
    
    def print_model_properties(self):
        raise NotImplementedError

    def convert_dh_to_mech(self, q_dh):
        raise NotImplementedError

    def convert_mech_to_dh(self, q_mech):
        raise NotImplementedError



class DH_loader(RobotLoader):
    
    """Child class: define manually robot model using DH convention (here SO100)"""
    
    def __init__(self):
        
        super().__init__()  # Initialize RobotLoader

        # Follow this convention: theta , d, a, alpha
        self.ROBOT_DH_TABLES = [
                [0, 0.0542, 0.0304, np.pi / 2],
                [0, 0.0, 0.116, 0.0],
                [0, 0.0, 0.1347, 0.0],
                [0, 0.0, 0.0, -np.pi / 2],
                [0, 0.0609, 0.0, 0.0],  # to increase length and include also gripper: [0, 0.155, 0.0, 0.0],
            ]
        
        # define mass for link i (n° links = n° joints +1)
        self.LINK_MASS = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]) # --> fake values
        
        # define COM for link i wrt origin frame i (n° links = n° joints +1)
        self.LINK_COM = [ 
                np.array([0.0, 0.05, 0.0]), # --> fake values
                np.array([0.0, 0.05, 0.0]), # --> fake values
                np.array([0.0, 0.05, 0.0]), # --> fake values
                np.array([0.0, 0.05, 0.0]), # --> fake values
                np.array([0.0, 0.05, 0.0]), # --> fake values
                np.array([0.0, 0.05, 0.0]), # --> fake values
            ]
        
        # define Inertia for link i wrt origin frame i (n° links = n° joints +1)
        self.LINK_INERTIA = [
                np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), # --> fake values
                np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), # --> fake values
                np.array([0.0, 0.0, 0.0, 0.0, 0.0542, 0.0, 0.0, 0.0, 0.0304]), # --> fake values
                np.array([0.0, 0.0, 0.0, 0.0, 0.0542, 0.0, 0.0, 0.0, 0.0304]), # --> fake values
                np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1347]), # --> fake values
                np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1347]), # --> fake values
            ]
    
        # mechanical joint limitis
        self.MECH_JOINT_LIMITS_LOW = np.array([-2.2000, -3.1416, 0.0000, -2.0000, -3.1416, -0.2000])
        self.MECH_JOINT_LIMITS_UP = np.array([2.2000, 0.2000, 3.1416, 1.8000, 3.1416, 2.0000])
    
        # set worldTbase frame (base-frame DH aligned wrt SO100 simulator)
        self.WORLD_T_TOOL = np.array([[0.0, 1.0, 0.0, 0.0], 
                                     [-1.0, 0.0, 0.0, -0.0453], 
                                     [0.0, 0.0, 1.0, 0.0647], 
                                     [0.0, 0.0, 0.0, 1.0]])
    
        # set nTtool frame (n-frame DH aligned wrt SO100 simulator)
        self.N_T_TOOL = np.array([[0.0, 0.0, -1.0, 0.0], 
                                  [1.0, 0.0, 0.0, 0.0], 
                                  [0.0, -1.0, 0.0, 0.0], 
                                  [0.0, 0.0, 0.0, 1.0]])

    def get_n_joints(self):
        return len(self.ROBOT_DH_TABLES)

    def get_n_links(self):
        return len(self.ROBOT_DH_TABLES) + 1
    
    def print_model_properties(self):
        
        print("Link names:")
        for n in range(self.get_n_links()):
            print(f" - Link {n}")

        print("\nJoint names:")
        for n in range(self.get_n_joints()):
            print(f" - Joint {n+1}")
        
    def convert_dh_to_mech(self, q_dh):
        
        """convert joint positions from DH to mechanical coordinates (for SO100)"""

        beta = np.deg2rad(14.45)  # make reference to dh2.png in README.md

        q_mech = np.zeros_like(q_dh)
        q_mech[0] = q_dh[0]
        q_mech[1] = -q_dh[1] - beta
        q_mech[2] = -q_dh[2] + beta
        q_mech[3] = -q_dh[3] - np.pi / 2
        q_mech[4] = -q_dh[4] - np.pi / 2

        return q_mech

    def convert_mech_to_dh(self, q_mech):
        
        """convert joint positions from mechanical to DH coordinates (for SO100)"""

        beta = np.deg2rad(14.45)  # make reference to dh2.png in README.md

        q_dh = np.zeros_like(q_mech)
        q_dh[0] = q_mech[0]
        q_dh[1] = -q_mech[1] - beta
        q_dh[2] = -q_mech[2] + beta
        q_dh[3] = -q_mech[3] - np.pi / 2
        q_dh[4] = -q_mech[4] - np.pi / 2

        return q_dh[:-1]  # skip last DOF because it is the gripper
    
    
    
class URDF_loader(RobotLoader):
    
    """Child class: load robot model using URDF file"""
    
    def __init__(self):

        super().__init__()  # Initialize RobotLoader

    def load(self, path):
        
        # load URDF
        self.robot = URDF.load(path)

        # init
        self.LINK_MASS = []
        self.LINK_COM = []
        self.LINK_INERTIA = []
        
        # add link mass properties
        for link in self.robot.links:
            inertial = link.inertial
            if inertial is not None:
                self.LINK_MASS.append(inertial.mass)
                self.LINK_COM.append(inertial.origin[:3, 3])  
                self.LINK_INERTIA.append(inertial.inertia.flatten())
            else:
                self.LINK_MASS.append(0.0)
                self.LINK_COM.append(np.zeros(3))
                self.LINK_INERTIA.append(np.zeros(9))

        # add joint limits
        self.MECH_JOINT_LIMITS_LOW = np.array([j.limit.lower for j in self.robot.joints if j.joint_type != 'fixed'])
        self.MECH_JOINT_LIMITS_UP = np.array([j.limit.upper for j in self.robot.joints if j.joint_type != 'fixed'])

        # add world and tool transforms
        self.WORLD_T_TOOL = np.eye(4)
        self.N_T_TOOL = np.eye(4)

        # add root link name
        self.ROOT_LINK_NAME = self.find_root_link()
        
    def get_n_joints(self):
        return len([j for j in self.robot.joints if j.joint_type != 'fixed'])
    
    def get_n_links(self):
        return len(self.robot.links)
    
    def print_model_properties(self): 
        
        print("Link names:")
        for link in self.robot.links:
            print(f" - {link.name}")

        print("\nJoint names:")
        for joint in self.robot.joints:
            print(f" - {joint.name} ({joint.joint_type})")

        print("\nRoot link name:", self.ROOT_LINK_NAME)

    def find_root_link(self):

        """Find root link name"""

        # find root
        all_links = {link.name for link in self.robot.links}
        child_links = {joint.child for joint in self.robot.joints}
        roots = list(all_links - child_links) # it shall remain just 1 element (root) in the set
        
        if len(roots) != 1:
            raise ValueError(f"Expected 1 root, found {roots}")
        
        return roots[0]



class RobotModel:

    """Wrapper: contains info related to the robot model (DH or URDF)"""
    
    def __init__(self, loader):
        
        # set robot model
        self.loader = loader
        self.link_mass = loader.LINK_MASS
        self.link_com = loader.LINK_COM
        self.link_inertia = loader.LINK_INERTIA
        self.mech_joint_limits_low = loader.MECH_JOINT_LIMITS_LOW
        self.mech_joint_limits_up = loader.MECH_JOINT_LIMITS_UP
        self.worldTbase = loader.WORLD_T_TOOL
        self.nTtool = loader.N_T_TOOL

        # load variables common only to DH or URDF
        self.dh_table = getattr(loader, 'ROBOT_DH_TABLES', None)
        self.root_link = getattr(loader, 'ROOT_LINK_NAME', None)

    def get_n_joints(self):
        return self.loader.get_n_joints()

    def get_n_links(self):
        return self.loader.get_n_links()
    
    def print_model_properties(self):
        return self.loader.print_model_properties()

    def convert_dh_to_mech(self, q_dh):
        return self.loader.convert_dh_to_mech(q_dh)

    def convert_mech_to_dh(self, q_mech):
        return self.loader.convert_mech_to_dh(q_mech)