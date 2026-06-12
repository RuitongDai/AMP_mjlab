"""Unitree G1 constants."""

from pathlib import Path

import mujoco

from src import SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
# from src.assets.robots.moya.x3_actuators import *
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import (
  ElectricActuator,
  reflected_inertia_from_two_stage_planetary,
)
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

X3_XML: Path = (
  SRC_PATH / "assets" / "robots" / "moya" / "xmls" / "Moya01_V2.xml"
)
assert X3_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, X3_XML.parent / "assets", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(X3_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec



X3_ACTUATOR_LEGS = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_hip_yaw_joint",
    ".*_hip_roll_joint",
    ".*_knee_joint"
  ),
  stiffness=100.0,
  damping=2.0,
  effort_limit=75.0,
  armature=0.01
)

X3_ACTUATOR_LEGS_HIPS_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_pitch_joint",),
  stiffness=100.0,
  damping=2.0,
  effort_limit=75.0,
  armature=0.01
)

X3_ACTUATOR_FEET = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_ankle_pitch_joint",".*_ankle_roll_joint"),
  stiffness=30.0,
  damping=2.0,
  effort_limit=75.0,
  armature=0.01
)

X3_ACTUATOR_WAIST = BuiltinPositionActuatorCfg(
  target_names_expr=("waist_roll_joint","waist_yaw_joint"),
  stiffness=100.0,
  damping=2.0,
  effort_limit=50.0,
  armature=0.01
)

X3_ACTUATOR_SHOULDER = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_shoulder_pitch_joint",
    ".*_shoulder_roll_joint",
    ".*_shoulder_yaw_joint",
                ),
  stiffness=30.0,
  damping=2.0,
  effort_limit=25.0,
  armature=0.008
)

X3_ACTUATOR_FORE_ARM = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_elbow_joint",".*_wrist_roll_joint"),
  stiffness=30.0,
  damping=2.0,
  effort_limit=25.0,
  armature=0.005
)

X3_ACTUATOR_HAND = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_wrist_pitch_joint",".*_wrist_yaw_joint",),
  stiffness=20.0,
  damping=2.0,
  effort_limit=5.0,
  armature=0.005
)

HOME_KEY_FRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.80),
  joint_pos={
    ".*_hip_pitch_joint": -0.1,
    ".*_knee_joint": 0.2,
    ".*_ankle_pitch_joint": -0.1,
    ".*_elbow_joint": 0.0,
    "left_shoulder_roll_joint": 0.0,
    "left_shoulder_pitch_joint": 0.0,
    "right_shoulder_roll_joint": -0.0,
    "right_shoulder_pitch_joint": 0.0,
  },
  joint_vel={".*": 0.0},
)

X3_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    X3_ACTUATOR_LEGS,
    X3_ACTUATOR_LEGS_HIPS_PITCH,
    X3_ACTUATOR_FEET,
    X3_ACTUATOR_WAIST,
    X3_ACTUATOR_SHOULDER,
    X3_ACTUATOR_FORE_ARM,
    X3_ACTUATOR_HAND
  ),
  soft_joint_pos_limit_factor=0.9
)


FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={r"^(left|right)_foot[1-7]_collision$": 3, ".*_collision": 1},
  priority={r"^(left|right)_foot[1-7]_collision$": 1},
  friction={r"^(left|right)_foot[1-7]_collision$": (0.6,)},
)

FULL_COLLISION_WITHOUT_SELF = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype=0,
  conaffinity=1,
  condim={r"^(left|right)_foot[1-7]_collision$": 3, ".*_collision": 1},
  priority={r"^(left|right)_foot[1-7]_collision$": 1},
  friction={r"^(left|right)_foot[1-7]_collision$": (0.6,)},
)

# This disables all collisions except the feet.
# Feet get condim=3, all other geoms are disabled.
FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(r"^(left|right)_foot[1-7]_collision$",),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
)

def get_x3_robot_cfg() -> EntityCfg:

  return EntityCfg(
    init_state=HOME_KEY_FRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=X3_ARTICULATION
  )


X3_ACTION_SCALE : dict[str,float] = {}
for a in X3_ARTICULATION.actuators:
  assert isinstance(a,BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  names = a.target_names_expr
  assert e is not None
  for n in names:
    X3_ACTION_SCALE[n] = 0.25 * e / s

if __name__ == "__main__":
  import mujoco.viewer as viewer
  
  from mjlab.entity.entity import Entity

  robot = Entity(get_x3_robot_cfg())

  viewer.launch(robot.spec.compile())