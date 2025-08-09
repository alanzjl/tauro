import copy
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
from mujoco import viewer

SRC = Path("tauro_common/models/so_arm100/so_arm100.xml")
OUT = Path("tauro_common/models/so_arm100/scene_two_arms.xml")

# --- 1) Parse your original MJCF ---
root = ET.parse(SRC).getroot()


# helpers
def all_named_elems(e):
    for n in e.iter():
        if "name" in n.attrib:
            yield n


def retag_subtree(subtree, suffix):
    # rename names on bodies/joints/geoms/etc. and fix joint references inside actuators later
    for n in all_named_elems(subtree):
        n.set("name", f'{n.get("name")}_{suffix}')
    # also fix joint names referenced by geoms? (geoms don't reference joints; actuators handled later)
    return subtree


# grab sections we want to reuse as-is
assets = root.find("asset")
defaults = root.find("default")

# find the single robot base body and actuators from your file
worldbody_src = root.find("worldbody")
base_body_src = worldbody_src.find("./body[@name='Base']")
actuators_src = root.find("actuator")

if base_body_src is None or actuators_src is None:
    raise RuntimeError("Could not locate Base body or actuator section in source XML.")

# --- 2) Build a new wrapper MJCF scene ---
scene = ET.Element("mujoco", model="two_arms")
ET.SubElement(scene, "compiler", angle="radian", meshdir="assets/")  # keep your meshdir
ET.SubElement(scene, "option", integrator="Euler")

# reuse assets & defaults from your file (shared across both robots)
if assets is not None:
    scene.append(copy.deepcopy(assets))
if defaults is not None:
    scene.append(copy.deepcopy(defaults))

worldbody_new = ET.SubElement(scene, "worldbody")
# simple ground & a light
ET.SubElement(
    worldbody_new, "geom", name="floor", type="plane", size="5 5 0.1", rgba="0.9 0.9 0.9 1"
)
ET.SubElement(worldbody_new, "light", pos="0 0 3", dir="0 0 -1", diffuse=".8 .8 .8")


# --- 3) Make two robot instances with distinct names/poses ---
def make_robot_instance(suffix, pos):
    wrapper = ET.SubElement(worldbody_new, "body", name=f"robot_{suffix}", pos=pos)
    robot = copy.deepcopy(base_body_src)
    retag_subtree(robot, suffix)  # rename Base -> Base_suffix etc.
    wrapper.append(robot)


make_robot_instance("r1", "-0.35 0 0")
make_robot_instance("r2", "0.35 0 0")

# --- 4) Duplicate actuators, retarget to renamed joints ---
actuators_new = ET.SubElement(scene, "actuator")
for which in ("r1", "r2"):
    for a in actuators_src.findall("position"):
        a2 = copy.deepcopy(a)
        # rename actuator itself
        if "name" in a2.attrib:
            a2.set("name", f'{a2.get("name")}_{which}')
        # retarget joint="JointName" -> "JointName_suffix"
        if "joint" in a2.attrib:
            a2.set("joint", f'{a2.get("joint")}_{which}')
        actuators_new.append(a2)

# (optional) keep contacts/keys if you want; they’re not required for a quick viz
# Minimal viewer doesn’t need keyframes/excludes; omit for brevity.

# --- 5) Write scene and launch viewer ---
tree = ET.ElementTree(scene)
tree.write(OUT, encoding="utf-8", xml_declaration=True)

print(f"Wrote {OUT.resolve()}")

# Run the viewer
m = mujoco.MjModel.from_xml_path(str(OUT))
d = mujoco.MjData(m)
with viewer.launch_passive(m, d) as v:
    while v.is_running():
        mujoco.mj_step(m, d)
