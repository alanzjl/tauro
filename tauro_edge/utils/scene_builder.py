"""MuJoCo scene builder for multi-robot simulations."""

import copy
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml


class SceneBuilder:
    """Build MuJoCo scenes with multiple robots dynamically."""

    def __init__(self, base_scene_path: Path, robot_config_path: Path):
        """Initialize scene builder.

        Args:
            base_scene_path: Path to base scene XML (floor, table, objects)
            robot_config_path: Path to YAML config with robot placements
        """
        self.base_scene_path = base_scene_path
        self.robot_config_path = robot_config_path

        # Parse base scene
        self.base_tree = ET.parse(base_scene_path)
        self.base_root = self.base_tree.getroot()

        # Load robot configuration
        with open(robot_config_path) as f:
            self.robot_config = yaml.safe_load(f)

    def _rename_elements(self, element: ET.Element, prefix: str):
        """Recursively rename all named elements with a prefix.

        Args:
            element: XML element to process
            prefix: Prefix to add to names
        """
        # Rename the element itself if it has a name
        if "name" in element.attrib:
            original_name = element.get("name")
            element.set("name", f"{prefix}/{original_name}")

        # Handle special attributes that reference names
        if "joint" in element.attrib:
            original_joint = element.get("joint")
            element.set("joint", f"{prefix}/{original_joint}")

        if "body1" in element.attrib:
            original_body1 = element.get("body1")
            # Special case for "Base" which gets renamed to prefix directly
            if original_body1 == "Base":
                element.set("body1", prefix)
            else:
                element.set("body1", f"{prefix}/{original_body1}")

        if "body2" in element.attrib:
            original_body2 = element.get("body2")
            # Special case for "Base" which gets renamed to prefix directly
            if original_body2 == "Base":
                element.set("body2", prefix)
            else:
                element.set("body2", f"{prefix}/{original_body2}")

        # Recursively process children
        for child in element:
            self._rename_elements(child, prefix)

    def _load_robot_model(self, robot_type: str) -> ET.Element:
        """Load a robot model XML.

        Args:
            robot_type: Type of robot (e.g., "so100_follower")

        Returns:
            Parsed XML root element
        """
        # Map robot types to model directories
        model_mapping = {
            "so100_follower": "so_arm100",
            "so101_follower": "so_arm100",  # Using same model for now
            "so_arm100": "so_arm100",  # Direct mapping for compatibility
        }

        model_dir = model_mapping.get(robot_type, robot_type)

        robot_path = (
            Path(__file__).parent.parent.parent
            / "tauro_common"
            / "models"
            / model_dir
            / f"{model_dir}.xml"
        )

        if not robot_path.exists():
            raise FileNotFoundError(f"Robot model not found: {robot_path}")

        tree = ET.parse(robot_path)
        return tree.getroot()

    def _merge_assets(self, scene_root: ET.Element, robot_root: ET.Element, robot_id: str):
        """Merge robot assets into scene, avoiding duplicates.

        Args:
            scene_root: Scene XML root
            robot_root: Robot XML root
            robot_id: Unique robot identifier
        """
        scene_assets = scene_root.find("asset")
        robot_assets = robot_root.find("asset")

        if scene_assets is None:
            scene_assets = ET.SubElement(scene_root, "asset")

        if robot_assets is not None:
            # Check which assets already exist
            existing_meshes = {mesh.get("name") for mesh in scene_assets.findall("mesh")}
            existing_materials = {mat.get("name") for mat in scene_assets.findall("material")}

            # Add non-duplicate meshes
            for mesh in robot_assets.findall("mesh"):
                mesh_name = mesh.get("name")
                if mesh_name not in existing_meshes:
                    scene_assets.append(copy.deepcopy(mesh))

            # Add non-duplicate materials
            for material in robot_assets.findall("material"):
                mat_name = material.get("name")
                if mat_name not in existing_materials:
                    scene_assets.append(copy.deepcopy(material))

    def _merge_defaults(self, scene_root: ET.Element, robot_root: ET.Element):
        """Merge robot defaults into scene, avoiding duplicates.

        Args:
            scene_root: Scene XML root
            robot_root: Robot XML root
        """
        scene_defaults = scene_root.find("default")
        robot_defaults = robot_root.find("default")

        if robot_defaults is not None and scene_defaults is None:
            # If scene has no defaults, insert robot defaults before worldbody
            defaults_copy = copy.deepcopy(robot_defaults)
            # Find worldbody position
            worldbody = scene_root.find("worldbody")
            if worldbody is not None:
                # Insert before worldbody
                worldbody_index = list(scene_root).index(worldbody)
                scene_root.insert(worldbody_index, defaults_copy)
            else:
                # No worldbody yet, append
                scene_root.append(defaults_copy)
        elif robot_defaults is not None and scene_defaults is not None:
            # Check if the robot default class already exists
            robot_class = robot_defaults.find("./default[@class='so_arm100']")
            if robot_class is not None:
                existing_class = scene_defaults.find("./default[@class='so_arm100']")
                if existing_class is None:
                    scene_defaults.append(copy.deepcopy(robot_class))

    def build_scene(self) -> str:
        """Build the complete scene with all robots.

        Returns:
            Path to temporary XML file with complete scene
        """
        # Start with base scene
        scene_root = copy.deepcopy(self.base_root)

        # Process robots first to merge assets and defaults
        robots = self.robot_config.get("robots", [])
        if robots:
            # Load first robot to get assets and defaults
            first_robot_type = robots[0].get("type", "so100_follower")
            first_robot_root = self._load_robot_model(first_robot_type)

            # Merge assets and defaults BEFORE worldbody
            self._merge_assets(scene_root, first_robot_root, robots[0]["id"])
            self._merge_defaults(scene_root, first_robot_root)

        # Find or create worldbody
        worldbody = scene_root.find("worldbody")
        if worldbody is None:
            worldbody = ET.SubElement(scene_root, "worldbody")

        # Find or create actuator section
        actuators = scene_root.find("actuator")
        if actuators is None:
            actuators = ET.SubElement(scene_root, "actuator")

        # Process each robot in configuration
        for robot_def in robots:
            robot_id = robot_def["id"]
            robot_type = robot_def.get("type", "so100_follower")
            position = robot_def.get("position", [0, 0, 0])
            rotation = robot_def.get("rotation", [0, 0, 0])

            # Load robot model
            robot_root = self._load_robot_model(robot_type)

            # Find robot body in the model
            robot_worldbody = robot_root.find("worldbody")
            if robot_worldbody is None:
                continue

            robot_base = robot_worldbody.find("./body[@name='Base']")
            if robot_base is None:
                # Try to find first body
                robot_base = robot_worldbody.find("body")
                if robot_base is None:
                    continue

            # Copy robot body and rename all elements
            robot_body = copy.deepcopy(robot_base)
            self._rename_elements(robot_body, robot_id)

            # Apply position and rotation directly to the robot base
            robot_body.set("name", robot_id)
            robot_body.set("pos", " ".join(str(p) for p in position))
            if rotation != [0, 0, 0]:
                robot_body.set("euler", " ".join(str(r) for r in rotation))

            # Add to worldbody
            worldbody.append(robot_body)

            # Copy and adapt actuators
            robot_actuators = robot_root.find("actuator")
            if robot_actuators is not None:
                for actuator in robot_actuators:
                    act_copy = copy.deepcopy(actuator)
                    self._rename_elements(act_copy, robot_id)
                    actuators.append(act_copy)

            # Copy and adapt contact exclusions
            robot_contact = robot_root.find("contact")
            if robot_contact is not None:
                scene_contact = scene_root.find("contact")
                if scene_contact is None:
                    scene_contact = ET.SubElement(scene_root, "contact")

                for exclude in robot_contact.findall("exclude"):
                    exclude_copy = copy.deepcopy(exclude)
                    self._rename_elements(exclude_copy, robot_id)
                    scene_contact.append(exclude_copy)

        # Write to temporary file
        temp_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".xml", delete=False, dir=Path(self.base_scene_path).parent
        )

        tree = ET.ElementTree(scene_root)
        tree.write(temp_file.name, encoding="unicode", xml_declaration=True)
        temp_file.close()

        return temp_file.name
