import bpy
import mathutils
import random
import math
import json
from math import radians, degrees
from mathutils import Vector, Euler
import os
from tqdm import trange
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Create PseudoKitchens datasets")

    subparsers = parser.add_subparsers(dest="command", required=True)

    create_parser = subparsers.add_parser("create", help="Create a Pseudokitchens dataset")
    create_parser.add_argument("--train-size", type=int, required=True, help="Number of samples for training")
    create_parser.add_argument("--val-size", type=int, required=True, help="Number of samples for validation")
    create_parser.add_argument("--test-size", type=int, required=True, help="Number of samples for testing")
    create_parser.add_argument("--recipeless-test-size", type=int, default=0, help="Number of samples for recipeless testing")
    create_parser.add_argument("--dataset-dir", type=Path, required=True, help="Path to the dataset output directory")

    generate_examples_parser = subparsers.add_parser("generate_examples", help="Generate instances for a PseudoKitchens dataset")
    generate_examples_parser.add_argument("--dir", type=Path, required=True, help="Directory to save generated instances")
    generate_examples_parser.add_argument("--first-number", type=int, required=True, help="First number to use for naming instances")
    generate_examples_parser.add_argument("--last-number", type=int, required=True, help="Last number to use for naming instances")
    generate_examples_parser.add_argument("--recipeless", action="store_true", help="Generate instances without recipes")
    generate_examples_parser.add_argument("--filename-length", type=int, default=0, help="Length of the filename for each instance")

    ingredient_distribution_parser = subparsers.add_parser("ingredient_distribution", help="Print the distribution of ingredients in the instances in a folder")
    ingredient_distribution_parser.add_argument("--dir", type=Path, required=True, help="Directory containing the instances to analyse")

    theoretical_ingredient_distribution_parser = subparsers.add_parser("theoretical_ingredient_distribution", help="Print the theoretical distribution of ingredients in the dataset")

    args = parser.parse_args()

    return args

def hide(obj):
    obj.hide_render = True
    obj.hide_set(True)
    for child in obj.children:
        hide(child)

def show(obj):
    obj.hide_render = False
    obj.hide_set(False)
    for child in obj.children:
        show(child)

kelvin_table = {
    3000: (255, 180, 107),
    3100: (255, 184, 114),
    3200: (255, 187, 120),
    3300: (255, 190, 126),
    3400: (255, 193, 132),
    3500: (255, 196, 137),
    3600: (255, 199, 143),
    3700: (255, 201, 148),
    3800: (255, 204, 153),
    3900: (255, 206, 159),
    4000: (255, 209, 163),
    4100: (255, 211, 168),
    4200: (255, 213, 173),
    4300: (255, 215, 177),
    4400: (255, 217, 182),
    4500: (255, 219, 186),
    4600: (255, 221, 190),
    4700: (255, 223, 194),
    4800: (255, 225, 198),
    4900: (255, 227, 202),
    5000: (255, 228, 206),
    5100: (255, 230, 210),
    5200: (255, 232, 213),
    5300: (255, 233, 217),
    5400: (255, 235, 220),
    5500: (255, 236, 224),
    5600: (255, 238, 227),
    5700: (255, 239, 230),
    5800: (255, 240, 233),
    5900: (255, 242, 236),
    6000: (255, 243, 239),
    6100: (255, 244, 242),
    6200: (255, 245, 245),
    6300: (255, 246, 247),
    6400: (255, 248, 251),
    6500: (255, 249, 253),
    6600: (254, 249, 255),
    6700: (252, 247, 255),
    6800: (249, 246, 255),
    6900: (247, 245, 255),
    7000: (245, 243, 255),
    7100: (243, 242, 255),
    7200: (240, 241, 255),
    7300: (239, 240, 255),
    7400: (237, 239, 255),
    7500: (235, 238, 255),
    7600: (233, 237, 255),
    7700: (231, 236, 255),
    7800: (230, 235, 255),
    7900: (228, 234, 255),
    8000: (227, 233, 255),
}

ingredient_groups = {
    "Fruit": ["Banana", "Orange", "Apple", "Pear", "Pineapple"],
    "Vegetables": ["Onion", "Carrot", "Potato", "Pepper", "Courgette"],
    "Pasta": ["Macaroni", "Spaghetti"]
}

recipes = {
    "Fruit Salad": [
        "Fruit"
    ],
    "Vegetable Pasta": [
        "Pasta",
        "Onion",
        "Garlic",
        "Oil",
        "Vegetables",
        "Spice",
        "Tin Tomatoes"
    ],
    "Risotto": [
        "Cheese",
        "Onion",
        "Garlic",
        "Vegetables",
        "Oil",
        "Spice",
        "Rice"
    ],
    "Chips": [
        "Potato",
        "Oil",
        "Flour",
        "Garlic",
        "Spice"
    ],
    "Chilli": [
        "Mince",
        "Oil",
        "Onion",
        "Garlic",
        "Chilli",
        "Tin Tomatoes",
        "Spice",
        "Rice"
    ],
    "Smoothie": [
        "Milk",
        "Yoghurt",
        "Fruit"
    ],
    "Hot Chocolate": [
        "Chocolate",
        "Milk"
    ],
    "Banana Bread": [
        "Butter",
        "Sugar",
        "Egg",
        "Flour",
        "Banana"
    ],
    "Chocolate Fudge Cake": [
        "Egg",
        "Sugar",
        "Oil",
        "Flour",
        "Chocolate",
        "Syrup",
        "Milk"
    ],
    "Carbonara": [
        "Garlic",
        "Meat",
        "Butter",
        "Cheese",
        "Egg",
        "Spaghetti",
        "Spice"
    ]
}
recipes_ordered = sorted(recipes.keys())

ingredients = set()
for group_ingredients in ingredient_groups.values():
    ingredients.update(group_ingredients)
    
for recipe_ingredients in recipes.values():
    for ingredient in recipe_ingredients:
        if ingredient not in ingredient_groups:
            ingredients.add(ingredient)

ingredient_probabilities = {}
for ingredient in ingredients:
    ingredient_probabilities[ingredient] = 0.0
    for recipe_ingredients in recipes.values():
        if ingredient in recipe_ingredients:
            ingredient_probabilities[ingredient] += 1.0 / len(recipes)
            continue
        for group_name, group_ingredients in ingredient_groups.items():
            if group_name in recipe_ingredients and ingredient in group_ingredients:
                usable_group_ingredients = [i for i in group_ingredients if i not in recipe_ingredients]
                if group_name == "Pasta":
                    ingredient_probabilities[ingredient] += (1.0 / len(recipes)) * (1.0 / len(usable_group_ingredients))
                else:
                    ingredient_probabilities[ingredient] += (1.0 / len(recipes)) * (0.5 + 1.0 / (2 * len(usable_group_ingredients)))
                break

def count_objects():
    ingredient_counts = {}
    for ingredient in sorted(ingredients):
        i = 1
        while f"{ingredient} {i}" in bpy.data.objects:
            i += 1

        assert i > 1, f"Could not find any {ingredient} objects."

        ingredient_counts[ingredient] = i - 1

        print(f"Found {i - 1} {ingredient} object(s).")

    n_random_objects = 0
    while f"Object {n_random_objects + 1}" in bpy.data.objects:
        n_random_objects += 1
    print(f"Found {n_random_objects} random object(s).")

    n_kitchens = 0
    while f"Kitchen {n_kitchens + 1}" in bpy.data.collections:
        kitchen_name = f"Kitchen {n_kitchens + 1}"
        n_kitchens += 1
    print(f"Found {n_kitchens} kitchen(s).")

    return {
        "ingredient_counts": ingredient_counts,
        "n_random_objects": n_random_objects,
        "n_kitchens": n_kitchens
    }

def hide_all(object_counts):
    vlayer = bpy.context.scene.view_layers["ViewLayer"]
    for ingredient, count in object_counts["ingredient_counts"].items():
        for i in range(1, count + 1):
            hide(bpy.data.objects[f"{ingredient} {i}"])
    for i in range(1, object_counts["n_random_objects"] + 1):
        hide(bpy.data.objects[f"Object {i}"])
    for i in range(1, object_counts["n_kitchens"] + 1):
        bpy.data.collections[f"Kitchen {i}"].hide_render = True
        vlayer.layer_collection.children["Kitchens"].children[f"Kitchen {i}"].hide_viewport = True

def set_random_floor():
    floor_number = random.choice([1, 2, 3, 4, 5])
    bpy.data.objects["Walls"].data.materials[1] = bpy.data.materials[f"Floor {floor_number}"]
    return floor_number

def set_random_wall():
    wall_number = random.choice([1, 2, 3, 4, 5])
    bpy.data.objects["Walls"].data.materials[0] = bpy.data.materials[f"Wall {wall_number}"]
    return wall_number

def set_random_light_position():
    light_object = bpy.data.objects["Light"]

    x_min = light_object["x_min"]
    x_max = light_object["x_max"]
    random_x = random.uniform(x_min, x_max)

    y_min = light_object["y_min"]
    y_max = light_object["y_max"]
    random_y = random.uniform(y_min, y_max)
    
    position = Vector((random_x, random_y, light_object.location.z))

    light_object.location = position

    return (random_x, random_y)

def set_random_light_power():
    light_object = bpy.data.objects["Light"]
    
    power_min = light_object["power_min"]
    power_max = light_object["power_max"]

    random_power = random.uniform(power_min, power_max)
    
    light_object.data.energy = random_power
    
    return random_power

def set_random_light_colour():
    light_object = bpy.data.objects["Light"]

    temperature = random.choice(list(kelvin_table.keys()))
    c = kelvin_table[temperature]
    colour = (c[0] / 255.0, c[1] / 255.0, c[2] / 255.0)

    light_object.data.color = colour
    return colour

def set_random_camera_position(kitchen_name):
    camera_pivot_object = bpy.data.objects[f"{kitchen_name} camera pivot"]

    rotation_x_min = camera_pivot_object["rotation_x_min"]
    rotation_x_max = camera_pivot_object["rotation_x_max"]
    random_rotation_x = random.uniform(rotation_x_min, rotation_x_max)

    rotation_z_min = camera_pivot_object["rotation_z_min"]
    rotation_z_max = camera_pivot_object["rotation_z_max"]
    random_rotation_z = random.uniform(rotation_z_min, rotation_z_max)

    scale_min = camera_pivot_object["scale_min"]
    scale_max = camera_pivot_object["scale_max"]
    random_scale = random.uniform(scale_min, scale_max)

    camera_pivot_object.rotation_euler = (random_rotation_x, 0, random_rotation_z)

    camera_pivot_object.scale = (random_scale, random_scale, random_scale)

    return random_rotation_x, random_rotation_z, random_scale

def set_random_egg_box_openness():
    random_rotation_x = radians(random.uniform(-147, 0))
    bpy.data.objects["Egg 5"].pose.bones["CAP"].rotation_euler = (random_rotation_x, 0, 0)
    return random_rotation_x

def apply_random_transform(object_name, scale_range=(0.8, 1.2)):
    """Apply random rotation and scaling to object"""
    obj = bpy.data.objects[object_name]
    
    # Random rotation around Z-axis (0 to 360 degrees)
    random_rotation = random.uniform(0, 2 * math.pi)
    obj.rotation_euler = (0, 0, random_rotation)

    # Random uniform scaling
    random_scale = random.uniform(scale_range[0], scale_range[1])
    obj.scale = (random_scale, random_scale, random_scale)

    # Update the scene to reflect changes
    bpy.context.view_layer.update()
    
    return degrees(random_rotation), random_scale

class Surfaces:
    """Manages multiple flat surfaces and tracks object placements"""

    def __init__(self, kitchen_name):
        """
        Initialize surfaces for a given kitchen
        
        Args:
            kitchen_name: String identifier for the kitchen (e.g., "Kitchen 1")
        """
        self.kitchen_name = kitchen_name
        self.surfaces = {}
        self.placed_objects = {}  # surface_name -> list of object bounds

        # Find all surfaces matching the pattern
        self._discover_surfaces()

    def _discover_surfaces(self):
        """Find all surfaces matching the pattern '{kitchen} surface {i}'"""
        i = 1
        while f"{self.kitchen_name} surface {i}" in bpy.data.objects:
            surface_name = f"{self.kitchen_name} surface {i}"
            surface_obj = bpy.data.objects[surface_name]
            if surface_obj.type == "MESH":
                # Calculate surface area and store surface info
                bounds = self._get_object_bounds(surface_obj)
                area = bounds["size"].x * bounds["size"].y

                self.surfaces[surface_name] = {
                    "object": surface_obj,
                    "area": area,
                    "bounds": bounds,
                    "z_level": bounds["max"].z  # Assuming flat surface
                }
                self.placed_objects[surface_name] = []
                print(f"Found surface: {surface_name} (area: {area:.2f})")
            else:
                print(f"Warning: {surface_name} is not a mesh object")
            i += 1

        if not self.surfaces:
            raise ValueError(f"No surfaces found for kitchen '{self.kitchen_name}'")
        
        print(f"Discovered {len(self.surfaces)} surfaces for {self.kitchen_name}")
    
    def _get_object_bounds(self, obj):
        """Get the bounding box of an object and all its children in world coordinates"""

        def collect_bounds(current_obj):
            """Recursively collect bounding box corners from object and its children"""
            # Add current object's bounding box corners
            if current_obj.type == "MESH":
                depsgraph = bpy.context.evaluated_depsgraph_get()
                eval_obj = current_obj.evaluated_get(depsgraph)
                mesh = eval_obj.to_mesh()
                vertices = [current_obj.matrix_world @ v.co for v in mesh.vertices]

                min_x = min(vertex.x for vertex in vertices)
                max_x = max(vertex.x for vertex in vertices)
                min_y = min(vertex.y for vertex in vertices)
                max_y = max(vertex.y for vertex in vertices)
                min_z = min(vertex.z for vertex in vertices)
                max_z = max(vertex.z for vertex in vertices)
            else:
                min_x = min_y = min_z = float("inf")
                max_x = max_y = max_z = float("-inf")

            # Recursively process children
            for child in current_obj.children:
                min_x_c, min_y_c, min_z_c, max_x_c, max_y_c, max_z_c = collect_bounds(child)
                min_x = min(min_x, min_x_c)
                min_y = min(min_y, min_y_c)
                min_z = min(min_z, min_z_c)
                max_x = max(max_x, max_x_c)
                max_y = max(max_y, max_y_c)
                max_z = max(max_z, max_z_c)

            return min_x, min_y, min_z, max_x, max_y, max_z

        # Collect all bounding box corners
        min_x, min_y, min_z, max_x, max_y, max_z = collect_bounds(obj)

        min_bound = Vector((min_x, min_y, min_z))
        max_bound = Vector((max_x, max_y, max_z))

        # Calculate displacement from object's origin (world position)
        obj_origin = obj.matrix_world.translation

        # Calculate min/max displacement in each direction from object's origin
        min_displacement = Vector((
            min_x - obj_origin.x,  # Furthest extent in negative x direction
            min_y - obj_origin.y,  # Furthest extent in negative y direction
            min_z - obj_origin.z   # Furthest extent in negative z direction
        ))

        max_displacement = Vector((
            max_x - obj_origin.x,  # Furthest extent in positive x direction
            max_y - obj_origin.y,  # Furthest extent in positive y direction
            max_z - obj_origin.z   # Furthest extent in positive z direction
        ))

        return {
            "min": min_bound,
            "max": max_bound,
            "size": max_bound - min_bound,
            "displacement": {
                "min": min_displacement,
                "max": max_displacement
            }
        }

    def _choose_surface_weighted_by_area(self):
        """Choose a surface randomly, weighted by area"""
        if not self.surfaces:
            return None
        
        # Create weighted list based on areas
        surface_names = list(self.surfaces.keys())
        areas = [self.surfaces[name]["area"] for name in surface_names]
        
        # Use random.choices for weighted selection
        chosen_surface = random.choices(surface_names, weights=areas, k=1)[0]
        return chosen_surface

    def _check_collision_on_surface(self, obj_bounds, surface_name, safety_margin=0.01):
        """Check if object bounds would collide with existing objects on surface"""
        placed_objects = self.placed_objects[surface_name]
        
        for existing_bounds in placed_objects:
            # Check for bounding box overlap with safety margin
            if (obj_bounds["min"].x - safety_margin < existing_bounds["max"].x and
                obj_bounds["max"].x + safety_margin > existing_bounds["min"].x and
                obj_bounds["min"].y - safety_margin < existing_bounds["max"].y and
                obj_bounds["max"].y + safety_margin > existing_bounds["min"].y):
                return True  # Collision detected
        
        return False  # No collision

    def place(self, object_name, max_attempts=1000, safety_margin=0.01):
        """
        Place an object on one of the surfaces

        Args:
            object_name: Name of the object to place
            max_attempts: Maximum number of placement attempts
            safety_margin: Minimum distance from other objects

        Returns:
            bool: True if placement was successful, False otherwise
        """
        obj_to_place = bpy.data.objects[object_name]

        # Get object bounds after transformation
        obj_bounds = self._get_object_bounds(obj_to_place)
        obj_size = obj_bounds["size"]

        # Calculate the bottom offset (distance from object center to bottom)
        obj_bottom_offset = obj_bounds["displacement"]["min"].z
        
        # Try to place the object
        for attempt in range(max_attempts):
            surface_name = self._choose_surface_weighted_by_area()
            
            surface_info = self.surfaces[surface_name]
            surface_bounds = surface_info["bounds"]
            z_level = surface_info["z_level"]
            
            # Calculate minimum/maximum x and y coordinates that would place the object
            # within the surface
            min_x = surface_bounds["min"].x - obj_bounds["displacement"]["min"].x
            max_x = surface_bounds["max"].x - obj_bounds["displacement"]["max"].x
            min_y = surface_bounds["min"].y - obj_bounds["displacement"]["min"].y
            max_y = surface_bounds["max"].y - obj_bounds["displacement"]["max"].y
            
            if min_x > max_x or min_y > max_y:
                # Object does not fit on surface
                print("Object does not fit on surface.")
                continue

            # Generate random position within surface bounds
            random_x = random.uniform(min_x, max_x)
            random_y = random.uniform(min_y, max_y)
            
            # Calculate final position (bottom of object touches surface)
            final_position = Vector((random_x, random_y, z_level - obj_bottom_offset))
            
            # Move object to test position
            obj_to_place.location = final_position
            bpy.context.view_layer.update()

            # Get bounds at new position
            new_bounds = self._get_object_bounds(obj_to_place)

            # Check for collisions with existing objects on this surface
            if not self._check_collision_on_surface(new_bounds, surface_name, safety_margin):
                # Successful placement!
                self.placed_objects[surface_name].append(new_bounds)
                print(f"Object '{object_name}' placed on '{surface_name}' at {final_position}")
                print(f"obj_bottom_offset={obj_bottom_offset}")
                return (final_position.x, final_position.y, final_position.z)

        # Failed to place after max attempts
        print(f"Failed to place object '{object_name}' after {max_attempts} attempts")
        return None

def prepare_instance(object_counts):
    hide_all(object_counts)
    instance_data = {}

    kitchen_number = random.randint(1, object_counts["n_kitchens"])
    instance_data["kitchen_number"] = kitchen_number
    kitchen_name = f"Kitchen {kitchen_number}"
    bpy.data.collections[kitchen_name].hide_render = False
    vlayer = bpy.context.scene.view_layers["ViewLayer"]
    vlayer.layer_collection.children["Kitchens"].children[kitchen_name].hide_viewport = False
    
    (instance_data["camera_rotation_x"],
     instance_data["camera_rotation_z"],
     instance_data["camera_scale"]) = set_random_camera_position(f"Kitchen {kitchen_number}")
     
    bpy.context.scene.camera = bpy.data.objects[f"Kitchen {kitchen_number} camera"]
    
    instance_data["floor_number"] = set_random_floor()
    instance_data["wall_number"] = set_random_wall()
    
    instance_data["light_x"], instance_data["light_y"] = set_random_light_position()
    instance_data["light_power"] = set_random_light_power()
    instance_data["light_colour"] = set_random_light_colour()
  
    return instance_data

def place_ingredient(ingredient, object_counts, kitchen_surfaces):
    objects_placed = []
    if ingredient == "Spice":
        n = random.randint(1, 3)
    else:
        n = 1
    object_numbers = random.sample(
        range(1, object_counts["ingredient_counts"][ingredient] + 1), k=n)
    for object_number in object_numbers:
        object_data = {}

        object_name = f"{ingredient} {object_number}"
        object_data["object_name"] = object_name
        
        (object_data["z_rotation"],
         object_data["scale"]) = apply_random_transform(object_name)
            
        object_data["position"] = kitchen_surfaces.place(object_name)
        
        if object_name == "Egg 5":
            object_data["egg_cap_rotation"] = set_random_egg_box_openness()

        if object_data["position"] is not None:
            show(bpy.data.objects[object_name])
            objects_placed.append(object_data)

    return objects_placed

def place_random_object(object_number, kitchen_surfaces):
    object_data = {}

    object_name = f"Object {object_number}"
    object_data["object_name"] = object_name

    (object_data["z_rotation"],
     object_data["scale"]) = apply_random_transform(object_name)

    object_data["position"] = kitchen_surfaces.place(object_name)

    if object_data["position"] is not None:
        show(bpy.data.objects[object_name])
        return object_data
    else:
        return None

def create_random_recipeless_instance(object_counts):
    instance_data = prepare_instance(object_counts)
    
    kitchen_surfaces = Surfaces(f"Kitchen {instance_data['kitchen_number']}")
    instance_data["objects"] = []

    for ingredient in ingredients:
        if random.random() < ingredient_probabilities[ingredient]:
            instance_data["objects"].extend(place_ingredient(ingredient, object_counts, kitchen_surfaces))

    n_random = random.randint(0, 3)
    random_objects = random.sample(range(1, object_counts["n_random_objects"] + 1), k=n_random)
    for object_number in random_objects:
        object_data = place_random_object(object_number, kitchen_surfaces)
        if object_data is not None:
            instance_data["objects"].append(object_data)
        
    return instance_data

def create_random_instance(object_counts):
    instance_data = prepare_instance(object_counts)

    recipe_idx = random.randint(0, len(recipes_ordered) - 1)
    instance_data["recipe_idx"] = recipe_idx
    recipe_name = recipes_ordered[recipe_idx]
    instance_data["recipe_name"] = recipe_name

    kitchen_surfaces = Surfaces(f"Kitchen {instance_data['kitchen_number']}")

    instance_data["objects"] = []

    for ingredient in recipes[recipe_name]:
        ingredients_to_place = []
        if ingredient in ingredient_groups:
            usable_group_ingredients = [i for i in ingredient_groups[ingredient] if i not in recipes[recipe_name]]
            if ingredient == "Pasta":
                n = 1
            else:
                n = random.randint(1, len(usable_group_ingredients))
            ingredients_to_place.extend(random.sample(usable_group_ingredients, k=n))
        else:
            ingredients_to_place.append(ingredient)
        
        for ingredient in ingredients_to_place:
            instance_data["objects"].extend(place_ingredient(ingredient, object_counts, kitchen_surfaces))

    n_random = random.randint(0, 3)
    random_objects = random.sample(range(1, object_counts["n_random_objects"] + 1), k=n_random)
    for object_number in random_objects:
        object_data = place_random_object(object_number, kitchen_surfaces)
        if object_data is not None:
            instance_data["objects"].append(object_data)
        
    return instance_data

def render(instance_data, save_dir, name):
    bpy.context.scene.frame_set(1)
    bpy.context.scene.node_tree.nodes["File Output"].base_path = os.path.join(save_dir, "crypto")
    bpy.data.scenes["Scene"].render.filepath = os.path.join(save_dir, name)
    bpy.context.view_layer.update()
    bpy.ops.render.render(write_still=True)
    os.rename(os.path.join(save_dir, "crypto0001.exr"), os.path.join(save_dir, name + ".exr"))

    with open(os.path.join(save_dir, name + ".json"), "w") as f:
        json.dump(instance_data, f)

def blender_init():
    bpy.ops.wm.open_mainfile(filepath="Kitchen Dataset.blend")
    
    bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "OPTIX"
    bpy.context.preferences.addons["cycles"].preferences.refresh_devices()
    bpy.context.preferences.addons["cycles"].preferences.devices["NVIDIA GeForce RTX 4090"].use = True
    bpy.context.scene.cycles.device = "GPU"

def generate_examples(dir, first_number, last_number, recipeless, object_counts, filename_length=0):
    filename_length = max(filename_length, len(str(last_number)))
    dir.mkdir(parents=True, exist_ok=True)
    for i in trange(first_number, last_number + 1):
        if recipeless:
            instance_data = create_random_recipeless_instance(object_counts)
        else:
            instance_data = create_random_instance(object_counts)
        render(instance_data, str(dir.resolve()), str(i).zfill(filename_length))

def create(train_size, val_size, test_size, recipeless_test_size, dataset_dir):
    blender_init()
    object_counts = count_objects()

    dataset_info = {
        "train_size": train_size,
        "val_size": val_size,
        "test_size": test_size,
        "object_counts": object_counts,
        "ingredient_groups": ingredient_groups,
        "recipes": recipes
    }

    if recipeless_test_size > 0:
        dataset_info["recipeless_test_size"] = recipeless_test_size

    dataset_dir.mkdir(parents=True)
    with (dataset_dir / "info.json").open(mode="w") as f:
        json.dump(dataset_info, f)

    folders = [dataset_dir / "train", dataset_dir / "val", dataset_dir / "test"]
    sizes = [train_size, val_size, test_size]

    for folder, size in list(zip(folders, sizes)):
        generate_examples(folder, 1, size, recipeless=False, object_counts=object_counts)

    if recipeless_test_size > 0:
        recipeless_test_folder = dataset_dir / "recipeless_test"
        generate_examples(recipeless_test_folder, 1, recipeless_test_size, recipeless=True, object_counts=object_counts)

def print_ingredient_distribution(dir):
    ingredient_counts = {ingredient: 0 for ingredient in ingredients}
    total_instances = 0

    for file in dir.glob("*.json"):
        total_instances += 1
        already_counted = set()
        with file.open() as f:
            instance_data = json.load(f)
            for obj in instance_data["objects"]:
                ingredient_name = obj["object_name"].rsplit(" ", 1)[0]
                if ingredient_name in ingredient_counts and ingredient_name not in already_counted:
                    ingredient_counts[ingredient_name] += 1
                    already_counted.add(ingredient_name)

    print("Ingredient distribution:")
    for ingredient, count in sorted(ingredient_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{ingredient}: {count} ({(count / total_instances) * 100:.2f}%)")

if __name__ == "__main__":
    args = parse_args()

    if args.command == "create":
        create(args.train_size, args.val_size, args.test_size, args.recipeless_test_size, args.dataset_dir)
    elif args.command == "generate_examples":
        blender_init()
        object_counts = count_objects()
        generate_examples(args.dir, args.first_number, args.last_number, args.recipeless, object_counts, args.filename_length)
    elif args.command == "ingredient_distribution":
        print_ingredient_distribution(args.dir)
    elif args.command == "theoretical_ingredient_distribution":
        print("Theoretical ingredient distribution:")
        for ingredient, probability in sorted(ingredient_probabilities.items(), key=lambda x: x[1], reverse=True):
            print(f"{ingredient}: {probability:.4f}")
