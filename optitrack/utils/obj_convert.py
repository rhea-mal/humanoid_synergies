import bpy
import os

def find_obj_files(directory):
    obj_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.obj'):
                obj_files.append(os.path.join(root, file))
    return obj_files

def load_and_export_obj_files(obj_files, export_directory):
    if not os.path.exists(export_directory):
        os.makedirs(export_directory)
    
    for obj_file in obj_files:
        # Clear existing scene
        bpy.ops.wm.read_factory_settings(use_empty=True)
        
        # Import the obj file
        bpy.ops.import_scene.obj(filepath=obj_file)
        
        # Define the export path
        obj_filename = os.path.basename(obj_file)
        export_path = os.path.join(export_directory, obj_filename)
        
        # Export the obj file
        bpy.ops.export_scene.obj(filepath=export_path)
        
        print(f"Exported: {export_path}")

def main():
    source_directory = "/Users/rheamalhotra/Desktop/robotics/optitrack-robot-dance/model/HRP4c/graphics"
    export_directory = "//Users/rheamalhotra/Desktop/robotics/optitrack-robot-dance/model/HRP4c/graphics_new"
    
    obj_files = find_obj_files(source_directory)
    load_and_export_obj_files(obj_files, export_directory)

if __name__ == "__main__":
    main()
