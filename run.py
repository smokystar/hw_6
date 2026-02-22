import sys
import os
from rag_animation import generate_animation

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def main():
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = "танец макарена"
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    poses_db = os.path.join(script_dir, "poses_database.json")
    
    output_path = f"{query.replace(' ', '_')}.gif"
    
    generate_animation(
        query=query,
        poses_db_path=poses_db,
        output_path=output_path
    )

if __name__ == "__main__":
    main()
