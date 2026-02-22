import json
import base64
import requests
import numpy as np
from PIL import Image, ImageDraw
from typing import List, Dict, Tuple, Optional
import io

OLLAMA_URL = "http://localhost:11434"
POSE_API_URL = "http://localhost:8001"
EMBEDDING_MODEL = "nomic-embed-text"


def load_poses_database(path: str) -> List[Dict]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_embedding(text: str) -> Optional[List[float]]:
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/embeddings",
            json={"model": EMBEDDING_MODEL, "prompt": text},
            timeout=30
        )
        response.raise_for_status()
        return response.json()["embedding"]
    except Exception as e:
        print(f"Ошибка: {e}")
        return None


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    a, b = np.array(vec1), np.array(vec2)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


class PoseRAG:    
    def __init__(self, poses: List[Dict]):
        self.poses = poses
        self.embeddings: List[Optional[List[float]]] = []
        self._use_embeddings = False
    
    def build_index(self) -> bool:
        test_emb = get_embedding("test")
        if test_emb is None:
            return False
        
        self.embeddings = []
        for i, pose_data in enumerate(self.poses):
            emb = get_embedding(pose_data["description"])
            self.embeddings.append(emb)
            if (i + 1) % 20 == 0:
                print(f"Обработано {i + 1}/{len(self.poses)}")
        
        valid = sum(1 for e in self.embeddings if e is not None)
        self._use_embeddings = valid > 0
        print(f"Индекс: {valid}/{len(self.poses)} эмбеддингов")
        return self._use_embeddings
    
    def search(self, query: str, top_k: int = 8) -> List[Tuple[Dict, float]]:
        if self._use_embeddings:
            return self._semantic_search(query, top_k)
        return self._keyword_search(query, top_k)
    
    def _semantic_search(self, query: str, top_k: int) -> List[Tuple[Dict, float]]:
        query_emb = get_embedding(query)
        if query_emb is None:
            return self._keyword_search(query, top_k)
        
        scores = []
        for i, pose_emb in enumerate(self.embeddings):
            if pose_emb is not None:
                score = cosine_similarity(query_emb, pose_emb)
                scores.append((self.poses[i], score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
    
    def _keyword_search(self, query: str, top_k: int) -> List[Tuple[Dict, float]]:
        query_lower = query.lower()
        keywords = query_lower.split()
        
        scores = []
        for pose_data in self.poses:
            desc = pose_data["description"].lower()
            score = sum(2 if kw in desc else 0 for kw in keywords)
            
            if score > 0:
                scores.append((pose_data, float(score)))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


def render_pose(pose: Dict) -> Image.Image:
    response = requests.post(
        f"{POSE_API_URL}/visualize",
        json={"pose": pose},
        timeout=10
    )
    data = response.json()
    img_data = base64.b64decode(data["image"])
    return Image.open(io.BytesIO(img_data))


def add_text(img: Image.Image, text: str, frame_num: int) -> Image.Image:
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), f"#{frame_num}", fill='#4ecca3')
    text = text[:42] + "..." if len(text) > 45 else text
    draw.text((10, img.height - 30), text, fill='#eeeeee')
    return img


def create_gif(poses: List[Dict], output_path: str, duration: int = 400, show_text: bool = True) -> str:
    frames = []
    
    for i, pose_data in enumerate(poses):
        frame = render_pose(pose_data["pose"])
        
        if show_text:
            frame = add_text(frame, pose_data["description"], i + 1)
        
        frames.append(frame)
        print(f"Кадр {i+1}: {pose_data['description'][:50]}...")
    
    if frames:
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration,
            loop=0
        )
        print(f"Сохранено: {output_path}")
    
    return output_path


def generate_animation(query: str, poses_db_path: str, output_path: str = "animation.gif") -> str:
    poses = load_poses_database(poses_db_path)
    print(f"Загружено {len(poses)} поз")
    
    rag = PoseRAG(poses)
    rag.build_index()
        
    found = [p for p, _ in rag.search(query, top_k=8)]
    
    for i, p in enumerate(found):
        print(f"{i+1}. {p['description'][:55]}...")

    gif_path = create_gif(found, output_path)
    
    return gif_path


if __name__ == "__main__":
    import sys
    
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "танец макарена"
    
    generate_animation(
        query=query,
        poses_db_path="poses_database.json",
        output_path=f"{query}.gif"
    )