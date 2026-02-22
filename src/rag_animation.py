import json
import base64
import requests
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
from typing import List, Dict, Tuple, Optional
import io

OLLAMA_URL = "http://localhost:11434"
POSE_API_URL = "http://localhost:8001"
EMBEDDING_MODEL = "nomic-embed-text"

IMG_WIDTH = 400
IMG_HEIGHT = 500
FIGURE_SCALE = 3.0


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
        keyword_results = self._keyword_search(query, top_k)
        
        if len(keyword_results) >= top_k:
            return keyword_results[:top_k]
        
        if self._use_embeddings:
            semantic_results = self._semantic_search(query, top_k * 2)
            seen = {p['description'] for p, _ in keyword_results}
            for p, score in semantic_results:
                if p['description'] not in seen:
                    keyword_results.append((p, score))
                if len(keyword_results) >= top_k:
                    break
        
        return keyword_results[:top_k]
    
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


def sort_poses_with_llm(poses: List[Dict], query: str) -> List[Dict]:
    
    poses_desc = "\n".join([
        f"{i}: {p['description']}" 
        for i, p in enumerate(poses)
    ])
    
    prompt = f"""Задача: упорядочить позы для анимации "{query}".

Позы:
{poses_desc}

Верни только индексы в правильном порядке через запятую (например: 2,0,4,1,3,5,6,7).
Ответ:"""

    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": "qwen2.5:1.5b",
                "prompt": prompt,
                "stream": False
            },
            timeout=30
        )
        response.raise_for_status()
        
        result = response.json()["response"].strip()
        print(f"LLM ответ: {result}")
        
        indices = [int(x.strip()) for x in result.split(",") if x.strip().isdigit()]
        
        indices = [i for i in indices if 0 <= i < len(poses)]
        
        sorted_poses = [poses[i] for i in indices]
        
        for p in poses:
            if p not in sorted_poses:
                sorted_poses.append(p)
        
        return sorted_poses
        
    except Exception as e:
        print(f"LLM сортировка не удалась: {e}")
        return poses


def visualize_pose_api(pose: Dict) -> Optional[Image.Image]:
    try:
        response = requests.post(
            f"{POSE_API_URL}/visualize",
            json={"pose": pose},
            timeout=10
        )
        response.raise_for_status()
        
        data = response.json()
        
        if data.get("success") and data.get("image"):
            img_data = base64.b64decode(data["image"])
            return Image.open(io.BytesIO(img_data))
        
        print(f"{data}")
        return None
        
    except requests.exceptions.ConnectionError:
        return None
    except Exception as e:
        print(f"Ошибка: {e}")
        return None


def check_pose_api() -> bool:
    try:
        response = requests.get(f"{POSE_API_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def render_pose_builtin(pose: Dict, width: int = IMG_WIDTH, height: int = IMG_HEIGHT) -> Image.Image:
    img = Image.new('RGB', (width, height), color='#1a1a2e')
    draw = ImageDraw.Draw(img)
    
    cx, cy = width // 2, height // 2
    scale = FIGURE_SCALE
    
    torso = pose.get("Torso", [0, 0])
    head = pose.get("Head", [0, 60])
    rh = pose.get("RH", [30, 35])
    lh = pose.get("LH", [-30, 35])
    rk = pose.get("RK", [15, -50])
    lk = pose.get("LK", [-15, -50])
    
    torso_x = cx + torso[0] * scale
    torso_y = cy - torso[1] * scale
    
    head_x = cx + head[0] * scale
    head_y = cy - head[1] * scale - 60
    
    shoulder_y = torso_y - 30
    r_shoulder = (torso_x + 30, shoulder_y)
    l_shoulder = (torso_x - 30, shoulder_y)
    
    r_hand = (r_shoulder[0] + rh[0] * scale * 0.9, r_shoulder[1] - rh[1] * scale * 0.9)
    l_hand = (l_shoulder[0] + lh[0] * scale * 0.9, l_shoulder[1] - lh[1] * scale * 0.9)
    
    hip_y = torso_y + 50
    r_hip = (torso_x + 18, hip_y)
    l_hip = (torso_x - 18, hip_y)
    
    r_knee = (r_hip[0] + rk[0] * scale * 0.4, r_hip[1] - rk[1] * scale * 0.6)
    l_knee = (l_hip[0] + lk[0] * scale * 0.4, l_hip[1] - lk[1] * scale * 0.6)
    r_foot = (r_knee[0] + rk[0] * scale * 0.3, r_knee[1] + 55)
    l_foot = (l_knee[0] + lk[0] * scale * 0.3, l_knee[1] + 55)
    
    body_color = '#4A90E2'
    head_color = '#FFD700'
    right_color = '#E74C3C'
    left_color = '#2ECC71'
    w = 6
    
    spine_top = ((r_shoulder[0] + l_shoulder[0]) // 2, shoulder_y)
    draw.line([spine_top, (torso_x, hip_y)], fill=body_color, width=w)
    draw.line([r_shoulder, l_shoulder], fill=body_color, width=w)
    draw.line([r_hip, l_hip], fill=body_color, width=w)
    draw.line([spine_top, (head_x, head_y + 25)], fill=body_color, width=w)
    
    draw.ellipse([head_x-25, head_y-25, head_x+25, head_y+25], fill=head_color)
    
    draw.line([r_shoulder, r_hand], fill=right_color, width=w)
    draw.line([r_hip, r_knee], fill=right_color, width=w)
    draw.line([r_knee, r_foot], fill=right_color, width=w)
    
    draw.line([l_shoulder, l_hand], fill=left_color, width=w)
    draw.line([l_hip, l_knee], fill=left_color, width=w)
    draw.line([l_knee, l_foot], fill=left_color, width=w)
    
    for j in [r_shoulder, l_shoulder, r_hand, l_hand, r_hip, l_hip, r_knee, l_knee]:
        draw.ellipse([j[0]-6, j[1]-6, j[0]+6, j[1]+6], fill='white')
    
    return img


def render_pose(pose: Dict) -> Image.Image:
    img = visualize_pose_api(pose)
    if img is not None:
        return img
    return render_pose_builtin(pose)


def add_text(img: Image.Image, text: str, frame_num: int) -> Image.Image:
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), f"#{frame_num}", fill='#4ecca3')
    text = text[:42] + "..." if len(text) > 45 else text
    draw.text((10, img.height - 30), text, fill='#eeeeee')
    return img


def create_gif(poses: List[Dict], output_path: str, duration: int = 400, show_text: bool = True) -> str:
    frames = []
    
    api_available = check_pose_api()
    if api_available:
        print("API доступен")
    else:
        print("API недоступен")
    
    for i, pose_data in enumerate(poses):
        frame = render_pose(pose_data["pose"])
        
        if show_text:
            frame = add_text(frame, pose_data["description"], i + 1)
        
        frames.append(frame)
        print(f"   Кадр {i+1}: {pose_data['description'][:100]}...")
    
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
    print(f"Найдено {len(found)} поз")
    
    found = sort_poses_with_llm(found, query)
    
    for i, p in enumerate(found):
        print(f"   {i+1}. {p['description'][:55]}...")

    gif_path = create_gif(found, output_path)
    
    return gif_path


if __name__ == "__main__":
    import sys
    
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "какой-то танец"
    
    generate_animation(
        query=query,
        poses_db_path="poses_database.json",
        output_path=f"{query}.gif"
    )