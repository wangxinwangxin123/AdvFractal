import os
import cv2
import math
import numpy as np
import random
from pathlib import Path
import detect_single_image
def rotate_point(cx, cy, x, y, angle_deg):
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    x -= cx
    y -= cy
    x_new = x * cos_a - y * sin_a
    y_new = x * sin_a + y * cos_a
    return int(x_new + cx), int(y_new + cy)
def draw_h_fractal_param(img, center, size, depth, angle_list, level=0, thickness=None):
    if thickness is None:
        raise ValueError("thickness ")
    if depth == 0 or thickness < 1:
        return
    x, y = center
    height = size
    width = int(size * 1.5)
    half_h = height // 2
    half_w = width // 2
    angle = angle_list if isinstance(angle_list, (int, float)) else angle_list[level] if level < len(angle_list) else 0
    tl = (x - half_w, y - half_h)
    tr = (x + half_w, y - half_h)
    bl = (x - half_w, y + half_h)
    br = (x + half_w, y + half_h)
    ml = (x - half_w, y)
    mr = (x + half_w, y)
    tl = rotate_point(x, y, *tl, angle)
    tr = rotate_point(x, y, *tr, angle)
    bl = rotate_point(x, y, *bl, angle)
    br = rotate_point(x, y, *br, angle)
    ml = rotate_point(x, y, *ml, angle)
    mr = rotate_point(x, y, *mr, angle)
    cv2.line(img, tl, bl, (0, 0, 0), int(thickness))
    cv2.line(img, tr, br, (0, 0, 0), int(thickness))
    cv2.line(img, ml, mr, (0, 0, 0), int(thickness))
    for pt in [tl, tr, bl, br]:
        draw_h_fractal_param(img, pt, size // 2, depth - 1, angle_list, level + 1, max(1, thickness // 2))


def decode(ind, box_width, box_height, x_offset, y_offset):
    cx = int(ind[0] * box_width) + x_offset
    cy = int(ind[1] * box_height) + y_offset
    size = int(ind[2] * min(box_width, box_height))
    depth = int(ind[3])
    angles = ind[4:6]
    return (cx, cy, size, depth, angles)

def fitness(ind, full_img, box_width, box_height, x_offset, y_offset, img_name):
    cx, cy, size, depth, angles = decode(ind, box_width, box_height, x_offset, y_offset)
    attacked_img = full_img.copy()
    initial_thickness = max(1, int(box_width /12))
    draw_h_fractal_param(attacked_img, (cx, cy), size, depth, angles, thickness=initial_thickness)
    results = detect_single_image.yolof_inf(attacked_img)
    scores = [box[4] for box in results[0]]
    save_path = os.path.join(r"/results/success3", img_name + '_success.jpg')
    cv2.imwrite(save_path, attacked_img)
    if not scores:
        save_path = os.path.join(r"D:/project/ronghe/results/success2", img_name + '_success.jpg')
        cv2.imwrite(save_path, attacked_img)
        return 0
    avg_conf = sum(scores) / len(scores)
    return avg_conf

def initialize_particles(pop_size):
    particles = []
    velocities = []
    for _ in range(pop_size):
        particle = [
            random.uniform(0.3, 0.6),      # x
            random.uniform(0.3, 0.6),      # y
            0.4,      # size
            3,          # depth: 限制最多3层
        ] + [random.uniform(-90, 90) for _ in range(3)]  # 3个角度值
        velocity = [random.uniform(-0.05, 0.05) for _ in range(len(particle))]
        particles.append(particle)
        velocities.append(velocity)
    return particles, velocities



if __name__ == '__main__':
    pop_size = 40
    generation = 50
    w, c1, c2 = 0.7, 1.5, 1.5
    state = {'ASR': 0, 'count': 0, 'count1': 0, 'success_num': 0}
    image_dir = Path(r"D:\project\ronghe\val_people_2")
    images = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.jpeg")) + list(image_dir.glob("*.png"))
    for img_path in images:
        print(f"\n>>> 正常处理图像 {img_path.name}")
        img = cv2.imread(str(img_path))
        results = detect_single_image.yolof_inf(img)
        if not results or len(results[0]) == 0 or len(results[0]) > 1:
            print("跳过（无目标或多目标）")
            continue
        state['count1'] += 1
        x1, y1, x2, y2 = map(int, results[0][0][:4])
        box_width = x2 - x1
        box_height = y2 - y1
        particles, velocities = initialize_particles(pop_size)
        pbest_positions = particles.copy()
        pbest_scores = [float('inf')] * pop_size
        gbest_position = None
        gbest_score = float('inf')
        for gen in range(generation):
            print(f"\n 第 {gen + 1} 代开始 ===================")
            success = False
            for i, particle in enumerate(particles):
                score = fitness(particle, img, box_width, box_height, x1, y1, img_path.name)
                state['count'] += 1
                if score < pbest_scores[i]:
                    pbest_scores[i] = score
                    pbest_positions[i] = particle.copy()
                if score < gbest_score:
                    gbest_score = score
                    gbest_position = particle.copy()
                print(f"  粒子 {i + 1}: 适应度 = {score:.4f}, 参数 = {particle}")
                if score == 0:
                    state['ASR'] += 1
                    success = True
                    print(f"攻击成功: {img_path.name}")
                    break
            if success:
                break
            else:
                print(f"图像 {img_path.name} 攻击未成功，当前最优置信度 = {gbest_score:.4f}")
            for i in range(pop_size):
                for d in range(len(particles[i])):
                    r1, r2 = random.random(), random.random()
                    velocities[i][d] = (
                        w * velocities[i][d] +
                        c1 * r1 * (pbest_positions[i][d] - particles[i][d]) +
                        c2 * r2 * (gbest_position[d] - particles[i][d])
                    )
                    particles[i][d] += velocities[i][d]
                    if d == 0 or d == 1:
                        particles[i][d] = max(0, min(1, particles[i][d]))
                    elif d == 2:
                        particles[i][d] = 0.4
                    elif d == 3:
                        particles[i][d] = 3
                    else:
                        particles[i][d] = max(-90, min(90, particles[i][d]))
    print("\n===== 总体攻击结果统计 =====")
    print(f"攻击成功率: {state['ASR'] / state['count1'] if state['count'] else 0:.4f}")
    print(f"平均查询次数: {state['count'] / state['count1'] if state['count1'] else 0:.2f}")
    print(f"总攻击图片数: {state['count1']}")
    print(f"攻击成功图片数: {state['ASR']}")