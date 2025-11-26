#meta_learning_train_phase

import numpy as np
import random
import time
import pygame
import json
import os
from maze_env_rl import MazeEnv


MAP_DIR = "maps"
META_TRAIN_MAPS = [f for f in os.listdir(MAP_DIR) if f.startswith("meta_maze_")]
META_TRAIN_MAPS.sort()
TEST_MAP = "maze1.txt"   # held-out test maze

# Hyperparameters
alpha = 0.7
gamma = 0.9
episodes = 500
STOP_AFTER_STREAK = 2
BEST_STEP_THRESHOLD = 25
action_size = 5


def get_Q(Q, key):
    if key not in Q:
        Q[key] = np.zeros(action_size)
    return Q[key]


def get_key(env):
    return tuple(env.rat_pos)


# ----------------------------------------------------
# PHASE 1: META-TRAINING — learn shared Q across 30 maps
# ----------------------------------------------------
def meta_train_on_maps(meta_maps, epsilon=0.3):
    shared_Q = {}
    meta_results = []

    print(f"\n🧠 Starting meta-training on {len(meta_maps)} mazes...\n")

    for idx, map_file in enumerate(meta_maps):
        map_path = os.path.join(MAP_DIR, map_file)
        env = MazeEnv(map_path, render_mode=False)

        success_streak = 0
        best_reward = -float("inf")
        start_time = time.time()

        print(f"🌍 Training on task {idx+1}/{len(meta_maps)} → {map_file}")

        for ep in range(episodes):
            state = env.reset()
            total_reward = 0
            step_count = 0
            push_count = 0
            climb_count = 0
            success = False

            for step in range(400):
                key = get_key(env)
                if random.random() < epsilon:
                    action = random.randint(0, action_size - 1)
                else:
                    action = np.argmax(get_Q(shared_Q, key))

                next_state, reward, done, climbed = env.step(action)
                next_key = get_key(env)

                old_value = get_Q(shared_Q, key)[action]
                next_max = np.max(get_Q(shared_Q, next_key))
                shared_Q[key][action] = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)

                total_reward += reward
                step_count += 1
                if action == 4:
                    push_count += 1
                if climbed:
                    climb_count += 1
                if done:
                    success = reward > 0
                    break

            efficient = success and step_count <= BEST_STEP_THRESHOLD
            if efficient:
                success_streak += 1
            else:
                success_streak = 0

            if success_streak >= STOP_AFTER_STREAK:
                print(f"✅ {map_file} mastered efficiently in {ep+1} episodes.")
                break

            if total_reward > best_reward:
                best_reward = total_reward

        duration = round(time.time() - start_time, 2)
        meta_results.append({
            "map": map_file,
            "episodes": ep + 1,
            "best_reward": round(best_reward, 2),
            "time": duration
        })
        env.close()

        epsilon = max(0.20, epsilon * 0.9)  # gradually reduce exploration

    print("\n✅ Meta-training complete across all maps!\n")
    return shared_Q, meta_results, epsilon


# ----------------------------------------------------
# PHASE 2: META-TEST — adapt quickly on held-out maze
# ----------------------------------------------------
def meta_test_on_maze(test_map, meta_Q, epsilon):
    print(f"🚀 Meta-testing on held-out maze → {test_map}\n")

    env = MazeEnv(os.path.join(MAP_DIR, test_map), render_mode=True)
    Q = {k: np.copy(v) for k, v in meta_Q.items()}
    results = []
    best_reward = -float("inf")
    success_streak = 0

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        step_count = 0
        push_count = 0
        climb_count = 0
        success = False

        for step in range(400):
            pygame.event.pump()
            env.render()
            time.sleep(0.03)

            key = get_key(env)
            if random.random() < epsilon:
                action = random.randint(0, action_size - 1)
            else:
                action = np.argmax(get_Q(Q, key))

            next_state, reward, done, climbed = env.step(action)
            next_key = get_key(env)

            old_value = get_Q(Q, key)[action]
            next_max = np.max(get_Q(Q, next_key))
            Q[key][action] = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)

            total_reward += reward
            step_count += 1
            if action == 4:
                push_count += 1
            if climbed:
                climb_count += 1
            if done:
                success = reward > 0
                break

        efficient = success and step_count <= BEST_STEP_THRESHOLD
        results.append({
            "episode": ep + 1,
            "steps": step_count,
            "pushes": push_count,
            "climbs": climb_count,
            "reward": round(total_reward, 2),
            "success": success,
            "efficient": efficient
        })

        epsilon = max(0.10, epsilon * 0.998)
        if efficient:
            success_streak += 1
        else:
            success_streak = 0
        if success_streak >= STOP_AFTER_STREAK:
            print(f"🎉 Meta-agent mastered {test_map} efficiently in {ep+1} episodes!")
            break

        if total_reward > best_reward:
            best_reward = total_reward

        print(f"Episode {ep+1} | Reward: {total_reward:.2f} | Steps: {step_count} | "
              f"Pushes: {push_count} | Climbs: {climb_count} | "
              f"{'✅ Success' if success else '❌ Fail'} "
              f"{'⚡ Efficient' if efficient else ''} | Epsilon: {epsilon:.2f}")

    env.close()
    return results


# ----------------------------------------------------
# MAIN EXECUTION
# ----------------------------------------------------
if __name__ == "__main__":
    # PHASE 1 — Meta-training on 30 random mazes
    meta_Q, meta_results, epsilon = meta_train_on_maps(META_TRAIN_MAPS, epsilon=0.3)

    with open("meta_training_phase.json", "w") as f:
        json.dump(meta_results, f, indent=4)

    print("✅ Meta-training results saved → meta_training_phase.json")

    # PHASE 2 — Meta-testing on held-out maze1.txt
    test_results = meta_test_on_maze(TEST_MAP, meta_Q, epsilon)

    output = {
        "meta_training_summary": meta_results,
        "meta_testing_results": test_results
    }

    with open("meta_learning_final_output.json", "w") as f:
        json.dump(output, f, indent=4)

    print("\n✅ Full meta-learning experiment finished!")
    print("📁 Results saved → meta_learning_final_output.json")
