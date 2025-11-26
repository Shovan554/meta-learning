import numpy as np
import random
import time
import pygame
import json
from maze_env_rl import MazeEnv

env = MazeEnv("maps/maze1.txt", render_mode=True)  # create env
state_size = env.get_state().shape[0]  # state length
action_size = 5  # up, down, left, right, push

Q = {}  # q-table dict

def get_key():
    return tuple(env.rat_pos)  # state key based on rat position

def get_Q(key):
    if key not in Q:  # create new row if unseen
        Q[key] = np.zeros(action_size)  # zero-init actions
    return Q[key]  # return q-values for this state

alpha = 0.7  # learning rate
gamma = 0.9  # discount factor
epsilon = 0.3  # exploration rate
episodes = 500  # max episodes

results = []  # store stats
best_reward = -float('inf')  # track best reward
success_streak = 0  # consecutive efficient successes
STOP_AFTER_STREAK = 2  # stop after 2 in a row
BEST_STEP_THRESHOLD = 25  # max steps for "efficient"

for ep in range(episodes):
    state = env.reset()  # reset env
    total_reward = 0  # episode reward
    step_count = 0  # step counter
    push_count = 0  # push counter
    climb_count = 0  # climb counter
    success = False  # success flag

    for step in range(400):
        pygame.event.pump()  # keep pygame responsive
        env.render()  # draw frame
        time.sleep(0.03)  # slow for human eye

        key = get_key()  # current state key
        if random.random() < epsilon:  # epsilon-greedy
            action = random.randint(0, action_size - 1)  # random action
        else:
            action = np.argmax(get_Q(key))  # greedy action

        next_state, reward, done, climbed = env.step(action)  # env step
        next_key = get_key()  # next state key

        old_value = get_Q(key)[action]  # current Q(s,a)
        next_max = np.max(get_Q(next_key))  # max_a' Q(s',a')
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)  # q-update
        Q[key][action] = new_value  # store updated value

        total_reward += reward  # accumulate reward
        step_count += 1  # increment step

        if action == 4:
            push_count += 1  # count pushes
        if climbed:
            climb_count += 1  # count climbs

        if done:
            success = reward > 0  # success if final reward positive (reached cheese)
            break  # end episode

    # ✅ compute efficient ONCE per episode, after inner loop
    efficient = success and step_count <= BEST_STEP_THRESHOLD  # fast and successful

    results.append({
        "episode": ep + 1,
        "steps": step_count,
        "pushes": push_count,
        "climbs": climb_count,
        "reward": round(total_reward, 2),
        "success": success,
        "efficient": efficient
    })

    epsilon = max(0.10, epsilon * 0.998)  # decay epsilon but keep at least 0.10

    if efficient:  # only count efficient successes toward stopping
        success_streak += 1  # increase streak
    else:
        success_streak = 0  # reset streak

    if success_streak >= STOP_AFTER_STREAK:  # check stopping condition
        print(f"\n🎉 Agent mastered the maze efficiently! ({success_streak} fast successes in a row)")
        break  # stop training

    if total_reward > best_reward:
        best_reward = total_reward  # update best reward

    print(f"Episode {ep+1} | Reward: {total_reward:.2f} | Steps: {step_count} | "
          f"Pushes: {push_count} | Climbs: {climb_count} | "
          f"{'✅ Success' if success else '❌ Fail'} "
          f"{'⚡ Efficient' if efficient else ''} | Epsilon: {epsilon:.2f}")

env.close()  # close window

output = {
    "metadata": {
        "total_episodes": len(results),  # how many episodes actually ran
        "stop_condition": f"{STOP_AFTER_STREAK} consecutive efficient successes (≤{BEST_STEP_THRESHOLD} steps)",  # stop rule
        "best_reward": round(best_reward, 2)  # best observed reward
    },
    "results": results  # per-episode stats
}

with open("reinforcement_learning_agent_output.json", "w") as f:
    json.dump(output, f, indent=4)  # save json

print("\n✅ Training complete!")  # done
print("📁 Results saved as reinforcement_learning_agent_output.json")  # saved file
