from symmetric_play.utils.loader import load_from_name
import numpy as np
import imageio
import os.path

def test_orig(name, num_timesteps, gif=False):
    model, env = load_from_name(name, load_env=True)
    mode = 'rgb_array' if gif else 'human'
    images = []
    obs = env.reset()
    img = env.render(mode=mode)
    for i in range(num_timesteps):
        images.append(img)
        action, _ = model.predict(obs)
        obs, reward, done,_ = env.step(action)
        img = env.render(mode=mode)
        if isinstance(done, list):
            done = np.array(done)
        if isinstance(done, np.ndarray):
            done = done.any()
        if done:
            obs = env.reset()
            break

    if gif:
        id = 0
        while True:
            file_path = 'output/' + name + '/test_' + str(id) + '.gif'
            if not os.path.isfile(file_path):
                break
            id += 1

        print(file_path)
        imageio.mimsave(file_path, [np.array(img) for i, img in enumerate(images) if i%2 == 0], fps=29, subrectangles=True)
        env.close()

# Following 2 functions adapted from Joey's code and modified to work for multiple agents
# mention me in Slack [@Sumer] if broken
def eval_policy(model, env, num_ep=10, deterministic=True, verbose=1, gif=False, render=False):
    ep_rewards, ep_lens, ep_infos = list(), list(), list()
    mode = 'rgb_array' if gif else 'human'
    frames = list()
    for ep_index in range(num_ep):
        obs = env.reset()
        done = False
        ep_rew, ep_len = None, 0
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, info = env.step(action)
            ep_len += 1
            ep_rew = reward
            if 'frames' in info:
                frames.extend(info['frames'])
            if render:
                frames.append(env.render(mode=mode))
            if isinstance(done, list):
                done = np.array(done)
            if isinstance(done, np.ndarray):
                done = done.any()
        ep_rewards.append(ep_rew)
        ep_lens.append(ep_len)
        ep_infos.append(info)
        if verbose:
            print("Finished Episode", ep_index + 1, "Reward:", ep_rew, "Length:", ep_len)
    print("Completed Eval of", num_ep, "Episodes")
    print("Avg. Reward:", np.mean(ep_rewards), "Avg. Length", np.mean(ep_lens))
    return np.mean(ep_rewards), frames

def test(name, num_ep=10, deterministic=True, verbose=1, gif=False):
    model, env = load_from_name(name, load_env=True)
    _, frames = eval_policy(model, env, num_ep=num_ep, deterministic=deterministic, verbose=verbose, gif=gif, render=True)
    if gif:
        import imageio
        if name.endswith('/'):
            name = name[:-1]
        # if name.startswith(BASE):
        #     # Remove the base
        #     name = name[len(BASE):]
        #     if name.startswith('/'):
        #         name = name[1:]
        render_path = os.path.join('output/', name + '.gif')
        print("Saving gif to", render_path)
        os.makedirs(os.path.dirname(render_path), exist_ok=True)
        imageio.mimsave(render_path, frames[::4], subrectangles=True, duration=0.05)
    env.close()
    del model
