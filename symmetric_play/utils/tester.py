from symmetric_play.utils.loader import load_from_name
import numpy as np
import imageio

def test(name, num_timesteps, gif=False):
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
    file_path = 'output/' + name + '.gif'
    imageio.mimsave(file_path, [np.array(img) for i, img in enumerate(images) if i%2 == 0], fps=29, subrectangles=True)
    env.close()
