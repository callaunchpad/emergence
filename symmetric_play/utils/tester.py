from symmetric_play.utils.loader import load_from_name
import numpy as np

def test(name, num_timesteps, gif=False):
    model, env = load_from_name(name, load_env=True)
    mode = 'rgb_array' if gif else 'human'
    images = []
    obs = env.reset()
    img = env.render(mode=mode)
    for i in range(num_timesteps):
        images.append(img)
        action, _ = model.predict(obs)
        obs, done, reward ,_ = env.step(action)
        img = env.render(mode=mode)
        if isinstance(done, list):
            done = np.array(done)
        if isinstance(done, np.ndarray):
            done = done.any()
        if done:
            obs = env.reset()
        

    env.close()