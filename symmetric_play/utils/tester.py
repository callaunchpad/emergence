from symmetric_play.utils.loader import load_from_name

def test(name, num_timesteps):
    model, env = load_from_name(name, load_env=True)

    images = []
    obs = env.reset()
    img = env.render(mode='rgb_array')
    for i in range(num_timesteps):
        images.append(img)
        action, _ = model.predict(obs)
        obs, _, _ ,_ = env.step(action)
        img = env.render(mode='rgb_array')
    env.close()