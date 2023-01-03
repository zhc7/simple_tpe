from typing import *

import numpy as np


def plot(lx, gx, a, b):
    import matplotlib.pyplot as plt
    X = np.linspace(a, b, 100)
    X.sort()
    Y1 = pdf(X, *gx)
    Y2 = pdf(X, *lx)
    Y = Y2 / Y1
    Y1 /= Y1.sum()
    Y2 /= Y2.sum()
    Y /= Y.sum()
    plt.plot(X, Y1, color="red", label='g')
    plt.plot(X, Y2, color="blue", label='l')
    plt.plot(X, Y, color="yellow", label='l/g')
    plt.legend()
    plt.show()


def posterior(obs, prior_mu, prior_sigma):
    """:return posterior distribution of l(x) or g(x) under TPE model"""
    mus = np.array(obs)
    order = np.argsort(mus)
    prior_pos = np.searchsorted(mus[order], prior_mu)
    sorted_mus = np.zeros(len(mus) + 1)
    sorted_mus[:prior_pos] = mus[order[:prior_pos]]
    sorted_mus[prior_pos] = prior_mu
    sorted_mus[prior_pos + 1:] = mus[order[prior_pos:]]
    sigma = np.zeros_like(sorted_mus)
    sigma[1:-1] = np.maximum(sorted_mus[1:-1] - sorted_mus[0:-2], sorted_mus[2:] - sorted_mus[1:-1])
    sigma[0] = sorted_mus[1] - sorted_mus[0]
    sigma[-1] = sorted_mus[-1] - sorted_mus[-2]

    max_sigma = prior_sigma
    min_sigma = prior_sigma / (len(sorted_mus) + 1)

    sigma = np.clip(sigma, min_sigma, max_sigma)
    return sorted_mus, sigma


def draw(mus, sigmas, n):
    """:return n samples"""
    active = np.argmax(np.random.multinomial(1, [1 / len(mus)] * len(mus), n), axis=1)
    sample = np.random.normal(mus[active], sigmas[active])
    return sample


def pdf(samples, mus, sigmas):
    """return log function value of l(x) or g(x), which is composition of Gaussian dists"""
    dist = samples[:, None] - mus
    exps = np.exp(- dist ** 2 / 2 / sigmas ** 2)
    scales = 1 / len(mus) / np.sqrt(2 * np.pi * sigmas ** 2)
    return np.sum(scales * exps, axis=1)


def search(history: List[Tuple[Dict[str, Union[float, int, str]], float]], space: Dict[str, Union[tuple, list]],
           min_size=12, gamma=0.2, n_samples=100, pure_random=False) -> Dict[str, Union[float, int]]:
    """TPE algorithm"""
    if len(history) < min_size or pure_random:
        ret = {}
        for key in space.keys():
            if type(space[key]) is tuple:
                a, b = space[key]
                ret[key] = a + (b - a) * np.random.random()
            else:
                ret[key] = np.random.choice(space[key])
        return ret

    ret = {}
    # type def: 0 = uniform choice, 1 = uniform
    seperated_history: dict[str: Tuple[list, list, int]] = \
        {key: ([], [], int(type(space[key]) == tuple)) for key in space}
    for h in history:
        hyp, loss = h
        for key in hyp.keys():
            seperated_history[key][0].append(hyp[key])
            seperated_history[key][1].append(loss)

    for hyp in seperated_history:
        x, loss, typ = seperated_history[hyp]
        if typ == 0:
            # paper: proportional to N*p_i + C_i, here we assume p_i = 1 / N
            C = {key: 1 for key in space[hyp]}
            for o in x:
                C[o] += 1
            c_list = []
            for key in space[hyp]:
                c_list.append(C[key])
            c_list = np.array(c_list, dtype=np.float32)
            c_list /= np.sum(c_list)
            x_place = np.argmax(np.random.multinomial(1, c_list))
            x_star = space[hyp][x_place]
        else:
            y_star = np.percentile(loss, gamma * 100)
            obs_lower = []
            obs_greater = []
            for i, y in enumerate(loss):
                if y <= y_star:
                    obs_lower.append(x[i])
                else:
                    obs_greater.append(x[i])
            a, b = space[hyp]
            assert b > a
            prior_mu = (a + b) / 2
            prior_sigma = (b - a) / 2
            lx = posterior(obs_lower, prior_mu, prior_sigma)
            gx = posterior(obs_greater, prior_mu, prior_sigma)
            candidates = draw(lx[0], lx[1], n_samples)
            candidates = np.clip(candidates, *space[hyp])
            x_place = np.argmin(pdf(candidates, *gx) / pdf(candidates, *lx))
            x_star = candidates[x_place]
        ret[hyp] = x_star

    return ret


def optimize(fn, n, *args, **kwargs):
    history = []
    best = (None, np.inf)
    out_size = n / 20
    for i in range(n):
        x = search(history, *args, **kwargs)
        loss = fn(**x)
        history.append((x, loss))
        if loss < best[1]:
            best = (x, loss)
        if (i+1) % out_size == 0:
            print(best)
    return best


if __name__ == '__main__':
    def target(x):
        return (x - 3) ** 2 + 2


    print(optimize(target, 30, space={'x': (-10, 10)}, pure_random=False))
