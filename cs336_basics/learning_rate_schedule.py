import math

def learning_rate_schedule(t, lr_max, lr_min, t_warm_up, t_cos_anneal):
    if t < t_warm_up:
        return t / t_warm_up * lr_max
    elif t >= t_warm_up and t <= t_cos_anneal:
        return lr_min + 0.5 * (1 + math.cos((t - t_warm_up) / (t_cos_anneal - t_warm_up) * math.pi)) * (lr_max - lr_min)
    else:
        return lr_min