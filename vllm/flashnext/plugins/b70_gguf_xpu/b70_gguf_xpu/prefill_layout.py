"""Retile GDN's last value axis without transposing the token matrix."""


def restore_value_axis(x, layout):
    k = int(x.shape[-1])
    width = int(layout.head_dim)
    hpg = int(layout.heads_per_group)
    heads, rem = divmod(k, width)
    groups, rem_g = divmod(heads, hpg)
    if rem or rem_g:
        raise ValueError(f"invalid GDN value axis K={k}, head_dim={width}, hpg={hpg}")
    return x.reshape(*x.shape[:-1], groups, hpg, width).transpose(-3, -2).reshape(x.shape).contiguous()
