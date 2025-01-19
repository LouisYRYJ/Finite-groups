from model import *
from utils import *
from llc import *
from tqdm.notebook import tqdm

# Implementations of measures described in "Fantastic Generalization Measures and Where to Find Them" by Jiang et al.
# Take (init_models, models, train_data) as input
# where init_models and modelsare InstancedModules with same number of instances
# and returns one scalar per instance

# Code adapted from https://github.com/nitarshan/robust-generalization-measures/blob/master/data/generation/measures.py

def train_acc(models, dataset):
    train_data = t.tensor(dataset.train_data)
    x = train_data[:,:2].to(device)
    z = train_data[:,2].to(device)
    logits = models(x)
    return get_accuracy(logits, z)

def perturbed_model(models, sigmas, rng, mag=False):
    perturbed = deepcopy(models)
    for p in perturbed.parameters():
        sigmas = sigmas.reshape((p.shape[0],) + (1,) * (p.dim() - 1)) # for broadcast to p.shape
        if mag:
            noise = t.normal(0., sigmas * p.abs(), generator=rng)
        else:
            noise = t.normal(0., sigmas, generator=rng)
        p.data += noise
    return perturbed

@t.no_grad()
def pacbayes_sigma(
    models, dataset, seed=777, search_depth=15, samples=30, target_delta=0.1, acc_tol=1e-2, mag=False, verbose=False
):
    rng = t.Generator(device=device)
    rng.manual_seed(seed)
    orig_accs = train_acc(models, dataset)

    # phase 1: doubling
    sigmas = t.ones(len(models), device=device)
    done = t.zeros_like(sigmas, dtype=t.bool)
    while not done.all():
        perturb_accs = []
        for _ in range(samples):
            perturbed = perturbed_model(models, sigmas, rng, mag=mag)
            perturb_accs.append(train_acc(perturbed, dataset))
            del perturbed
        perturb_accs = t.stack(perturb_accs, dim=0).mean(dim=0)
        done |= (perturb_accs - orig_accs).abs() > target_delta
        if done.all():
            break
        sigmas[~done] *= 2
        if verbose:
            print('PHASE 1 CONVERGED', done.sum())
    
    # phase 2: binary search
    lower = t.zeros_like(sigmas)
    done = t.zeros_like(sigmas, dtype=t.bool)
    upper = sigmas
    for _ in tqdm(range(search_depth)):
        mid = (lower + upper) / 2
        perturb_accs = []
        for __ in range(samples):
            perturbed = perturbed_model(models, mid, rng)
            perturb_accs.append(train_acc(perturbed, dataset))
            del perturbed
        perturb_accs = t.stack(perturb_accs, dim=0).mean(dim=0)
        delta = (perturb_accs - orig_accs).abs()
        done |= (delta - target_delta).abs() < acc_tol
        lower[~done] = t.where(delta[~done] < target_delta, mid[~done], lower[~done])
        upper[~done] = t.where(delta[~done] >= target_delta, mid[~done], upper[~done])
        if verbose:
            print('PHASE 2 CONVERGED', done.sum())
            print('MID', mid[:10])
            print('DELTA', delta[:10])
        if done.all():
            break
    if not done.all():
        print(f'WARN: Binary search with depth {search_depth} failed to converge for {(~done).sum()} instances!')
    return mid

@t.no_grad()
def path_norm(models):
    a1 = t.ones((1, len(models), models.N), device=device)
    a2 = t.ones_like(a1)
    x1 = einops.einsum(
        a1,
        models.embedding_left ** 2,
        "batch_size instances d_vocab, instances d_vocab embed_dim -> batch_size instances embed_dim",
    )
    x2 = einops.einsum(
        a2,
        models.embedding_right ** 2,
        "batch_size instances d_vocab, instances d_vocab embed_dim -> batch_size instances embed_dim",
    )
    if isinstance(models, MLP4):
        h1 = x1
        h2 = x2
    else:
        h1 = einops.einsum(
            x1,
            models.linear_left ** 2,
            "batch_size instances embed_dim, instances embed_dim hidden -> batch_size instances hidden",
        )
        h2= einops.einsum(
            x2,
            models.linear_right ** 2,
            "batch_size instances embed_dim, instances embed_dim hidden -> batch_size instances hidden",
        )
    out = einops.einsum(
        models.activation(h1 + h2),
        models.unembedding ** 2,
        "batch_size instances hidden, instances hidden d_vocab-> batch_size instances d_vocab ",
    )
    if models.unembed_bias is not None:
        # Not sure if this is the correct way to deal with bias
        out += einops.repeat(
            models.unembed_bias ** 2,
            'instances d_vocab -> batch_size instances d_vocab',
            batch_size=out.shape[0]
        )
    return out.sum(dim=(0, 2))

@t.no_grad()
def get_gen_measures(init_models, models, dataset):
    measures = dict()
    m = len(dataset.train_data)

    def get_weights(models):
        return [p.data.detach() for n, p in models.named_parameters() if 'bias' not in n]
    
    weights = get_weights(models)
    dist_init_weights = [p-q for p, q in zip(weights, get_weights(init_models))]
    d = len(weights)

    def get_vec_params(weights):
        return t.cat([w.flatten(start_dim=1) for w in weights], dim=1)

    w_vec = get_vec_params(weights)
    dist_w_vec = get_vec_params(dist_init_weights)
    num_params = w_vec.shape[1]
    
    measures['L2'] = w_vec.norm(dim=1, p=2)
    measures['L2_DIST'] = dist_w_vec.norm(dim=1, p=2)

    print('MARGIN')
    train_data = t.tensor(dataset.train_data)
    x = train_data[:,:2].to(device)
    z = train_data[:,2].to(device)
    logits = models(x)

    margin = get_margin(logits, z, quantile=0.1)
    measures['INVERSE_MARGIN'] = 1 / margin ** 2
    
    print('NORMS')
    fro_norms = t.stack([t.norm(w, p='fro', dim=(-2, -1)) ** 2 for w in weights], dim=0)  # d instances
    spec_norms = t.stack([t.linalg.matrix_norm(w, ord=2, dim=(-2, -1)) ** 2 for w in weights], dim=0)
    dist_fro_norms = t.stack([t.norm(w, p='fro', dim=(-2, -1)) ** 2 for w in dist_init_weights], dim=0)
    dist_spec_norms = t.stack([t.linalg.matrix_norm(w, ord=2, dim=(-2, -1)) ** 2 for w in dist_init_weights], dim=0)
    
    measures['LOG_PROD_OF_SPEC'] = spec_norms.log().sum(dim=0)
    measures['LOG_PROD_OF_SPEC_OVER_MARGIN'] = measures['LOG_PROD_OF_SPEC'] - 2 * margin.log()
    measures['LOG_SPEC_INIT_MAIN'] = measures['LOG_PROD_OF_SPEC_OVER_MARGIN'] + (dist_fro_norms / spec_norms).sum().log()
    measures['FRO_OVER_SPEC'] = (fro_norms / spec_norms).sum(dim=0)
    measures['LOG_SPEC_ORIG_MAIN'] = measures['LOG_PROD_OF_SPEC_OVER_MARGIN'] + measures['FRO_OVER_SPEC'].log()
    measures['LOG_SUM_OF_SPEC_OVER_MARGIN'] = math.log(d) + (1/d) * (measures['LOG_PROD_OF_SPEC'] -  2 * margin.log())
    measures['LOG_SUM_OF_SPEC'] = math.log(d) + (1/d) * measures['LOG_PROD_OF_SPEC']

    measures['LOG_PROD_OF_FRO'] = fro_norms.log().sum(dim=0)
    measures['LOG_PROD_OF_FRO_OVER_MARGIN'] = measures['LOG_PROD_OF_FRO'] -  2 * margin.log()
    measures['LOG_SUM_OF_FRO_OVER_MARGIN'] = math.log(d) + (1/d) * (measures['LOG_PROD_OF_FRO'] -  2 * margin.log())
    measures['LOG_SUM_OF_FRO'] = math.log(d) + (1/d) * measures['LOG_PROD_OF_FRO']

    measures['FRO_DIST'] = dist_fro_norms.sum(dim=0)
    measures['DIST_SPEC_INIT'] = dist_spec_norms.sum(dim=0)
    measures['PARAM_NORM'] = fro_norms.sum(dim=0)

    print('PATH_NORM')
    measures['PATH_NORM'] = path_norm(models)
    measures['PATH_NORM_OVER_MARGIN'] = measures['PATH_NORM'] / margin ** 2

    print('PACBAYES')
    sigma = pacbayes_sigma(models, dataset)
    def pacbayes_bound(vec, sigma):
        return (vec.norm(p=2, dim=tuple(range(1, len(vec.shape)))) ** 2) / (4 * sigma ** 2) + t.log(m / sigma) + 10
    measures['PACBAYES_INIT'] = pacbayes_bound(dist_w_vec, sigma)
    measures['PACBAYES_ORIG'] = pacbayes_bound(w_vec, sigma)
    measures['PACBAYES_FLATNESS'] = 1 / sigma ** 2

    print('PACBAYES_MAG')
    mag_sigma = pacbayes_sigma(models, dataset, mag=True)
    def pacbayes_mag_bound(vec, mag_sigma):
        numerator = (mag_sigma ** 2 + 1) * (vec.norm(p=2, dim=tuple(range(1, len(vec.shape))))**2) / num_params
        numerator = numerator.reshape((vec.shape[0],) + (1,) * (vec.dim() - 1))
        mag_sigma = mag_sigma.reshape((vec.shape[0],) + (1,) * (vec.dim() - 1)) # for broadcast to vec.shape
        denominator = mag_sigma ** 2 * vec ** 2  # Jiang et al paper has dist_w_vec instead of vec here. but probably a typo??
        return 1/4 * (numerator / denominator).log().sum(dim=1) + t.log(m / mag_sigma.squeeze(1)) + 10
    measures['PACBAYES_MAG_INIT'] = pacbayes_mag_bound(dist_w_vec, mag_sigma)
    measures['PACBAYES_MAG_ORIG'] = pacbayes_mag_bound(w_vec, mag_sigma)
    measures['PACBAYES_MAG_FLATNESS'] = 1 / mag_sigma ** 2

    def adjust_measure(name, value):
        # Not really clear on which measures should be normalized by m
        # Of course it doesn't actually matter; m is constant across all models
        if name == 'INVERSE_MARGIN':
            return value
        if name.startswith('LOG_'):
            return 0.5 * (value - np.log(m))
        else:
            return np.sqrt(value / m)
    return {k: adjust_measure(k, v) for k, v in measures.items()}
