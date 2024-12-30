# Any util functions that import from models.py should be defined here, not in utils.py
from utils import *
from model import *
import copy
import inspect

@jaxtyped(typechecker=beartype)
@t.no_grad()
def model_table(model: InstancedModule) -> Int[t.Tensor, 'instance N N']:
    """Returns the model's current multiplication table"""
    inputs = t.tensor(list(product(range(model.N), repeat=2)), dtype=int).to(device)
    model.eval()
    logits = model(inputs)  # shape N^2 x instance x N
    max_prob_entry = t.argmax(logits, dim=-1)  # shape N^2 x instance
    return einops.rearrange(max_prob_entry, " (n m) instance -> instance n m", n=model.N)  # shape instance x N x N

def get_number_from_filename(filename):
    match = re.search(r"(\d+)", filename)
    if match:
        return int(match.group(1))
    return -1

def load_model_paths(path, sel=None):
    from train import Parameters
    model_paths = []

    with open(path + "/params.json", "r") as f:
        json_str = f.read()
        params = json.loads(json_str)
        sig = inspect.signature(Parameters)
        for key in list(params.keys()):
            if key not in sig.parameters:
                print('WARNING: Removing unknown training parameter', key)
                del params[key]
        params = Parameters(**params)

    for root, dirs, files in os.walk(path + "/ckpts"):
        for filename in sorted(files, key=get_number_from_filename):
            if 'final' not in filename:
                model_paths.append(os.path.join(root, filename))

    if (isinstance(sel, str) and sel.lower() == 'final') or len(model_paths) == 0:
        model_paths = [os.path.join(root, 'final.pt')]
    elif sel is not None:
        model_paths = model_paths[sel]
        if isinstance(sel, int):
            model_paths = [model_paths]

    return model_paths, params

def load_models(path, sel=None):
    model_paths, params = load_model_paths(path, sel=sel)
    models = []
    itr = model_paths if len(model_paths) < 5 else tqdm(model_paths)
    for model_path in itr:
        model = MODEL_DICT[params.model](params=params)
        model.load_state_dict(t.load(model_path, map_location=device))
        models.append(model)
    return models, params

def load_models_itr(path, sel=None):
    model_paths, params = load_model_paths(path, sel=sel)
    N = len(string_to_groups(params.group_string)[0])
    for model_path in model_paths:
        model = MODEL_DICT[params.model](params=params)
        model.load_state_dict(t.load(model_path, map_location=device))
        yield model

def dl_model(name, model_dir=os.getcwd() + '/models'):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_dir = model_dir + '/' + name
    if not os.path.exists(model_dir):
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=f'wiwu2390/{name}', local_dir=model_dir)
    return load_models(model_dir)

def ablate_loss(model, data, ln, rn, un):
    ablate_model = copy.deepcopy(model)
    ablate_model.linear.data = t.eye(ln.shape[1]).unsqueeze(0)
    ablate_model.embedding_left.data = ln.unsqueeze(0)
    ablate_model.embedding_right.data = rn.unsqueeze(0)
    ablate_model.unembedding.data = un.unsqueeze(0).mT
    return test_loss(ablate_model.to(device), data)

def ablate_idx_loss(model, idxs):
    ln, rn= model.get_neurons()
    un = model.unembedding.data.detach()
    ln, rn, un = ln.squeeze(0).to('cpu'), rn.squeeze(0).to('cpu'), un.squeeze(0).to('cpu').T
    ln, rn, un = ln[:, idxs], rn[:, idxs], un[:, idxs]
    return ablate_loss(ln, rn, un)

def weight_norm(model):
    # Don't reduce along instance dimension
    ret = sum([t.norm(p, dim=tuple(range(1, len(p.shape)))) for p in model.parameters()]).squeeze(0)
    if ret.numel() == 1:
        return ret.item()
    return ret
    
