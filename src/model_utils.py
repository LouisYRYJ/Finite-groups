# Any util functions that import from models.py should be defined here, not in utils.py
from utils import *
from model import *

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
        params = Parameters(**json.loads(json_str))

    for root, dirs, files in os.walk(path + "/ckpts"):
        for filename in sorted(files, key=get_number_from_filename)[1:]:
            model_paths.append(os.path.join(root, filename))

    if (isinstance(sel, str) and sel.lower() == 'final') or len(model_paths) == 0:
        model_paths = [os.path.join(root, 'final.pt')]
    elif sel is not None:
        model_paths = model_paths[sel]

    return model_paths, params

def load_models(path, sel=None):
    model_paths, params = load_model_paths(path, sel=sel)
    models = []
    itr = model_paths if len(model_paths) < 5 else tqdm(model_paths)
    for model_path in itr:
        model = MODEL_DICT[params.model](params=params)
        model.load_state_dict(t.load(model_path))
        models.append(model)
    return models, params

def load_models_itr(path, sel=None):
    model_paths, params = load_model_paths(path, sel=sel)
    N = len(string_to_groups(params.group_string)[0])
    for model_path in model_paths:
        model = MODEL_DICT[params.model](params=params)
        model.load_state_dict(t.load(model_path))
        yield model