from utils import *
from group import Group
from named_groups import *
from group_data import *

@jaxtyped(typechecker=beartype)
def string_to_groups(strings: Union[str, tuple[str, ...], list[str]]) -> list[Group]:
    """
    Input string s should be calls to above functions (returning NxN multiplication tables), delimited by ';'
    """
    if isinstance(strings, str):
        strings = strings.split(";")
    ret = []
    for s in strings:
        group = eval(s)
        if isinstance(group, Group):
            if group.name == "group":
                group.name = s
            ret.append(group)
        elif isinstance(group, tuple) or isinstance(group, list):
            for i, g in enumerate(group):
                if g.name == "group":
                    g.name = f"{s}_{i}"
            ret.extend(group)
        else:
            raise ValueError(f"Invalid group: {s}")
    return ret
