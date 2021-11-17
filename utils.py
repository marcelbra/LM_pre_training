def _filter(x):
    if (not x
            or len(x.split(" ")) <= 6
            or x.startswith("Links")
            or x.startswith("Category")
            or x.startswith("List of")):
        return False
    return True

def _clean(x):
    return x

def _flatten(x):
    return [item for sentence in x for item in sentence]