def one_hot_encode(_df, _col):
    _values = set(_df[_col].values)
    for v in _values:
        _df[v] = _df[_col].apply(lambda x : float(x == v) )
    return _df