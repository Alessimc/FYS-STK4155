def format_param_grid(param_grid):
    return "_".join(f"{key}_{'_'.join(map(str, values))}" for key, values in param_grid.items())