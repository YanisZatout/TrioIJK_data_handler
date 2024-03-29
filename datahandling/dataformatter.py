from typing import List, Tuple


def format_model(models: List[str]) -> Tuple[List[str], List[str], List[str]]:
    """
    Formatting function for mixed model string
    models: str
        String with each LES model inside it
    Returns:
    model_names: str
        String with model names
    """
    models = [m.split("/")[-1] if not m.split("/")[-1] == ""
              else m.split("/")[-2] for m in models]

    model_names = [m.split("_")[0] for m in models]
    model_names = [m.replace("A", "AMD") for m in model_names]
    model_names = [m.replace("c", " comp ") for m in model_names]
    model_names = [m.replace("s", " scal ") for m in model_names]

    discretisation_qdm = [m.split("_")[1] for m in models]
    discretisation_mass = [m.split("_")[2] for m in models]
    return model_names, discretisation_qdm, discretisation_mass
