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


def key_equivalent(entry: str, sqrt: bool = False)->str:
    out = None
    if "rms" in entry:
        out = r"\langle " f"{entry[0].upper()}'^2" r"\rangle^{+,dev}"
    if "uv" in entry:
        out = r"\langle " f"{entry[0].upper()}'{entry[1].upper()}'" r"\rangle^+"
    if "theta" in entry:
        out = r"\langle " f"{entry[0].upper()}'" r"\theta'" r"\rangle^+"
    if "theta_rms" in entry:
        out = r"\langle " r"\theta'^2" r"\rangle^+"
    if "U" in entry or "V" in entry or "T" in entry:
        out = r"\langle " f"{entry[0].upper()}" r"\rangle^+"
    if "phi" in entry.lower() or "lambdadtdz" in entry.lower():
        out = r"\langle " r"\phi" r"\rangle"
    if "cf" in entry.lower():
        return r"Cf"
    if sqrt:
        out = r"\sqrt{" f"{out}" r"}"
    return "$" + str(out) + "$"
