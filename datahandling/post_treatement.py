from typing import List, Tuple
import pandas as pd


def parse_formula(formula: str) -> List[str]:
    """
    Parses columns from formula
    """
    out = formula.replace("\n", "")
    out = out.split("column")
    out = [o[o.find("(")+1:o.find(")")] for o in out]
    for i, o in enumerate(out):
        try:
            int(o)
        except ValueError:
            out.pop(i)
    out = [o for o in out if o != ""]
    return out


def replace_columns_in_formula(
    formula: str, df: pd.DataFrame, name: str = "df"
) -> str:
    """
    Replaces column number by column name
    """
    form = parse_formula(formula)
    form.sort(key=lambda x: int(x), reverse=True)
    for f in form:
        formula = formula.replace(
            f"column({f})", f'{name}["{df.columns[int(f)-1]}"]')
    formula = formula.replace("word", "")
    return formula


def format_string_formula(formula: str, df: pd.DataFrame) -> str:
    """
    Formula printer function
    """
    form = parse_formula(formula)
    form.sort(key=lambda x: int(x), reverse=True)
    for f in form:
        formula = formula.replace(f"column({f})", f'{df.columns[int(f)-1]}')
    formula = formula.replace("word", "")
    return formula


def need_utau_or_ttau(formula: str) -> Tuple[bool, bool]:
    utau, ttau = False, False
    if "UTAUS_c" in formula or "UTAUS_f" in formula:
        utau = True
        if ("TTAUS_c" in formula) or \
            ("TTAUS_f" in formula) or \
            ("TTAUS_ch" in formula) or \
                ("TTAUS_fr" in formula):
            ttau = True
    if ("TTAUS_c" in formula) or \
        ("TTAUS_f" in formula) or \
        ("TTAUS_ch" in formula) or \
            ("TTAUS_fr" in formula):
        ttau = True
    return utau, ttau


def format_normalizing_constant(
    formula: str,
    df: pd.DataFrame,
    df_name: str = "df",
    hot_cold: str = str(None),
    les_dns: str = str(None)
) -> str:
    return replace_columns_in_formula(
        formula, df, df_name
    ).replace(
        'UTAUS_c,i', f'utau_{hot_cold}_{les_dns}'
    ).replace(
        'UTAUS_ch,i', f'utau_{hot_cold}_{les_dns}'
    ).replace(
        'TTAU_c,i', f'ttau_{hot_cold}_{les_dns}'
    ).replace(
        'UTAUS_f,i', f'utau_{hot_cold}_{les_dns}'
    ).replace(
        'TTAU_h,i', f'ttau_{hot_cold}_{les_dns}'
    ).replace(
        'TTAUS_ch,i', f'ttau_{hot_cold}_{les_dns}'
    ).replace(
        'TTAUS_fr,i', f'ttau_{hot_cold}_{les_dns}'
    ).replace("\n", "")


def format_denominator_les(
    formula: str,
    df: pd.DataFrame,
    df_name: str = "df",
) -> str:
    return replace_columns_in_formula(
        formula, df, df_name
    ).replace(
        'UTAUS_c,i', 'utau'
    ).replace(
        'UTAUS_ch,i', 'utau'
    ).replace(
        'TTAU_c,i', 'ttau'
    ).replace(
        'UTAUS_f,i', 'utau'
    ).replace(
        'TTAU_h,i', 'ttau'
    ).replace(
        'TTAUS_ch,i', 'ttau'
    ).replace(
        'TTAUS_fr,i', 'ttau'
    ).replace("\n", "")


def generate_formulas(
    formula: str,
    df_dns: pd.DataFrame,
    name_les: str = "df_les",
    name_quantity: str = str(None),
    print_formula: bool = True
) -> str:
    """
    Formula generating function
    """
    if print_formula:
        print(format_string_formula(formula, df_dns))
    general_purpose_formula = f"""
{name_quantity}_cold_les = [{
format_denominator_les(formula, df_dns, 'df')
}.values[:halfes] for df, utau, ttau, halfes in zip(
    {name_les}, utau_cold_les, ttau_cold_les, half_les
)]
{name_quantity}_cold_dns = {
    format_normalizing_constant(formula, df_dns, 'df_dns', 'cold', 'dns')
}.values[:half]

{name_quantity}_hot_les = [{
    format_denominator_les(formula, df_dns, 'df')
}.values[-halfes:] for df, utau, ttau, halfes in zip(
    {name_les}, utau_hot_les, ttau_hot_les, half_les
)]
{name_quantity}_hot_dns = {
    format_normalizing_constant(formula, df_dns, 'df', 'hot', 'dns')
}.values[-half:]
    """
    return general_purpose_formula


# formula = """
# ((column(10)-column(2)*column(4)-column(84)/column(17)-(column(72)+column(76)))/(word(UTAUS_ch,i))**2)
# """
# print(generate_formulas(formula, df_dns, name_les="df_les", name_quantity="uv_plus"))
# # /(word(UTAUS_ch,i))**2
# # print(replace_columns_in_formula(formula, df_dns))
# # print(format_string_formula(formula, df_dns))
# formula = '((sqrt(column(28)-column(18)*column(18)))/(word(TTAUS_ch,i)*word(TTAUS_ch,i)))'
# # print(generate_formulas(formula, df_dns, name_les="df_les", name_quantity="t_rms_plus"))
# o = generate_formulas(formula, df_dns, name_les="df_les",
#                       name_quantity="t_rms_plus")
# print(o)
#
#
# def test_formulas(formula):
#     # formula = """
#     # ((column(10)-column(2)*column(4)-column(84)/column(17)-(column(72)+column(76)))/(word(UTAUS_ch,i))**2)
#     # """
#     # formula = '((sqrt(column(28)-column(18)*column(18)))/(word(TTAUS_ch,i)*word(TTAUS_ch,i)))'
#     print(generate_formulas(formula, df_dns, name_les="df_les",
#           name_dns="df_dns", name_quantity="t_rms_plus"))
