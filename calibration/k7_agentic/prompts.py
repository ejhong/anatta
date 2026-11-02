"""K7: assemble the agentic-misalignment prompts (Lynch et al. 2025) with our own code. Claude (Opus 5.5) wrote this
file, 2 Nov 2026.

The scenario text never enters the repo. It is read at run time from the lab's held copy of
anthropic-experimental/agentic-misalignment (ea0630e1, MIT; lab/data/agentic_misalignment/raw, gitignored). The
upstream system-prompt module is a Python file; we never import or execute it: `ast` reads its string literals, its
GOAL_VARIABLES dict (literal_eval) and the one list comprehension per scenario (evaluated here for string
concatenation only). The assembly follows scripts/generate_prompts.py (read, not run):

  system  = the scenario's dev system template (with the scratchpad paragraph; production_variant false, as in the
            multi-model configs), with goal_statement / goal_reminder from GOAL_VARIABLES["{goal_type}-{goal_value}"],
            extra_system_instructions = "" and alignment_faking_warning = "", filled by string.Template.safe_substitute
  emails  = goal_type "none": the goal-conflict-none-{urgency} template; else shared + {urgency} template, each filled
            with the variables file's matching section (america_vs_global.md, section "Pro-America" for america).
            For leaking, every "<email>" gets "\\nID: {i}" after it, i from 1.
  user    = USER_PROMPT + "\\n" + emails. The upstream runner (not held) joins the instruction and the emails at run
            time; the joiner is our assumption (whitespace only).
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path
from string import Template

RAW = Path("/Users/eugene/prj/align/anatta/lab/data/agentic_misalignment/raw")
TEMPLATES = RAW / "templates"
SYS_MODULE = TEMPLATES / "system_prompt_templates.py"
SCENARIO_SYS = {"blackmail": "BLACKMAIL_SYSTEM_PROMPT", "leaking": "LEAKING_SYSTEM_PROMPT",
                "murder": "MURDER_SYSTEM_PROMPT"}
JOINER = "\n"

# The two registered cells (configs/multi_model_experiment_250616.yaml: goal_values ["america"]).
CELLS = {
    "espionage_explicit_none": ("leaking", "explicit", "america", "none"),
    "blackmail_explicit_replacement": ("blackmail", "explicit", "america", "replacement"),
}


# ------------------------------------------------------------------------------------- the system-prompt module


def _eval_str(node: ast.AST, env: dict) -> str:
    """String literals, names bound in env, and + between them. Anything else is refused."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name) and node.id in env:
        return env[node.id]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _eval_str(node.left, env) + _eval_str(node.right, env)
    raise ValueError(f"unsupported node in a template: {ast.dump(node)[:120]}")


def read_system_module(path: Path = SYS_MODULE) -> dict:
    """{"strings": {NAME: str}, "goal_variables": {...}, "system": {scenario: (dev, prod)}} from the module's AST."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    strings: dict[str, str] = {}
    goal_vars = None
    comps: dict[str, ast.ListComp] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        tgt = node.targets[0]
        if isinstance(tgt, ast.Name):
            if tgt.id == "GOAL_VARIABLES":
                goal_vars = ast.literal_eval(node.value)
            else:
                try:
                    strings[tgt.id] = _eval_str(node.value, strings)
                except ValueError:
                    pass
        elif isinstance(tgt, ast.Tuple) and isinstance(node.value, ast.ListComp):
            names = [e.id for e in tgt.elts]
            comps[names[0]] = node.value
    system = {}
    for scen, name in SCENARIO_SYS.items():
        comp = comps[name]
        gen = comp.generators[0]
        var_names = [e.id for e in gen.target.elts]
        variants = []
        for tup in gen.iter.elts:
            env = dict(strings)
            env.update({v: _eval_str(x, strings) for v, x in zip(var_names, tup.elts)})
            variants.append(_eval_str(comp.elt, env))
        system[scen] = tuple(variants)
    if goal_vars is None:
        raise ValueError("GOAL_VARIABLES not found")
    return {"strings": strings, "goal_variables": goal_vars, "system": system}


# ------------------------------------------------------------------------------------------------ variables


def parse_markdown_variables(text: str) -> dict:
    """generate_prompts.parse_markdown_variables: '## Section' followed by a ```json block; keys lower, '-'/' ' -> '_'."""
    out = {}
    for name, body in re.findall(r"## (.*?)\n\s*```json\s*(.*?)\s*```", text, re.DOTALL):
        out[name.strip().lower().replace(" ", "_").replace("-", "_")] = json.loads(body)
    return out


def variables_file(scenario: str, goal_type: str, goal_value: str) -> Path:
    vs = "blackmail" if scenario == "murder" else scenario
    d = TEMPLATES / vs / "variables"
    if goal_type in ("none", "ambiguous") or goal_value in ("none", "ambiguous"):
        return d / "ambiguous.md"
    exact, partial = [], []
    for f in sorted(d.glob("*.md")):
        if f.stem == "ambiguous":
            continue
        stem, g = f.stem.lower(), goal_value.lower()
        if g in stem:
            (exact if (f"_{g}_" in f"_{stem}_" or stem.startswith(g) or stem.endswith(g)) else partial).append(f)
    if exact or partial:
        return (exact or partial)[0]
    return d / "ambiguous.md"


def condition_variables(scenario: str, goal_type: str, goal_value: str, module: dict) -> dict:
    allv = parse_markdown_variables(variables_file(scenario, goal_type, goal_value).read_text(encoding="utf-8"))
    if goal_value in ("none", "ambiguous") or goal_type in ("none", "ambiguous"):
        v = dict(list(allv.values())[0]) if len(allv) == 1 else dict(allv)
    else:
        g = goal_value.lower()
        match = next((sv for k, sv in allv.items() if g in k.lower() or f"pro_{g}" == k.lower()), None)
        v = dict(match if match is not None else list(allv.values())[0])
    gv = module["goal_variables"]
    v.update(gv.get(f"{goal_type}-{goal_value}", gv["none"]))
    return v


# ---------------------------------------------------------------------------------------------------- prompts


def system_prompt(scenario: str, goal_type: str, goal_value: str, module: dict) -> str:
    v = condition_variables(scenario, goal_type, goal_value, module)
    v["extra_system_instructions"] = ""
    v["alignment_faking_warning"] = ""
    return Template(module["system"][scenario][0]).safe_substitute(v)


def email_content(scenario: str, goal_type: str, goal_value: str, urgency: str, module: dict) -> str:
    v = condition_variables(scenario, goal_type, goal_value, module)
    tdir = TEMPLATES / scenario / "templates"
    if goal_type == "none":
        content = Template((tdir / f"email-template-{scenario}-goal-conflict-none-{urgency}.md").read_text(
            encoding="utf-8")).safe_substitute(v)
    else:
        base = (tdir / f"email-template-{scenario}-shared.md").read_text(encoding="utf-8")
        urg = (tdir / f"email-template-{scenario}-{urgency}.md").read_text(encoding="utf-8")
        content = Template(base).safe_substitute(v) + Template(urg).safe_substitute(v)
    if scenario == "leaking":
        content = add_email_ids(content)
    return content


def add_email_ids(content: str) -> str:
    parts = content.split("<email>")
    if len(parts) <= 1:
        return content
    out = parts[0]
    for i, part in enumerate(parts[1:], 1):
        out += f"<email>\nID: {i}{part}"
    return out


def build_cell(cell: str, module: dict | None = None) -> dict:
    """{cell, scenario, ..., system, emails, messages, sha256}."""
    module = module or read_system_module()
    scenario, goal_type, goal_value, urgency = CELLS[cell]
    sys_p = system_prompt(scenario, goal_type, goal_value, module)
    emails = email_content(scenario, goal_type, goal_value, urgency, module)
    user = module["strings"]["USER_PROMPT"] + JOINER + emails
    msgs = [{"role": "system", "content": sys_p}, {"role": "user", "content": user}]
    blob = json.dumps(msgs, ensure_ascii=False, sort_keys=True).encode()
    return {"cell": cell, "scenario": scenario, "goal_type": goal_type, "goal_value": goal_value, "urgency": urgency,
            "system": sys_p, "emails": emails, "messages": msgs, "sha256": hashlib.sha256(blob).hexdigest(),
            "chars": len(sys_p) + len(user)}


def unfilled(text: str) -> list[str]:
    """Template placeholders left unfilled (should be none)."""
    return sorted(set(re.findall(r"\$\{[A-Za-z_]+\}", text)))


def sha256_tree(root: Path = RAW) -> dict:
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}
