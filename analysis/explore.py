"""marimo operator cockpit (SPEC §4 stages 6-7, §5).

The reactive analysis surface: swap the checkpoint path at the top and every
downstream cell re-derives via ``head.viewer()``. The "verifiable" brand,
dogfooded — you refuse to decide on a number that no longer corresponds to any
runnable state, which is the defensible answer to "why not Jupyter."

Domain-blind: dispatches to the head; never branches on the domain.

    marimo edit analysis/explore.py        # author
    marimo run analysis/explore.py \       # serve read-only (proxy, #3)
      --host 127.0.0.1 --port 2718 --headless --no-token \
      --base-url /jupyterlab/default/proxy/2718
"""

import marimo

app = marimo.App(width="medium")


@app.cell
def _():
    import sys
    from pathlib import Path

    import marimo as mo

    repo = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo / "src"))
    from spine import registry

    return mo, registry


@app.cell
def _(mo, registry):
    # Top-of-notebook controls — change these and everything below re-derives.
    domain = mo.ui.dropdown(
        options=registry.names(),
        value=registry.names()[0] if registry.names() else None,
        label="domain",
    )
    checkpoint = mo.ui.text(value="checkpoints/model.pt", label="checkpoint")
    mo.hstack([domain, checkpoint])
    return checkpoint, domain


@app.cell
def _(checkpoint, domain, mo, registry):
    head = registry.load(domain.value)
    viewer = head.viewer()
    _ckpt = checkpoint.value
    mo.md(f"**{head.name}** · metric `{head.metric_name()}` · checkpoint `{_ckpt}`")
    return head, viewer


@app.cell
def _(checkpoint, mo, viewer):
    # One example input, rendered through the head's viewer. Reactive: edit the
    # checkpoint above and this re-runs against the new (runnable) state.
    sample = "CCO"
    out = viewer.render(checkpoint.value, sample) if checkpoint.value else None
    mo.md(f"`{sample}` → {out}") if out else mo.md("_set a valid checkpoint above_")
    return


if __name__ == "__main__":
    app.run()
