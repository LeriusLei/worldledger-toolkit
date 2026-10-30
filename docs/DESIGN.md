# Visual design

## Measured layers

Space expresses the distinction between a set of possible worlds and an action shared across them. Offset planes carry a repeated lattice; one continuous path supplies a visual reference through the layers. The geometry is original and conceptual. It does not encode an additional experiment or quantitative result.

The palette uses deep ink, warm paper and a small amount of green or blue to distinguish selected information. Color has a specific job: paths in the brand image, controls in a chart, and coherence in the component matrix. Consistent alignment, calibrated spacing and thin construction marks keep the visual detail intentional at both full size and README width.

Instrument Sans carries the primary hierarchy, while JetBrains Mono identifies code, values and interface names. The fonts are bundled under their original SIL Open Font Licenses. SVG exports outline the glyphs for consistent rendering; PNG exports provide a predictable GitHub preview. Typography remains separate from plotted measurements.

Each scientific graphic retains its underlying question, source and interpretation. The utility/uncertainty figure reads the published summary; the coherence figure follows the executable counterexample. The second design pass checks label separation, number alignment, caption legibility and the appearance at a narrower README width. Additional decoration is avoided when a clearer alignment resolves the problem.

## Regeneration

```bash
python -m pip install -e ".[plots]"
python scripts/create_figures.py
```

Source: [create_visuals.py](../scripts/create_visuals.py). Outputs: [banner](assets/banner.png), [system map](assets/architecture.png), [Sprinkler control](assets/sprinkler_results.png) and [coherence control](assets/coherence.png), with SVG counterparts in the same directory. Font notices are in [Instrument Sans OFL](../licenses/InstrumentSans-OFL.txt) and [JetBrains Mono OFL](../licenses/JetBrainsMono-OFL.txt).

## Documentation references

Reviewed upstream READMEs on 2026-10-30. These are observations and design inferences, not evidence that a particular layout causes adoption. No upstream logos, text blocks or diagrams were copied.

| Reference | Observed README practice | Applied here |
|---|---|---|
| [vLLM](https://github.com/vllm-project/vllm) | A concise purpose statement and prominent navigation to documentation | A clear opening and a compact documentation row |
| [uv](https://github.com/astral-sh/uv) | Concrete workflows, command/output examples and grouped highlights | A complete finite-world code example with its actual output |
| [DSPy](https://github.com/stanfordnlp/dspy) | A focused explanation of modular use and a direct documentation entry | Three package-level reuse paths |
| [Ray](https://github.com/ray-project/ray) | An overview diagram tied to named libraries and abstractions | A system map with package names and information boundaries |
