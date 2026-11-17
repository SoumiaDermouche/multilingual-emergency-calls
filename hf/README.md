---
license: etalab-2.0
language:
  - fr
  - en
  - it
  - ar
  - es
multilinguality: multilingual
task_categories:
  - automatic-speech-recognition
  - translation
size_categories:
  - n<1K
pretty_name: Multilingual Emergency Calls (simulated, SDIS 01)
tags:
  - emergency-calls
  - speech-translation
---

# Multilingual Emergency Calls (simulated, SDIS 01)

77 simulated emergency calls between a French-speaking fire department operator (SDIS 01, Ain, France)
and a caller speaking English, Italian, Arabic or Spanish, with their manual transcription and translation.

| Caller's language | Calls |
|---|---|
| English | 20 |
| Italian | 20 |
| Arabic | 20 |
| Spanish | 17 |

All calls are **role-played scenarios**: they contain no real emergency and no personal data.
All participants gave their consent for the distribution of the recordings.

## Data fields

One row per call (split `test`):

| Column | Description |
|---|---|
| `audio` | Recording of the call (mp3) |
| `call_id` | `<language>_<scenario>` (language name in French), e.g. `anglais_1.1` |
| `language` | Caller's language, ISO 639-1 code: `en`, `it`, `ar` or `es` (the operator always speaks French) |
| `transcription` | Manual transcription, in the language spoken by each speaker |
| `translation` | Manual translation: caller's turns into French, operator's turns into the caller's language |

Texts are split into turns, one per line, each starting with `Operator:` or `Caller:`.

## Usage

```python
from datasets import load_dataset

ds = load_dataset("SoumiaDermouche/multilingual-emergency-calls", split="test")
print(ds[0]["transcription"])

arabic = ds.filter(lambda row: row["language"] == "ar")
```

## License

[Licence Ouverte / Open Licence 2.0 (Etalab)](https://www.etalab.gouv.fr/licence-ouverte-open-licence/).

## Citation

If you use this dataset, please cite:

> S. Dermouche, E. Bellicaud, R. Hamdan, M. Coulmeau, C. Guyeux, D. Laiymani, C. Marquaille and G. Royer-Fey,
> "A Real-Time Speech Translation Pipeline for Emergency Calls",
> in *Proc. IEEE Spoken Language Technology Workshop (SLT)*, Italy, December 2026.

```bibtex
@inproceedings{dermouche2026realtime,
  title     = {A Real-Time Speech Translation Pipeline for Emergency Calls},
  author    = {Dermouche, Soumia and Bellicaud, Enzo and Hamdan, Rawi and Coulmeau, Maxime and
               Guyeux, Christophe and Laiymani, David and Marquaille, Christophe and Royer-Fey, Guillaume},
  booktitle = {Proceedings of the IEEE Spoken Language Technology Workshop (SLT)},
  address   = {Italy},
  month     = dec,
  year      = {2026}
}
```
