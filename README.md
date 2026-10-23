# Corpus d'appels d'urgence simulés — SDIS 01

Objectif : relire/corriger les transcriptions et traductions des appels simulés,
puis publier le corpus sur Hugging Face.

## Relecture

```bash
pip install pandas openpyxl odfpy gradio
python app.py
```

Puis ouvrir `…/proxy/7860/` (l'URL s'affiche au lancement, ou onglet *Ports* de VS Code).
Les re-corrections sont enregistrées dans `data/revisions.json` ; les tableurs
d'origine ne sont jamais modifiés.

## Données

- `data/audio/<langue>_<scénario>.mp3` (ex. `italien_2.10.mp3`)
- `data/translation+ transcription/` : un tableur par langue (anglais, italien
  — feuille « après correction » —, espagnol), une ligne par scénario.

L'opérateur parle français, l'appelant anglais, italien ou espagnol.
