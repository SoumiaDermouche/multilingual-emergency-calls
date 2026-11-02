# Corpus d'appels d'urgence simulés — SDIS 01

Objectif : relire/corriger les transcriptions et traductions des appels simulés,
puis publier le corpus sur Hugging Face.

## Relecture

```bash
pip install -r requirements.txt   # à refaire si le service Onyxia a redémarré
python app.py
```

Puis ouvrir `…/proxy/7860/` (l'URL s'affiche au lancement, ou onglet *Ports* de VS Code).
Les corrections sont enregistrées directement dans `data/corpus.csv`.
L'onglet « Questionnaire » affiche, appel par appel, les questions posées aux participants et leurs réponses.

## Données

- `data/audio/<langue>_<scénario>.mp3` (ex. `italien_2.10.mp3`)
- `data/corpus.csv` : une ligne par appel (77 appels : anglais 20, italien 20, arabe 20, espagnol 17), 17 colonnes
  - `call`, `langue`, `scenario`, `categorie_urgence`
  - `transcription_auto`, `transcription_manuelle` : transcription Whisper large-v3-turbo et sa correction
  - `traduction_auto`, `traduction_manuelle` : traduction mistral-small-24b et sa correction
  - `appelant_*` (5 notes de 1 à 5) : `transcription`, `traduction_vers_fr`, `traduction_vers_appelant`,
    `compris`, `confiance`
  - `operateur_*` (4 notes) : les mêmes sans `traduction_vers_appelant` (l'opérateur ne parle que français)

L'opérateur parle français, l'appelant anglais, italien, espagnol ou arabe.

## Annotation des erreurs

```bash
python pre_annotation.py   # pré-annotation automatique -> resultats/annotation/
python app.py              # onglet « Erreurs » : validation appel par appel
python annotation.py       # export des appels validés -> erreurs_validees.csv, resume_par_appel.csv
```

Les validations sont enregistrées dans `data/annotation_erreurs.json`. Les appels
en turc et en danois ne font pas partie de l'étude et sont ignorés.

Pour `pre_annotation.py` (modèle de NER, plus lourd) : `pip install -r requirements-analyse.txt`.
