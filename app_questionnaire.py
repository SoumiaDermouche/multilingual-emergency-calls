"""« Questionnaire » tab of the review app: the questionnaire answered by the caller and the operator,
shown call by call with the scores saved in data/corpus.csv (read only).
"""

import gradio as gr
import pandas as pd

from corpus import DIMENSIONS, DIMENSIONS_BY_ROLE, load_corpus

corpus = load_corpus()
ALL = "toutes"
ROLES = {"Appelant": "appelant", "Opérateur": "operateur"}
SCALE = [("1 – Très mauvaise qualité", 1), ("2 – Mauvaise qualité", 2), ("3 – Qualité moyenne", 3),
         ("4 – Bonne qualité", 4), ("5 – Excellente qualité", 5)]
# wording of the questionnaire given to the participants
QUESTIONS = {
    "appelant": {
        "transcription": "Qualité de la transcription.",
        "traduction_vers_fr": "Qualité de la traduction de ton message (dans ta langue) vers le français.",
        "traduction_vers_appelant": "Qualité de la traduction du message de l'opérateur vers ta langue.",
        "compris": "Sentiment d'être compris.",
        "confiance": "Confiance dans l'utilisation de cette solution en situation réelle.",
    },
    "operateur": {
        "transcription": "Qualité de la transcription.",
        "traduction_vers_fr": "Qualité de la traduction du message de l'appelant vers le français.",
        "compris": "Sentiment d'être compris.",
        "confiance": "Confiance dans l'utilisation de cette solution en situation réelle.",
    },
}


def ids_for(language):
    df = corpus if language == ALL else corpus[corpus.langue == language]
    return df.call.tolist()


def neighbour(language, call, step):
    ids = ids_for(language)
    i = ids.index(call) if call in ids else 0
    return ids[(i + step) % len(ids)]


def show(call, role_label):
    row = corpus[corpus.call == call].iloc[0]
    role = ROLES[role_label]
    info = f"### {call} — {row.langue}, scénario {row.scenario} ({row.categorie_urgence})"
    # one radio per dimension; the questions not asked to this role are hidden
    questions = []
    for d in DIMENSIONS:
        if d in DIMENSIONS_BY_ROLE[role]:
            score = row[f"{role}_{d}"]
            number = DIMENSIONS_BY_ROLE[role].index(d) + 1
            questions.append(gr.update(label=f"{number}. {QUESTIONS[role][d]}", visible=True,
                                       value=None if pd.isna(score) else int(score)))
        else:
            questions.append(gr.update(visible=False, value=None))
    return [info, row.audio, row.transcription_auto, row.traduction_auto] + questions


def build():
    with gr.Row():
        role = gr.Radio(list(ROLES), value="Appelant", label="Questionnaire", scale=1)
        language = gr.Dropdown([ALL] + sorted(corpus.langue.unique()), value=ALL, label="Langue", scale=1)
        call = gr.Dropdown(ids_for(ALL), value=corpus.call.iloc[0], label="Appel", scale=2)
        btn_prev = gr.Button("◀ Précédent", scale=0)
        btn_next = gr.Button("Suivant ▶", scale=0)

    info = gr.Markdown()
    audio = gr.Audio(type="filepath", label="Audio", interactive=False)
    with gr.Row():
        transcription = gr.Textbox(label="Transcription automatique", lines=14, max_lines=40, interactive=False)
        translation = gr.Textbox(label="Traduction automatique", lines=14, max_lines=40, interactive=False)

    gr.Markdown("#### Réponses enregistrées (échelle de Likert, 1 à 5)")
    questions = [gr.Radio(SCALE, interactive=False) for _ in DIMENSIONS]

    outputs = [info, audio, transcription, translation] + questions
    call.change(show, [call, role], outputs)
    role.change(show, [call, role], outputs)
    language.change(lambda l: gr.update(choices=ids_for(l), value=ids_for(l)[0]), language, call)
    btn_prev.click(lambda l, c: neighbour(l, c, -1), [language, call], call)
    btn_next.click(lambda l, c: neighbour(l, c, 1), [language, call], call)
    return call, role, outputs
