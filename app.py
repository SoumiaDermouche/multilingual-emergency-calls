"""Review interface: listen to each call and correct its transcription/translation (tab « Relecture »),
then check the pre-annotated errors call by call (tab « Erreurs », see app_errors.py); the tab
« Questionnaire » shows the questionnaire and the scores of each call (see app_questionnaire.py).

Run:  python app.py
On Onyxia (VS Code), then open the printed URL (…/proxy/7860/).
"""

import os
from pathlib import Path

import gradio as gr

import app_errors
import app_questionnaire
from corpus import AUDIO_DIR, load_corpus, save_revision

PORT = int(os.environ.get("PORT", 7860))
# temporary public link: PARTAGE=1 python app.py
SHARE = os.environ.get("PARTAGE") == "1"
# one account per line "name:password" (do not publish this file)
ACCOUNTS_FILE = Path(__file__).resolve().parent / "comptes.txt"

corpus = load_corpus()
ALL = "toutes"


def ids_for(language):
    df = corpus if language == ALL else corpus[corpus.langue == language]
    return df.call.tolist()


def show(call_id):
    row = corpus[corpus.call == call_id].iloc[0]

    info = [f"### {call_id} — {row.langue}, scénario {row.scenario} ({row.categorie_urgence})"]
    if row.audio is None:
        info.append("⚠️ **Audio introuvable** dans `data/audio/`.")

    return (
        "\n\n".join(info),
        row.audio,
        row.transcription_manuelle,
        row.traduction_manuelle,
        row.transcription_auto,
        row.traduction_auto,
    )


def neighbour(language, call_id, step):
    ids = ids_for(language)
    i = ids.index(call_id) if call_id in ids else 0
    return ids[(i + step) % len(ids)]


def save(call_id, transcription, translation):
    global corpus
    corpus = save_revision(call_id, transcription, translation)
    gr.Info(f"{call_id} enregistré")


with gr.Blocks(title="Relecture — appels d'urgence SDIS 01") as demo:
    gr.Markdown("# Relecture du corpus d'appels d'urgence simulés — SDIS 01")

    with gr.Tab("Relecture"):
        with gr.Row():
            language = gr.Dropdown([ALL] + sorted(corpus.langue.unique()), value=ALL, label="Langue de l'appelant", scale=1)
            call = gr.Dropdown(ids_for(ALL), value=corpus.call.iloc[0], label="Appel", scale=2)
            btn_prev = gr.Button("◀ Précédent", scale=0)
            btn_next = gr.Button("Suivant ▶", scale=0)

        info = gr.Markdown()
        audio = gr.Audio(type="filepath", label="Audio", interactive=False)

        with gr.Row():
            with gr.Column():
                transcription = gr.Textbox(label="Transcription manuelle", lines=18, max_lines=40)
                transcription_auto = gr.Textbox(label="Transcription automatique (lecture seule)", lines=18,
                                                max_lines=40, interactive=False)
            with gr.Column():
                translation = gr.Textbox(label="Traduction manuelle", lines=18, max_lines=40)
                translation_auto = gr.Textbox(label="Traduction automatique (lecture seule)", lines=18,
                                              max_lines=40, interactive=False)

        with gr.Row():
            btn_save = gr.Button("💾 Enregistrer", variant="secondary")
            btn_save_next = gr.Button("💾 Enregistrer et passer au suivant", variant="primary")
            btn_reset = gr.Button("↺ Annuler les modifications non enregistrées")

        outputs = [info, audio, transcription, translation, transcription_auto, translation_auto]

        call.change(show, call, outputs)
        language.change(lambda l: gr.update(choices=ids_for(l), value=ids_for(l)[0]), language, call)
        btn_prev.click(lambda l, c: neighbour(l, c, -1), [language, call], call)
        btn_next.click(lambda l, c: neighbour(l, c, 1), [language, call], call)

        fields = [call, transcription, translation]
        btn_save.click(save, fields)
        btn_save_next.click(save, fields).then(
            lambda l, c: neighbour(l, c, 1), [language, call], call
        )
        btn_reset.click(lambda c: show(c)[2:4], call, [transcription, translation])

        demo.load(show, call, outputs)

    with gr.Tab("Erreurs"):
        errors_tab = app_errors.build()
    if errors_tab:
        errors_call, errors_outputs = errors_tab
        demo.load(app_errors.show, errors_call, errors_outputs)

    with gr.Tab("Questionnaire"):
        questionnaire_call, questionnaire_role, questionnaire_outputs = app_questionnaire.build()
    demo.load(app_questionnaire.show, [questionnaire_call, questionnaire_role], questionnaire_outputs)


def read_accounts():
    if not ACCOUNTS_FILE.exists():
        return None
    lines = [l.strip() for l in ACCOUNTS_FILE.read_text(encoding="utf-8").splitlines()]
    return [tuple(l.split(":", 1)) for l in lines if l and not l.startswith("#")]


if __name__ == "__main__":
    accounts = read_accounts()
    if SHARE and not accounts:
        raise SystemExit(f"Lien public sans mot de passe refusé : créer {ACCOUNTS_FILE.name} (une ligne nom:motdepasse par personne)")
    # Behind the Onyxia VS Code proxy the app is served under /proxy/<port>/ (not via the public link)
    proxy = os.environ.get("VSCODE_PROXY_URI")
    root_path = f"/proxy/{PORT}" if proxy and not SHARE else ""
    if proxy and not SHARE:
        print("Ouvrir :", proxy.replace("{{port}}", str(PORT)))
    demo.launch(
        server_name="0.0.0.0",
        server_port=PORT,
        root_path=root_path,
        allowed_paths=[str(AUDIO_DIR)],
        share=SHARE,
        # locally, access already requires the Onyxia login (and Gradio's login page fails behind its proxy)
        auth=accounts if SHARE else None,
        auth_message="Relecture du corpus SDIS 01 — connectez-vous avec le compte reçu.",
    )
