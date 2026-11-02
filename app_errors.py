"""« Erreurs » tab of the review app: check the pre-annotated errors call by call (see annotation.py)."""

import gradio as gr
import pandas as pd

from annotation import (CATEGORIES, COLUMNS, call_table, clean_table, load_pre_annotation, load_validations,
                        number_turns, save_validation, summarize)
from corpus import AUDIO_DIR, load_corpus

errors, calls = load_pre_annotation()
validations = load_validations()
texts = load_corpus().set_index("call")
ALL = "toutes"

HELP = (
    "**Colonnes à vérifier :** `categorie`, `gravite`, `garder` (décocher = fausse erreur). "
    "Ajouter une ligne pour une erreur oubliée (laisser `id` vide).\n\n"
    + "\n\n".join(f"**Catégories {c} :** " + " · ".join(f"`{k}`" for k in v) for c, v in CATEGORIES.items())
    + "\n\n**Gravité :** `critique` (change l'intervention : adresse, numéro, état de la victime) · "
    "`majeure` (sens altéré mais récupérable) · `mineure` (forme seulement)"
)


def ids_for(language):
    c = calls if language == ALL else calls[calls.langue == language]
    return c.call.tolist()


def label(call):
    v = validations.get(call)
    mark = "✅" if v and v.get("valide") else ("✏️" if v else "⬜")
    return f"{mark} {call}"


def choices(language):
    return [(label(c), c) for c in ids_for(language)]


def progress():
    n = len(calls)
    validated = sum(1 for c in calls.call if validations.get(c, {}).get("valide"))
    return f"**Annotation des erreurs :** {validated}/{n} appels validés · {len(validations)} enregistrés"


def summary_text(table):
    s = summarize(table)
    if not s["par_categorie"]:
        return "**Résumé :** aucune erreur retenue."
    detail = " · ".join(f"{c[:5]}. {k} : {n}" for (c, k), n in sorted(s["par_categorie"].items()))
    return (f"**Résumé de l'appel :** {s['n_transcription']} erreurs de transcription, "
            f"{s['n_traduction']} de traduction · gravité max : **{s['gravite_max']}** · "
            f"erreur principale : **{s['erreur_principale']}**\n\n{detail}")


def show(call):
    a = calls.set_index("call").loc[call]
    t = texts.loc[call]
    v = validations.get(call, {})
    info = (f"### {call} — {a.langue}, {a.categorie_urgence} (scénario {a.scenario})\n"
            f"Durée : {a.duree_s:.0f} s · {a.tours_ref} tours de parole (auto : {a.tours_auto}) · "
            f"{a.mots_par_tour} mots/tour · ≈ {a.secondes_par_tour} s/tour")
    if v:
        info += f"\n\nDernière modification : {v['date']}" + (f" par {v['par']}" if v.get("par") else "")
    audio = AUDIO_DIR / f"{call}.mp3"
    table = call_table(errors, validations, call)
    return (info, str(audio) if audio.exists() else None,
            number_turns(t.transcription_manuelle), number_turns(t.transcription_auto),
            number_turns(t.traduction_manuelle), number_turns(t.traduction_auto),
            summary_text(table), table, bool(v.get("valide")), v.get("commentaire", ""))


def neighbour(language, call, step):
    ids = ids_for(language)
    i = ids.index(call) if call in ids else 0
    return ids[(i + step) % len(ids)]


def save(language, call, table, validated, comment, request: gr.Request):
    table, problems = clean_table(pd.DataFrame(table, columns=COLUMNS), call)
    if problems:
        raise gr.Error("Non enregistré :\n" + "\n".join(problems[:10]))
    save_validation(validations, call, table, validated, comment, by=request.username)
    gr.Info(f"{call} enregistré")
    return gr.update(choices=choices(language), value=call), progress(), table


def build():
    if calls.empty:
        gr.Markdown("⚠️ Pas de pré-annotation : lancer `python pre_annotation.py`.")
        return
    prog = gr.Markdown(progress())
    with gr.Row():
        language = gr.Dropdown([ALL] + sorted(calls.langue.unique()), value=ALL, label="Langue", scale=1)
        call = gr.Dropdown(choices(ALL), value=calls.call.iloc[0], label="Appel", scale=2)
        btn_prev = gr.Button("◀ Précédent", scale=0)
        btn_next = gr.Button("Suivant ▶", scale=0)

    info = gr.Markdown()
    audio = gr.Audio(type="filepath", label="Audio", interactive=False)
    with gr.Accordion("Textes de l'appel (numéros de tour entre crochets)", open=False):
        with gr.Row():
            transcription_m = gr.Textbox(label="Transcription manuelle", lines=12, max_lines=30, interactive=False)
            transcription_a = gr.Textbox(label="Transcription automatique", lines=12, max_lines=30, interactive=False)
        with gr.Row():
            translation_m = gr.Textbox(label="Traduction manuelle", lines=12, max_lines=30, interactive=False)
            translation_a = gr.Textbox(label="Traduction automatique", lines=12, max_lines=30, interactive=False)

    summary = gr.Markdown()
    with gr.Accordion("Aide : catégories et gravité", open=False):
        gr.Markdown(HELP)
    table = gr.Dataframe(
        headers=COLUMNS, datatype=["str"] * 9 + ["bool"], interactive=True, wrap=True, max_height=600,
        column_widths=["9%", "8%", "4%", "6%", "18%", "18%", "10%", "12%", "8%", "5%"],
        label="Erreurs pré-annotées (une ligne par erreur)",
    )
    with gr.Row():
        validated = gr.Checkbox(label="Appel validé")
        comment = gr.Textbox(label="Commentaire sur l'appel", lines=1, scale=3)
    with gr.Row():
        btn_save = gr.Button("💾 Enregistrer", variant="secondary")
        btn_save_next = gr.Button("💾 Enregistrer et passer au suivant", variant="primary")
        btn_reset = gr.Button("↺ Annuler les modifications non enregistrées")

    outputs = [info, audio, transcription_m, transcription_a, translation_m, translation_a, summary, table,
               validated, comment]
    call.change(show, call, outputs)
    language.change(lambda l: gr.update(choices=choices(l), value=ids_for(l)[0]), language, call)
    btn_prev.click(lambda l, c: neighbour(l, c, -1), [language, call], call)
    btn_next.click(lambda l, c: neighbour(l, c, 1), [language, call], call)
    table.change(lambda t: summary_text(pd.DataFrame(t, columns=COLUMNS)), table, summary)

    fields = [language, call, table, validated, comment]
    btn_save.click(save, fields, [call, prog, table])
    btn_save_next.click(save, fields, [call, prog, table]).success(
        lambda l, c: neighbour(l, c, 1), [language, call], call)
    btn_reset.click(show, call, outputs)
    return call, outputs
