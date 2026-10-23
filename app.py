"""Review interface: listen to each call and correct its transcription/translation.

Run:  python app.py
On Onyxia (VS Code), then open the printed URL (…/proxy/7860/).
"""

import os
from pathlib import Path

import gradio as gr

from corpus import AUDIO_DIR, charger_corpus, charger_revisions, enregistrer_revision

PORT = int(os.environ.get("PORT", 7860))
# temporary public link: PARTAGE=1 python app.py
PARTAGE = os.environ.get("PARTAGE") == "1"
# one account per line "name:password" (do not publish this file)
COMPTES_FILE = Path(__file__).resolve().parent / "comptes.txt"

corpus = charger_corpus()
revisions = charger_revisions()
TOUTES = "toutes"


def ids_pour(langue):
    df = corpus if langue == TOUTES else corpus[corpus.langue == langue]
    return df.id.tolist()


def libelle(id_appel):
    r = revisions.get(id_appel)
    marque = "✅" if r and r.get("valide") else ("✏️" if r else "⬜")
    return f"{marque} {id_appel}"


def choix_dropdown(langue):
    return [(libelle(i), i) for i in ids_pour(langue)]


def progression():
    n = len(corpus)
    valides = sum(1 for i in corpus.id if revisions.get(i, {}).get("valide"))
    modifies = sum(1 for i in corpus.id if i in revisions)
    return f"**Progression :** {valides}/{n} validés · {modifies} enregistrés · ✅ validé · ✏️ enregistré · ⬜ non relu"


def afficher(id_appel):
    ligne = corpus[corpus.id == id_appel].iloc[0]
    rev = revisions.get(id_appel)
    # manual version: last saved correction, otherwise the spreadsheet's corrected version;
    # the automatic version is only used when no manual version exists
    transcription = rev["transcription"] if rev else (ligne.transcription_corrigee or ligne.transcription)
    traduction = rev["traduction"] if rev else (ligne.traduction_corrigee or ligne.traduction)

    infos = [f"### {id_appel} — {ligne.langue}, scénario {ligne.numero_scenario}"]
    if ligne.audio is None:
        infos.append("⚠️ **Audio introuvable** dans `data/audio/`.")
    if not rev and not ligne.transcription_corrigee:
        infos.append("⚠️ Aucune version manuelle : la transcription automatique est affichée, à corriger.")
    if rev:
        par = f" par {rev['par']}" if rev.get("par") else ""
        infos.append(f"Dernière modification : {rev['date']}{par}")

    return (
        "\n\n".join(infos),
        ligne.audio,
        transcription,
        traduction,
        bool(rev and rev.get("valide")),
    )


def voisin(langue, id_appel, pas):
    ids = ids_pour(langue)
    i = ids.index(id_appel) if id_appel in ids else 0
    return ids[(i + pas) % len(ids)]


def sauver(langue, id_appel, transcription, traduction, valide, request: gr.Request):
    enregistrer_revision(revisions, id_appel, transcription, traduction, valide, par=request.username)
    gr.Info(f"{id_appel} enregistré")
    return gr.update(choices=choix_dropdown(langue), value=id_appel), progression()


with gr.Blocks(title="Relecture — appels d'urgence SDIS 01") as demo:
    gr.Markdown("# Relecture du corpus d'appels d'urgence simulés — SDIS 01")
    prog = gr.Markdown(progression())

    with gr.Row():
        langue = gr.Dropdown([TOUTES] + sorted(corpus.langue.unique()), value=TOUTES, label="Langue de l'appelant", scale=1)
        appel = gr.Dropdown(choix_dropdown(TOUTES), value=corpus.id.iloc[0], label="Appel", scale=2)
        btn_prec = gr.Button("◀ Précédent", scale=0)
        btn_suiv = gr.Button("Suivant ▶", scale=0)

    infos = gr.Markdown()
    audio = gr.Audio(type="filepath", label="Audio", interactive=False)

    with gr.Row():
        transcription = gr.Textbox(label="Transcription", lines=18, max_lines=40)
        traduction = gr.Textbox(label="Traduction", lines=18, max_lines=40)

    with gr.Row():
        valide = gr.Checkbox(label="Validé (relecture terminée)")
        btn_save = gr.Button("💾 Enregistrer", variant="secondary")
        btn_save_next = gr.Button("💾 Enregistrer et passer au suivant", variant="primary")
        btn_reset = gr.Button("↺ Annuler les modifications non enregistrées")

    sorties = [infos, audio, transcription, traduction, valide]

    appel.change(afficher, appel, sorties)
    langue.change(lambda l: gr.update(choices=choix_dropdown(l), value=ids_pour(l)[0]), langue, appel)
    btn_prec.click(lambda l, a: voisin(l, a, -1), [langue, appel], appel)
    btn_suiv.click(lambda l, a: voisin(l, a, 1), [langue, appel], appel)

    champs = [langue, appel, transcription, traduction, valide]
    btn_save.click(sauver, champs, [appel, prog])
    btn_save_next.click(sauver, champs, [appel, prog]).then(
        lambda l, a: voisin(l, a, 1), [langue, appel], appel
    )
    btn_reset.click(lambda a: afficher(a)[2:4], appel, [transcription, traduction])

    demo.load(afficher, appel, sorties)


def lire_comptes():
    if not COMPTES_FILE.exists():
        return None
    lignes = [l.strip() for l in COMPTES_FILE.read_text(encoding="utf-8").splitlines()]
    return [tuple(l.split(":", 1)) for l in lignes if l and not l.startswith("#")]


if __name__ == "__main__":
    comptes = lire_comptes()
    if PARTAGE and not comptes:
        raise SystemExit(f"Lien public sans mot de passe refusé : créer {COMPTES_FILE.name} (une ligne nom:motdepasse par personne)")
    # Behind the Onyxia VS Code proxy the app is served under /proxy/<port>/ (not via the public link)
    proxy = os.environ.get("VSCODE_PROXY_URI")
    root_path = f"/proxy/{PORT}" if proxy and not PARTAGE else ""
    if proxy and not PARTAGE:
        print("Ouvrir :", proxy.replace("{{port}}", str(PORT)))
    demo.launch(
        server_name="0.0.0.0",
        server_port=PORT,
        root_path=root_path,
        allowed_paths=[str(AUDIO_DIR)],
        share=PARTAGE,
        # locally, access already requires the Onyxia login (and Gradio's login page fails behind its proxy)
        auth=comptes if PARTAGE else None,
        auth_message="Relecture du corpus SDIS 01 — connectez-vous avec le compte reçu.",
    )
