"""Publish the corpus on Hugging Face: 77 calls, audio, call id, language, manual transcription and translation.

The dataset is exported in the "audiofolder" layout, which the Hugging Face viewer plays directly:
    hf_export/
      README.md               dataset card (copied from hf/README.md)
      test/metadata.csv       one row per call, "file_name" points to the audio file
      test/<call>.mp3

Usage:
    hf auth login                                         # once, with a "write" token (never put it in the code)
    python publish_hf.py --dry-run                        # build hf_export/ only, to check it
    python publish_hf.py --repo-id <user>/<dataset>       # build and upload (private repository)
    python publish_hf.py --repo-id <user>/<dataset> --public
"""

import argparse
import shutil
from pathlib import Path

from corpus import ROOT, load_corpus

CARD_FILE = ROOT / "hf" / "README.md"
EXPORT_DIR = ROOT / "hf_export"
SPLIT = "test"  # evaluation corpus: a single split

LANGUAGE_CODES = {"anglais": "en", "italien": "it", "arabe": "ar", "espagnol": "es"}
# only the manually corrected texts are published, under these names
TEXT_COLUMNS = {"transcription_manuelle": "transcription", "traduction_manuelle": "translation"}


def build_metadata():
    """One row per call: audio file name (relative to metadata.csv) + call id, language, manual transcription and translation."""
    corpus = load_corpus()
    missing = corpus.loc[corpus.audio.isna(), "call"].tolist()
    if missing:
        raise SystemExit(f"Audio file missing for: {missing}")
    unknown = set(corpus.langue) - set(LANGUAGE_CODES)
    if unknown:
        raise SystemExit(f"Languages without an ISO code: {unknown}")
    meta = corpus.rename(columns={"call": "call_id", **TEXT_COLUMNS})
    meta.insert(0, "file_name", corpus.call + ".mp3")
    meta["language"] = corpus.langue.map(LANGUAGE_CODES)
    return meta[["file_name", "call_id", "language", *TEXT_COLUMNS.values()]], corpus.audio


def build_export():
    """Write hf_export/ (card + metadata + audio copies); return its path."""
    meta, audio = build_metadata()
    if EXPORT_DIR.exists():
        shutil.rmtree(EXPORT_DIR)
    split_dir = EXPORT_DIR / SPLIT
    split_dir.mkdir(parents=True)
    shutil.copy2(CARD_FILE, EXPORT_DIR / "README.md")
    meta.to_csv(split_dir / "metadata.csv", index=False, encoding="utf-8")
    for path in audio:
        shutil.copy2(path, split_dir / Path(path).name)
    print(f"{len(meta)} calls exported to {EXPORT_DIR}, columns: {list(meta.columns)}")
    return EXPORT_DIR


def upload(folder, repo_id, private):
    from huggingface_hub import HfApi

    api = HfApi()  # uses the token saved by "hf auth login"
    print(f"Logged in as: {api.whoami()['name']}")
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    api.upload_folder(repo_id=repo_id, repo_type="dataset", folder_path=folder,
                      commit_message="Publish the multilingual emergency calls corpus")
    print(f"Done: https://huggingface.co/datasets/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-id", help="e.g. <hf_username>/multilingual-emergency-calls")
    parser.add_argument("--dry-run", action="store_true", help="build hf_export/ without uploading")
    parser.add_argument("--public", action="store_true", help="create the repository as public (default: private)")
    args = parser.parse_args()
    if not args.dry_run and not args.repo_id:
        parser.error("--repo-id is required unless --dry-run")

    folder = build_export()
    if not args.dry_run:
        upload(folder, args.repo_id, private=not args.public)


if __name__ == "__main__":
    main()
