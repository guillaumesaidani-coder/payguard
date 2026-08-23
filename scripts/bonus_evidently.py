"""PayGuard — BONUS facultatif (étape 8) : rapport de dérive avec Evidently. 💻 FOURNI.

Evidently est l'outil « clé en main » qui emballe PSI/KS & co dans un rapport HTML —
c'est lui qui sera utilisé au module 32 pour automatiser la détection sur InduSense.
Même version que le lock InduSense : evidently 0.7.x.

Installation (uniquement si vous faites ce bonus) :
    uv sync --frozen --extra dev --extra evidently        # voie uv
    pip install "evidently>=0.7,<0.8"            # voie pip/venv

Usage :  uv run python scripts/bonus_evidently.py --semaine 2

En cas de souci d'installation ou d'API : ce bonus est SANS enjeu — votre
drift_lab.py fait déjà tout le travail (c'est le « repli maison » officiel
du module 32). Passez à la synthèse.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

RACINE = Path(__file__).resolve().parents[1]
DATA = RACINE / "data"
RAPPORTS = RACINE / "reports"

FEATURES = ["montant_eur", "heure", "anciennete_client_j", "nb_articles", "distance_domicile_km"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Rapport Evidently PayGuard (bonus)")
    parser.add_argument("--semaine", type=int, required=True, choices=[1, 2, 3])
    args = parser.parse_args()

    try:
        # API Evidently 0.7.x (même famille de version que le lock InduSense)
        from evidently import Report
        from evidently.presets import DataDriftPreset
    except ImportError:
        raise SystemExit(
            "[ERREUR] Evidently n'est pas installé (ou API incompatible).\n"
            "   -> uv sync --frozen --extra dev --extra evidently   (ou pip install 'evidently>=0.7,<0.8')\n"
            "   Ce bonus est facultatif : drift_lab.py couvre déjà PSI + KS."
        )

    df_ref = pd.read_csv(DATA / "reference.csv")[FEATURES]
    df_cur = pd.read_csv(DATA / f"courant_semaine{args.semaine}.csv")[FEATURES]

    rapport = Report([DataDriftPreset()])
    resultat = rapport.run(reference_data=df_ref, current_data=df_cur)

    RAPPORTS.mkdir(parents=True, exist_ok=True)
    sortie = RAPPORTS / f"evidently_semaine{args.semaine}.html"
    resultat.save_html(str(sortie))
    print(f"[OK] Rapport écrit : {sortie}")
    print("   Ouvrez-le dans un navigateur et comparez ses verdicts à votre table drift_lab :")
    print("   mêmes features en dérive ? mêmes ordres de grandeur ? quels tests Evidently a-t-il choisis ?")


if __name__ == "__main__":
    main()
