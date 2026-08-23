"""PayGuard — Entraînement du modèle anti-fraude v1. 💻 FOURNI — à exécuter, puis à lire.

Ce script rejoue en accéléré ce que vous avez fait au Sprint 2 :
  1. split train / validation (stratifié — ici les transactions sont indépendantes,
     contrairement aux capteurs InduSense où le split devait être temporel) ;
  2. entraînement d'un HistGradientBoostingClassifier (vu au module 11) ;
  3. évaluation sur la VALIDATION : PR-AUC et ROC-AUC (module 21) ;
  4. choix du SEUIL par balayage du coût métier sur la validation (module 21) :
        coût = 50 € par fraude ratée (FN)  +  2 € par vérification inutile (FP)
     puis le seuil est GELÉ : c'est lui qui sera utilisé toutes les semaines suivantes.

Sorties :
  models/model.joblib     — le pipeline entraîné
  models/threshold.json   — seuil gelé + métriques de validation (la « carte d'identité »
                            du modèle, cf. model card du module 22)

Usage :  uv run python scripts/train_model.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

RACINE = Path(__file__).resolve().parents[1]
DATA = RACINE / "data"
MODELES = RACINE / "models"

FEATURES = ["montant_eur", "heure", "anciennete_client_j", "nb_articles", "distance_domicile_km"]
COUT_FN = 50.0  # € — fraude non détectée (remboursement client, litige)
COUT_FP = 2.0   # € — vérification manuelle inutile (équipe risque)


def choisir_seuil_par_cout(y_true: np.ndarray, proba: np.ndarray) -> tuple[float, float]:
    """Balaye les seuils sur la VALIDATION et retourne (seuil, coût) minimisant :
    coût = COUT_FN × FN + COUT_FP × FP.   (Règle d'or m21 : le seuil se choisit
    sur la validation par le coût métier — jamais sur le test / la prod.)"""
    meilleur_seuil, meilleur_cout = 0.5, float("inf")
    for seuil in np.arange(0.02, 0.981, 0.01):
        y_pred = (proba >= seuil).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        cout = COUT_FN * fn + COUT_FP * fp
        if cout < meilleur_cout:
            meilleur_seuil, meilleur_cout = float(round(seuil, 2)), float(cout)
    return meilleur_seuil, meilleur_cout


def main() -> None:
    chemin = DATA / "reference.csv"
    if not chemin.exists():
        raise SystemExit(
            "[ERREUR] data/reference.csv introuvable — vérifiez que vous lancez le script "
            "depuis le dossier du TP (les données sont fournies dans data/)."
        )

    ref = pd.read_csv(chemin)
    X, y = ref[FEATURES], ref["fraude"]
    print(f"Référence : {len(ref)} transactions, taux de fraude = {y.mean():.2%}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    modele = HistGradientBoostingClassifier(random_state=42)
    modele.fit(X_train, y_train)

    proba_val = modele.predict_proba(X_val)[:, 1]
    pr_auc = float(average_precision_score(y_val, proba_val))
    roc_auc = float(roc_auc_score(y_val, proba_val))

    seuil, cout = choisir_seuil_par_cout(y_val.to_numpy(), proba_val)
    y_pred = (proba_val >= seuil).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
    rappel = tp / (tp + fn)
    precision = tp / (tp + fp)
    taux_alerte = float(y_pred.mean())

    MODELES.mkdir(parents=True, exist_ok=True)
    joblib.dump(modele, MODELES / "model.joblib")
    carte = {
        "modele": "HistGradientBoostingClassifier(random_state=42)",
        "features": FEATURES,
        "seuil": seuil,
        "cout_fn_eur": COUT_FN,
        "cout_fp_eur": COUT_FP,
        "validation": {
            "pr_auc": round(pr_auc, 4),
            "roc_auc": round(roc_auc, 4),
            "rappel": round(float(rappel), 4),
            "precision": round(float(precision), 4),
            "taux_alerte": round(taux_alerte, 4),
            "prevalence_fraude": round(float(y_val.mean()), 4),
        },
    }
    (MODELES / "threshold.json").write_text(
        json.dumps(carte, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n=== Validation (25 % de la référence) ===")
    print(f"  PR-AUC  = {pr_auc:.3f}   (ligne de base = prévalence = {y_val.mean():.3f})")
    print(f"  ROC-AUC = {roc_auc:.3f}   (ligne de base = 0.500)")
    # Garder la sortie compatible avec la console Windows PowerShell 5.1 (CP-1252).
    print(f"  Seuil gelé (coût FN={COUT_FN:.0f}€ > FP={COUT_FP:.0f}€) : {seuil}")
    print(f"  Au seuil {seuil} : rappel = {rappel:.3f} · précision = {precision:.3f} "
          f"· taux d'alerte = {taux_alerte:.2%}")
    print(f"  Matrice de confusion [TN FP / FN TP] : [{tn} {fp} / {fn} {tp}]")
    print("\n[OK] Modèle et seuil gelés dans models/ — ne plus y toucher pendant le TP.")


if __name__ == "__main__":
    main()
