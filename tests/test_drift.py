"""Suite de tests du TP PayGuard. 💻 FOURNIE — c'est votre « definition of done ».

Deux familles :
  1. Tests UNITAIRES (aucune donnée requise) — valident vos fonctions psi(),
     ks_pvalue(), drift_table() sur des distributions synthétiques contrôlées.
     → uv run python -m pytest -k "psi or ks or verdict or table"
  2. Tests d'INTÉGRATION (données + modèle requis) — valident le scénario complet
     du TP : semaine calme silencieuse, covariate drift détecté, concept drift
     invisible au PSI mais visible sur le rappel.
     → uv run python -m pytest  (tout doit être vert à la fin du TP)

⚠️ Ne modifiez PAS ce fichier : adaptez votre code, pas les tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE / "scripts"))

import drift_lab  # noqa: E402

DATA = RACINE / "data"
MODELES = RACINE / "models"

# ---------------------------------------------------------------------------
# 1. Tests unitaires — fonctions psi / ks_pvalue / verdict / drift_table
# ---------------------------------------------------------------------------


def test_psi_quasi_nul_sur_distributions_identiques():
    rng = np.random.default_rng(0)
    ref = rng.normal(0.0, 1.0, 20_000)
    cur = rng.normal(0.0, 1.0, 5_000)
    assert drift_lab.psi(ref, cur) < 0.05


def test_psi_nul_sur_elle_meme():
    rng = np.random.default_rng(1)
    ref = rng.lognormal(3.0, 0.7, 10_000)
    assert abs(drift_lab.psi(ref, ref)) < 1e-3


def test_psi_detecte_un_decalage_de_moyenne():
    rng = np.random.default_rng(2)
    ref = rng.normal(0.0, 1.0, 20_000)
    cur = rng.normal(1.0, 1.0, 5_000)  # décalage de +1 écart-type
    assert drift_lab.psi(ref, cur) > 0.25


def test_psi_compte_les_valeurs_hors_plage_de_reference():
    # La MOITIÉ de la masse courante sort de la plage de la référence (pensez
    # inflation, changement d'unité…). Une implémentation qui laisse np.histogram
    # ignorer ces valeurs sous-estime gravement le PSI (~0,35, voire ~0 selon la
    # normalisation, au lieu de ~1,1). Indice : ouvrir les bords extrêmes (±inf).
    ref = np.linspace(0.0, 1.0, 5_000)
    cur = np.concatenate([np.linspace(0.0, 1.0, 1_000), np.full(1_000, 5.0)])
    assert drift_lab.psi(ref, cur) > 0.8


def test_ks_pvalue_elevee_si_distributions_identiques():
    rng = np.random.default_rng(3)
    ref = rng.normal(0.0, 1.0, 4_000)
    cur = rng.normal(0.0, 1.0, 4_000)
    assert drift_lab.ks_pvalue(ref, cur) > 0.001


def test_ks_pvalue_minuscule_si_decalage():
    rng = np.random.default_rng(4)
    ref = rng.normal(0.0, 1.0, 4_000)
    cur = rng.normal(1.0, 1.0, 4_000)
    assert drift_lab.ks_pvalue(ref, cur) < 1e-6


def test_verdict_psi_suit_les_seuils_du_cours():
    assert "RAS" in drift_lab.verdict_psi(0.05)
    assert "surveiller" in drift_lab.verdict_psi(0.18)
    assert "forte" in drift_lab.verdict_psi(0.60)


def test_drift_table_structure_et_tri():
    import pandas as pd

    rng = np.random.default_rng(5)
    df_ref = pd.DataFrame({"a": rng.normal(0, 1, 5_000), "b": rng.normal(0, 1, 5_000)})
    df_cur = pd.DataFrame({"a": rng.normal(0, 1, 2_000), "b": rng.normal(2, 1, 2_000)})
    table = drift_lab.drift_table(df_ref, df_cur, features=["a", "b"])
    assert list(table.columns) == ["feature", "psi", "ks_pvalue", "verdict"]
    assert len(table) == 2
    assert table.loc[0, "feature"] == "b"  # tri par PSI décroissant : b dérive, a non
    assert table.loc[0, "psi"] > table.loc[1, "psi"]


# ---------------------------------------------------------------------------
# 2. Tests d'intégration — scénario complet PayGuard
# ---------------------------------------------------------------------------

_donnees_presentes = (DATA / "reference.csv").exists() and (DATA / "courant_semaine3.csv").exists()
_modele_present = (MODELES / "model.joblib").exists() and (MODELES / "threshold.json").exists()

necessite_donnees = pytest.mark.skipif(
    not _donnees_presentes, reason="données absentes (dossier data/ fourni avec le TP)"
)
necessite_modele = pytest.mark.skipif(
    not (_donnees_presentes and _modele_present),
    reason="modèle absent — lancez d'abord : uv run python scripts/train_model.py",
)


def _charger(nom: str):
    import pandas as pd

    return pd.read_csv(DATA / nom)


@necessite_donnees
def test_integration_semaine1_aucune_derive():
    table = drift_lab.drift_table(_charger("reference.csv"), _charger("courant_semaine1.csv"))
    assert (table["psi"] < 0.10).all(), "semaine 1 : tous les PSI doivent être < 0,10 (RAS)"


@necessite_donnees
def test_integration_semaine2_covariate_drift_detecte():
    table = drift_lab.drift_table(_charger("reference.csv"), _charger("courant_semaine2.csv"))
    table = table.set_index("feature")
    assert table.loc["montant_eur", "psi"] > 0.25, "le montant doit être en dérive forte"
    assert (table["psi"] > 0.25).sum() >= 2, "au moins 2 features en dérive forte"
    assert table.loc["montant_eur", "ks_pvalue"] < 1e-4, "KS doit confirmer (p ≈ 0)"


@necessite_donnees
def test_integration_semaine3_silence_des_entrees():
    table = drift_lab.drift_table(_charger("reference.csv"), _charger("courant_semaine3.csv"))
    assert (table["psi"] < 0.10).all(), (
        "semaine 3 : les entrées n'ont PAS bougé — le PSI doit rester silencieux "
        "(c'est tout l'intérêt pédagogique de cette semaine)"
    )


@necessite_modele
def test_integration_semaine3_le_rappel_s_effondre():
    import evaluate_semaine

    m1 = evaluate_semaine.evaluer_semaine(1)
    m3 = evaluate_semaine.evaluer_semaine(3)
    assert m1["rappel"] > 0.60, "semaine 1 : le modèle doit encore bien rappeler la fraude"
    assert m3["rappel"] < 0.5 * m1["rappel"], (
        "semaine 3 : concept drift — le rappel doit s'effondrer alors que le PSI est muet"
    )
    # Panne SILENCIEUSE : côté opérations, rien ne bouge — le volume d'alertes
    # reste celui d'une semaine normale (à ±5 points), seul le rappel révèle le problème.
    assert abs(m3["taux_alerte"] - m1["taux_alerte"]) < 0.05, (
        "semaine 3 : le taux d'alerte doit rester ≈ normal (la dérive est invisible sans labels)"
    )
    # Le modèle a appris l'ANCIEN régime de fraude : sur le nouveau, il fait pire
    # que le hasard (ROC-AUC < 0,5) — il regarde littéralement du mauvais côté.
    assert m3["roc_auc"] < 0.5, "semaine 3 : ROC-AUC attendue sous 0,5 (pire que le hasard)"
