# PayGuard — Data drift adversarial — Sprint 3 CISIA 24/08/2026

TP d'application du **module 31 CISIA** (*Data drift & métriques de dérive — concepts*, S3-M09, US3.5),
**indépendant du fil rouge InduSense** : surveillance d'un modèle anti-fraude de paiement en production.
Données 100 % synthétiques (générateur seedé fourni côté formateur) — aucune donnée réelle, aucun téléchargement.

Ce dépôt public contient uniquement le **starter apprenant** : données synthétiques, consignes,
scripts à compléter et tests. Il ne contient ni corrigé, ni réponse, ni solution formateur.

Suivez **`TP_apprenant_payguard_drift.md`** ; l'installation, les commandes et les preuves attendues
y sont détaillées pas à pas.

## Cloner et créer sa branche

Dans **VS Code**, ouvrez `Terminal > Nouveau terminal`, choisissez **PowerShell**, puis saisissez :

```powershell
cd "$HOME\Documents"
git clone https://github.com/thomasfesq/CISIA_24082026_PayGuard.git
cd .\CISIA_24082026_PayGuard
git switch -c prenom-nom
$env:PYTHONUTF8 = "1"
uv sync --frozen --extra dev
uv run python -m pytest -q
uv run python scripts/train_model.py
```

Au départ, la suite pédagogique affiche volontairement **10 échecs, 1 réussite et 1 test ignoré** :
ce premier contrôle se fait avant l'entraînement, car un test d'intégration attend encore l'artefact
modèle. Les deux scripts principaux sont à compléter. Sur votre branche, la CI devient verte lorsque les
12 tests sont réussis. Le tag `v1.0-starter-20260824` permet de comparer votre travail à l'état initial.

## Installation (résumé — détails dans le TP, étape 0)

Prérequis de la session : **Python 3.13** (fichier `.python-version`). Voie recommandée CISIA : **uv**.

Sous Windows, décompresser dans un chemin court, par exemple
`%USERPROFILE%\CISIA\S3\tp_autonome_drift`. Un chemin très profond peut empêcher la matérialisation d'une
extension compilée de scikit-learn malgré un message de sync réussi. Dans ce cas, repartir d'une
décompression neuve dans ce chemin court et rejouer `uv sync --frozen --extra dev` — ne pas muter le lock.

| | Windows (PowerShell) | macOS (Terminal) |
|---|---|---|
| Installer uv (si absent) | `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 \| iex"` | `curl -LsSf https://astral.sh/uv/install.sh \| sh` (ou `brew install uv`) |
| Environnement | `uv sync --frozen --extra dev` | `uv sync --frozen --extra dev` |
| Vérification | `uv run python -c "import sklearn; print('OK')"` | idem |
| Repli sans uv | `py -3 -m venv .venv` · `.venv\Scripts\Activate.ps1` · `pip install -r requirements.txt` | `python3 -m venv .venv` · `source .venv/bin/activate` · `pip install -r requirements.txt` |

Sous Windows, poser une fois `$env:PYTHONUTF8 = "1"` dans le terminal du TP avant les scripts : leurs
messages pédagogiques contiennent des symboles Unicode que certaines consoles historiques ne savent pas afficher.

Le premier `uv sync --frozen --extra dev` crée `.venv/` à partir de `uv.lock` **sans recalculer le verrou** et installe aussi pytest (versions bornées dans `pyproject.toml` :
numpy, pandas, scipy, scikit-learn, matplotlib, joblib ; pytest via `--extra dev` ;
Evidently 0.7.x en option via `--extra evidently` pour le bonus — même famille de version que le lock InduSense).

## Structure

```
tp_autonome_drift/
├── README.md                        ← ce fichier
├── TP_apprenant_payguard_drift.md   ← énoncé pas à pas + questions (SANS réponses)
├── pyproject.toml / requirements.txt
├── data/
│   ├── reference.csv                ← 20 000 transactions labellisées (entraînement)
│   ├── courant_semaine{1,2,3}.csv   ← 3 semaines de production, SANS labels
│   └── labels/                      ← labels « arrivés en retard » (étape 6 uniquement)
├── scripts/
│   ├── train_model.py               💻 fourni — baseline Sprint 2 + seuil gelé par le coût
│   ├── drift_lab.py                 🧑‍🎓 starter à compléter — PSI, KS, table de dérive
│   ├── evaluate_semaine.py          🧑‍🎓 starter à compléter — métriques métier au seuil gelé
│   └── bonus_evidently.py           💻 fourni — bonus facultatif Evidently
├── tests/test_drift.py              💻 definition of done : 12 tests (8 unitaires + 4 intégration)
├── models/ · reports/               ← remplis par les scripts (vides au départ)
└── .github/workflows/tests.yml      💻 contrôle du starter sur main, 12 tests sur les branches
```

## Parcours en 30 secondes

```bash
uv run python scripts/train_model.py            # 1. geler modèle + seuil (fourni)
# compléter scripts/drift_lab.py, puis :
uv run python -m pytest -k "psi or ks or verdict or table"  # 2. 8 tests unitaires verts
uv run python scripts/drift_lab.py --semaine 1    # 3-5. auditer les 3 semaines (sans labels)
# compléter scripts/evaluate_semaine.py, puis :
uv run python scripts/evaluate_semaine.py --semaine 1   # 6. les labels arrivent...
uv run python -m pytest                          # fin d'étape 6 — definition of done : 12 verts
# étape 7 : rédiger reports/drift_spec_payguard.md (gabarit dans le TP)
```

Vérification du starter : l'entraînement fourni réussit, puis la suite initiale donne exactement
`10 failed, 1 passed, 1 skipped`. Après réalisation complète, elle doit donner `12 passed`.

## Observabilité

L'observabilité active est traitée en **M33-M34 sur InduSense**, avec une dépendance scellée et la stack
du repo fil rouge. Les anciens fichiers `observabilite/` et `scripts/export_metrics.py` sont conservés
uniquement dans la réserve formateur pour provenance ; ils ne font pas partie du parcours apprenant ni
du zip distribué, et aucune commande `--extra obs` ne doit être lancée.

## Intégration continue

`.github/workflows/tests.yml` (m24) sépare deux contrats :

- sur `main`, il vérifie que le dépôt reste le starter certifié et que l'entraînement fonctionne ;
- sur une branche apprenant ou une pull request, il exige les **12 tests réussis**.

Les fichiers `scripts/drift_lab.py` et `scripts/evaluate_semaine.py` de `main` restent donc les
starters. Travaillez toujours sur votre branche et ouvrez une pull request seulement lorsque vos
preuves sont vertes.

## Liens parcours

Concepts : module **31** (PSI, KS, covariate vs concept drift, drift spec, KPI technique vs métier).
Révisions embarquées : Sprint 2 **m11** (accuracy vs rappel, PR-AUC, coût FN≫FP), **m21** (ROC/PR,
seuil par coût gelé sur la validation), **m22** (model card). Débouche sur le module **32**
(automatisation : rapport, alerting, anti-bruit) — la drift spec produite ici s'y branche telle quelle.
