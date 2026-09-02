# Drift spec — PayGuard v1

## 1. Référence
- Jeu de référence : `data/reference.csv` (20 000 transactions labellisées, split d'entraînement/validation du Sprint 2), figé au moment du gel du modèle (`scripts/train_model.py`, `models/threshold.json`). À remplacer uniquement lors d'un réentraînement officiel (nouvelle validation, nouveau seuil gelé), jamais en douce sur une simple dérive constatée.

## 2. Fenêtre courante & fréquence
- Fenêtre : 1 semaine glissante de production (≈ 3 000 transactions). Fréquence de calcul : hebdomadaire (covariate monitoring, sans attendre les labels). Taille minimale d'échantillon : ≥ 1 000 transactions pour que PSI/KS restent stables.

## 3. Features surveillées & tests
| Feature | Test(s) | Seuil d'alerte | Justification |
|---|---|---|---|
| montant_eur | PSI + KS | PSI > 0,25 = alerte forte, 0,10–0,25 = à surveiller | Feature à forte contribution ; sensible aux opérations commerciales (Vente Flash) |
| heure | PSI + KS | PSI > 0,25 ; KS lu à titre indicatif seulement | Grand échantillon → KS quasi toujours significatif (p ≈ 1e-33 en semaine 1 déjà) sans dérive métier réelle : le PSI prime |
| anciennete_client_j | PSI + KS | PSI > 0,25 | Sensible à l'acquisition de nouveaux clients (campagnes) |
| nb_articles | PSI + KS | PSI > 0,25 | Sensible au comportement d'achat (panier promo) |
| distance_domicile_km | PSI + KS | PSI > 0,25 | Peu discriminante en semaine 2 mais conservée pour couverture complète |

## 4. KPI métier (second rideau, dès que les labels tombent)
- KPI : rappel (fraude) au seuil gelé. Plancher déclencheur : rappel < 0,70 (référence validation : 0,817). Délai de disponibilité : plusieurs semaines après la transaction (le temps que les fraudes soient confirmées).

## 5. Réactions au franchissement
- PSI > 0,25 sur ≥ 2 features → alerte « à investiguer », rapprocher d'un événement métier connu (campagne, saisonnalité) avant toute action sur le modèle.
- Rappel < plancher → alerte critique : geler les décisions automatiques si possible, lancer un réentraînement/recalibrage, ne pas se contenter de baisser le seuil (le seuil ne corrige pas un modèle qui a mal appris — cf. semaine 3, ROC-AUC = 0,278 < 0,5).
- Qui est prévenu : équipe risque (délai 24 h sur alerte PSI) et équipe data/ML (délai immédiat sur franchissement du plancher de rappel), par le canal de suivi hebdomadaire (`reports/suivi_semaines.csv` + revue).

## 6. Limites connues du dispositif
- Le monitoring des entrées (PSI/KS) est **structurellement aveugle au concept drift** : semaine 3 le prouve — PSI RAS sur les 5 features (max 0,008) alors que le rappel s'effondre à 0,072 et la ROC-AUC tombe à 0,278. P(X) inchangée n'implique pas P(Y|X) inchangée.
- KS est très sensible à la taille d'échantillon : avec 3 000 transactions, il peut déclarer une dérive « significative » (p très faible) pour un écart minuscule sans intérêt métier (cas `heure`, PSI = 0,087 mais p ≈ 1,4e-33) — ne jamais piloter une alerte sur la seule p-value.
