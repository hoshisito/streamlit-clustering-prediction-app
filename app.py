import re
from pathlib import Path

import joblib as jb
import numpy as np
import streamlit as st
from sklearn.preprocessing import normalize

st.set_page_config(page_title="Clustering - 4 modèles")

DOSSIER = Path(__file__).parent / "examen"


# ---------------------------------------------------------------
# Fonctions utiles (communes aux 4 modèles)
# ---------------------------------------------------------------
@st.cache_resource
def charger(sous_dossier, fichier):
    """Charge un fichier .joblib du dossier examen."""
    return jb.load(DOSSIER / sous_dossier / (fichier + ".joblib"))


def classe_la_plus_proche(vecteur, centroides):
    """Distance entre le vecteur et chaque centroïde, puis numéro de la classe la plus proche."""
    distances = np.linalg.norm(centroides - vecteur, axis=1)
    return distances.argmin(), distances


def afficher_resultat(classe, distances, noms_classes):
    st.success("Classe prédite : " + noms_classes[int(classe)])
    with st.expander("Voir les distances aux centroïdes"):
        st.write("La classe choisie est celle dont le centroïde est le plus proche.")
        tableau = {"Classe": [noms_classes[i] + f" (n°{i})" for i in range(len(distances))],
                   "Distance": distances.round(4)}
        st.dataframe(tableau, hide_index=True)


# ---------------------------------------------------------------
# 1. Customer Segmentation
# ---------------------------------------------------------------
def page_customer():
    st.title("Segmentation des clients")
    st.write("Saisissez les caractéristiques d'un client pour connaître sa classe (CAH).")

    colonnes = charger("customer_segmentation", "colonnes_customer")
    medianes = charger("customer_segmentation", "medianes_customer")
    centroides = charger("customer_segmentation", "centroides_customer")
    noms_classes = charger("customer_segmentation", "noms_classes_customer")

    # Un champ numérique par variable, pré-rempli avec la médiane
    valeurs = []
    for nom, mediane in zip(colonnes, medianes):
        valeurs.append(st.number_input(nom, value=float(mediane)))

    if st.button("Prédire la classe", key="bouton_customer"):
        # Dans le notebook, les données ont été normalisées : on fait pareil
        client = normalize(np.array([valeurs]))[0]
        classe, distances = classe_la_plus_proche(client, centroides)
        afficher_resultat(classe, distances, noms_classes)


# ---------------------------------------------------------------
# 2. Wheat Seeds
# ---------------------------------------------------------------
def page_wheat():
    st.title("Graines de blé")
    st.write("Saisissez les mesures d'une graine pour connaître sa classe (CAH).")

    colonnes = charger("wheat_seeds", "colonnes_wheat")
    medianes = charger("wheat_seeds", "medianes_wheat")
    centroides = charger("wheat_seeds", "centroides_wheat")
    noms_classes = charger("wheat_seeds", "noms_classes_wheat")

    # Un champ numérique par mesure, pré-rempli avec la médiane
    valeurs = []
    for nom, mediane in zip(colonnes, medianes):
        valeurs.append(st.number_input(nom, value=float(mediane), step=0.01, format="%.4f"))

    if st.button("Prédire la classe", key="bouton_wheat"):
        # Pas de normalisation dans ce notebook : on compare directement aux centroïdes
        graine = np.array(valeurs)
        classe, distances = classe_la_plus_proche(graine, centroides)
        afficher_resultat(classe, distances, noms_classes)


# ---------------------------------------------------------------
# 3. Mushroom
# ---------------------------------------------------------------
def page_mushroom():
    st.title("Champignons")
    st.write("Choisissez les caractéristiques d'un champignon (codes d'origine du jeu de données).")

    variables = charger("mushroom", "variables_mushroom")  # noms des 21 variables
    uniques = charger("mushroom", "uniques_mushroom")  # modalités de chaque variable
    colonnes = charger("mushroom", "colonnes_mushroom")  # colonnes après get_dummies
    centroides = charger("mushroom", "centroides_mushroom")
    noms_classes = charger("mushroom", "noms_classes_mushroom")

    # Une liste déroulante par variable
    choix = []
    for variable, modalites in zip(variables, uniques):
        choix.append(st.selectbox(variable, modalites))

    if st.button("Prédire la classe", key="bouton_mushroom"):
        # On refait l'encodage get_dummies : une colonne "variable_modalité" vaut 1 si elle est choisie
        colonnes_choisies = [v + "_" + c for v, c in zip(variables, choix)]
        champignon = np.array([1 if col in colonnes_choisies else 0 for col in colonnes])

        classe, distances = classe_la_plus_proche(champignon, centroides)
        afficher_resultat(classe, distances, noms_classes)
        st.warning("Ne jamais consommer un champignon sur la base de cette prédiction : "
                   "une classe contient presque toujours des champignons vénéneux.")


# ---------------------------------------------------------------
# 4. PriceRunner
# ---------------------------------------------------------------
def nettoyer(texte):
    """Même fonction que dans le notebook (elle n'a pas pu être sauvegardée)."""
    texte = texte.lower()
    texte = re.sub(r"[^a-z0-9\s]", " ", texte)
    return re.sub(r"\s+", " ", texte).strip()


def page_pricerunner():
    st.title("Catégories de produits (PriceRunner)")
    st.write("Saisissez le titre d'un produit (en anglais) pour connaître sa catégorie (CAH).")

    tfidf = charger("price_runner", "tfidf_pricerunner")
    centroides = charger("price_runner", "centroides_pricerunner")
    noms_classes = charger("price_runner", "noms_classes_pricerunner")

    titre = st.text_input("Titre du produit", value="samsung galaxy s8 64gb")

    if st.button("Prédire la classe", key="bouton_pricerunner"):
        # Nettoyer le titre, puis le transformer en vecteur de 300 mots (TF-IDF)
        vecteur = tfidf.transform([nettoyer(titre)]).toarray()[0]

        if vecteur.sum() == 0:
            st.warning("Aucun mot de ce titre n'est connu du modèle : essayez avec plus de détails.")
        else:
            classe, distances = classe_la_plus_proche(vecteur, centroides)
            afficher_resultat(classe, distances, noms_classes)


# ---------------------------------------------------------------
# Menu : une interface par modèle
# ---------------------------------------------------------------
modele = st.sidebar.radio("Choisissez un modèle",
                          ["Customer Segmentation", "Wheat Seeds", "Mushroom", "PriceRunner"])

if modele == "Customer Segmentation":
    page_customer()
elif modele == "Wheat Seeds":
    page_wheat()
elif modele == "Mushroom":
    page_mushroom()
else:
    page_pricerunner()
