# Script de Présentation - Comparaison DQN vs PPO sur CarRacing-v3

## Slide 1 : Titre
Bonjour à tous. Aujourd'hui, je vais vous présenter notre étude comparative de deux algorithmes d'apprentissage par renforcement profond : DQN et PPO, appliqués à l'environnement CarRacing-v3.

## Slide 2 : Motivation
Notre motivation principale est de répondre à une question fondamentale en apprentissage par renforcement : quelle est la meilleure approche entre les actions discrètes et les actions continues ?

Pour cela, nous avons choisi CarRacing-v3, un environnement de contrôle visuel qui supporte naturellement les deux représentations.

Nous cherchons à répondre à trois questions :
- Comment ces deux approches se comparent-elles en termes de performance ?
- Quelles sont leurs caractéristiques de stabilité ?
- Et quels conseils pratiques pouvons-nous donner aux praticiens ?

## Slide 3 : Environnement
L'environnement CarRacing-v3 est un simulateur de course où un agent contrôle une voiture sur un circuit aléatoire.

Comme vous pouvez le voir sur l'image, l'agent reçoit des images RGB de 96 par 96 pixels.

Pour DQN, nous utilisons 5 actions discrètes : ne rien faire, tourner à gauche, tourner à droite, accélérer, et freiner.

Pour PPO, nous utilisons un vecteur continu à 3 dimensions : le volant, l'accélérateur, et le frein.

Le système de récompense donne moins 0.1 point par image, plus 1000 divisé par N pour chaque case visitée.

Nous appliquons aussi un prétraitement : empilement de 4 images et saut de 2 images.

## Slide 4 : Espaces d'Actions
Voici une comparaison visuelle des deux espaces d'actions.

À gauche, les 5 actions discrètes de DQN, numérotées de 0 à 4.

À droite, les 3 actions continues de PPO, chacune avec sa plage de valeurs possible.

## Slide 5 : Architecture Partagée
Pour assurer une comparaison équitable, les deux algorithmes utilisent exactement la même architecture de réseau de neurones convolutif pour l'extraction de caractéristiques.

Nous avons adopté l'architecture Nature DQN : trois couches convolutives qui transforment l'image 96 par 96 en un vecteur de 4096 caractéristiques.

Seules les têtes finales diffèrent : DQN utilise deux couches fully-connected pour prédire les valeurs Q des 5 actions, tandis que PPO utilise des réseaux séparés pour l'acteur et le critique.

## Slide 6 : DQN vs PPO
Comparons maintenant les deux algorithmes en détail.

DQN est une méthode basée sur la valeur, avec apprentissage hors-politique. Il utilise un buffer de replay de 100 mille expériences et un réseau cible mis à jour tous les 1000 pas. L'exploration se fait avec epsilon-greedy, où epsilon diminue de 1.0 à 0.01.

PPO est une méthode de gradient de politique, avec apprentissage en-politique. Il collecte des rollouts de 2048 pas et effectue 10 époques de mises à jour avec un objectif clippé. Les avantages sont calculés avec GAE lambda égal 0.95.

## Slide 7 : Procédures d'Entraînement
Ce diagramme montre les différences dans les boucles d'entraînement.

DQN, à gauche, stocke toutes les transitions dans un buffer et échantillonne des batches aléatoires pour apprendre.

PPO, à droite, collecte un rollout complet, calcule les avantages, puis met à jour la politique plusieurs fois sur les mêmes données avant de les jeter.

## Slide 8 : Configuration Expérimentale
Notre configuration expérimentale comprend :
- 900 épisodes d'entraînement pour chaque agent
- Un critère de convergence : score moyen supérieur ou égal à 700 avec écart-type inférieur ou égal à 10
- Des sauvegardes tous les 100 épisodes
- Des évaluations de 10 épisodes tous les 100 épisodes d'entraînement
- Implémentation en PyTorch avec Gymnasium
- Des seeds aléatoires fixes pour la reproductibilité

## Slide 9 : Courbes d'Apprentissage
Voici les courbes d'apprentissage des deux agents sur 900 épisodes.

La ligne bleue représente DQN, la ligne verte représente PPO. Les lignes transparentes montrent les scores bruts, les lignes pleines montrent les moyennes mobiles.

On observe que PPO apprend plus rapidement au début, mais DQN atteint une meilleure performance finale et plus stable.

La ligne rouge en pointillés indique notre objectif de 700 points.

## Slide 10 : Résumé des Résultats
Les résultats clés sont les suivants :

Pour les 100 derniers épisodes :
- DQN obtient un score moyen de 877.86, avec un écart-type de seulement 40.24
- PPO obtient 752.43, avec un écart-type beaucoup plus élevé de 199.79

Sur l'ensemble de l'entraînement, les moyennes globales sont similaires : 573.95 pour DQN et 569.05 pour PPO.

Important : aucun des deux algorithmes n'a convergé en 900 épisodes selon nos critères stricts.

Mais DQN montre clairement une meilleure stabilité finale.

## Slide 11 : Analyse de Stabilité
Ces boîtes à moustaches illustrent la distribution des scores pour les 100 derniers épisodes.

Pour DQN, on voit une distribution très serrée autour de 880, avec peu de valeurs aberrantes.

Pour PPO, la distribution est beaucoup plus large, avec le quartile inférieur autour de 650 et plusieurs valeurs aberrantes en dessous de 400.

Cela confirme la supériorité de DQN en termes de stabilité.

## Slide 12 : Discussion
Pourquoi DQN a-t-il surpassé PPO ?

Premièrement, la discrétisation de l'espace d'actions. Les 5 actions discrètes sont suffisantes pour cette tâche et réduisent l'espace d'exploration.

Deuxièmement, l'apprentissage hors-politique. Le replay buffer permet d'apprendre de manière plus efficace à partir d'expériences passées diverses.

Troisièmement, un compromis intéressant sur la stabilité : PPO a une variance globale plus faible, 307 contre 372, mais DQN a une bien meilleure stabilité finale avec un écart-type de 40 contre 200.

## Slide 13 : Recommandations Pratiques
Quand utiliser chaque algorithme ?

Utilisez DQN quand :
- Les actions se décomposent naturellement en choix discrets
- L'efficacité d'échantillonnage est critique
- Vous avez besoin de stabilité dans la performance finale
- Vous travaillez avec des observations visuelles

L'insight clé : la discrétisation n'est pas toujours néfaste, elle peut simplifier l'exploration efficacement.

Utilisez PPO quand :
- Vous avez un contrôle continu de haute dimension
- Les approximations discrètes sont inadéquates
- Vous avez besoin de trajectoires lisses
- Vous disposez d'un budget d'entraînement plus long

L'insight clé : PPO apprend de manière plus cohérente au début, mais avec plus de variance dans les phases tardives.

## Slide 14 : Limitations et Travaux Futurs
Notre étude a certaines limitations :
- Recherche d'hyperparamètres limitée
- Un seul environnement testé
- Algorithmes vanilla sans variantes modernes

Les directions futures incluent :
- Tester sur plusieurs environnements
- Explorer des variantes comme Rainbow DQN ou PPG
- Investiguer SAC pour le contrôle continu
- Approches hybrides discrètes-continues
- Entraînement prolongé au-delà de 900 épisodes

## Slide 15 : Conclusion
En conclusion :

DQN a obtenu une performance finale supérieure : 877.86 contre 752.43 pour PPO.

DQN a démontré une meilleure stabilité avec un écart-type de 40.24 contre 199.79.

PPO a montré un apprentissage initial plus fluide mais une variance plus élevée plus tard.

La représentation de l'espace d'actions est un choix important qui impacte significativement la performance.

Les actions discrètes peuvent être très efficaces quand elles sont bien conçues.

Le choix doit être guidé par les caractéristiques de la tâche et les contraintes du projet.

## Slide 16 : Remerciements
Merci beaucoup pour votre attention.

Je serais ravi de répondre à vos questions.

---

# Notes pour la Présentation

**Durée estimée** : 10-12 minutes

**Conseils** :
- Parler clairement et pas trop vite
- Pointer les éléments importants sur les graphiques
- Maintenir un contact visuel avec l'audience
- Être prêt à expliquer les concepts techniques si nécessaire
- Avoir des exemples concrets pour illustrer les applications pratiques

**Points d'attention** :
- Bien prononcer "apprentissage par renforcement" (pas "renforcement learning")
- Expliquer "hors-politique" vs "en-politique" si demandé
- Préparer des réponses sur pourquoi CarRacing-v3 a été choisi
- Être capable d'expliquer le replay buffer et GAE simplement
