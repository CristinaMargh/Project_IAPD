# Ghid de prezentare (română)

## Phase 1

Arătăm 1.000 de melodii distincte și datele disponibile: artist, titlu, gen, an, link YouTube și vizualizări istorice. Explicăm că majoritatea atributelor de ML provin din metadate. Arătăm probleme **reale**: 149 genuri lipsă, 7 artiști lipsă și 43 valori extreme ale vizualizărilor brute. Separat, demonstrăm rezistența la missing values cu un experiment în care ascundem intenționat unele caracteristici într-o COPIE, etichetată corespunzător. Preprocesarea statistică se învață doar din train. Pentru final propunem regresie liniară, Ridge și Random Forest cu validare pe artiști diferiți.

## Phase 2

Explicăm splitul de 801 train / 199 test și faptul că nu există artiști comuni. Comparăm rezultatele cu un baseline și cu o variantă Random Forest ponderată. Cel mai bun model a fost Random Forest (test R² aproximativ 0,271; RMSE aproximativ 2,939 în unități logaritmice). Demonstrăm și scenariul cu date deteriorate controlat pe exact aceleași ID-uri de test (R² ~0,257). Discutăm erorile, importanța caracteristicilor și biasul de selecție al catalogului. Nu afirmăm că prezicem sigur succesul unei piese noi.

## Întrebări probabile

**De ce avem missing values artificiale?** Pentru un studiu controlat de robustețe; nu le confundăm cu erorile originale.

**De ce păstrăm outlierii din popularitate?** Hiturile extreme sunt reale; eliminarea lor ar schimba problema. Transformăm targetul cu log1p.

**De ce nu includem like-uri și views curente?** Acestea reflectă deja popularitatea și pot induce data leakage.

**De ce nu avem BPM/energy/key?** Nu sunt disponibile complet și licențiat în sursa folosită; nu inventăm valori.

**De ce RF nu are un R² mare?** Avem predictori slabi, în principal text și metadate, un eșantion mic și probabil nereprezentativ și vizualizări istorice măsurate la un moment necunoscut.
